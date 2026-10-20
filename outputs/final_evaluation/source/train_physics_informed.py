"""Joint sample-dependent-exponent learning with group-disjoint cross-validation."""
import argparse
import copy
import hashlib
import itertools
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from preprocess import FEATURES, load_workbook
from create_partitions import validate_assignments
from train_data_driven import build_network, input_values, rmsle_base10


class PhysicsInformedNetwork(torch.nn.Module):
    def __init__(self, candidate, config):
        super().__init__()
        self.predictor = build_network(config["backbone"])
        self.m_indices = list(range(9)) if candidate["m_inputs"] == "all" else [0, 1]
        incoming = len(self.m_indices)
        layers = []
        for width in candidate["m_hidden_sizes"]:
            layer = torch.nn.Linear(incoming, width)
            torch.nn.init.xavier_uniform_(layer.weight, gain=torch.nn.init.calculate_gain("tanh"))
            torch.nn.init.zeros_(layer.bias)
            layers.extend([layer, torch.nn.Tanh()])
            incoming = width
        final = torch.nn.Linear(incoming, 1)
        torch.nn.init.xavier_uniform_(final.weight)
        torch.nn.init.zeros_(final.bias)
        self.exponent = torch.nn.Sequential(*layers, final, torch.nn.Softplus())

    def forward(self, x):
        return self.predictor(x).squeeze(-1), self.exponent(x[:, self.m_indices]).squeeze(-1)


def physical_log10(rows, m):
    # Raw x1 and x2 retain the supplied numerical units, independently of scaling.
    log_x1 = torch.as_tensor(np.log10(rows.x1.to_numpy()), dtype=m.dtype)
    log_x2 = torch.as_tensor(np.log10(rows.x2.to_numpy()), dtype=m.dtype)
    return 2 * log_x2 - np.log10(72.2) + m * log_x1


def exponent_statistics(m):
    return {"m_mean": float(np.mean(m)), "m_median": float(np.median(m)),
        "m_std": float(np.std(m)), "m_q10": float(np.quantile(m, .1)),
        "m_q90": float(np.quantile(m, .9)), "m_min": float(np.min(m)), "m_max": float(np.max(m))}


def physics_fit(fitting, validation, candidate, config, seed, final_epochs=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    backbone = config["backbone"]
    scaler = StandardScaler().fit(input_values(fitting, backbone, config))
    x = torch.tensor(scaler.transform(input_values(fitting, backbone, config)), dtype=torch.float32)
    y = torch.tensor(np.log10(fitting.y.to_numpy()), dtype=torch.float32)
    # Precompute fixed physical coefficients, not sample-dependent exponents.
    intercept = torch.tensor(2 * np.log10(fitting.x2.to_numpy()) - np.log10(72.2), dtype=torch.float32)
    slope = torch.tensor(np.log10(fitting.x1.to_numpy()), dtype=torch.float32)
    if validation is not None:
        xv = torch.tensor(scaler.transform(input_values(validation, backbone, config)), dtype=torch.float32)
        yv = torch.tensor(np.log10(validation.y.to_numpy()), dtype=torch.float32)
    elif final_epochs is None:
        raise ValueError("Training without validation requires a preselected duration.")
    model = PhysicsInformedNetwork(candidate, config)
    opt = config["optimizer"]
    optimizer = torch.optim.Adam(model.parameters(), lr=candidate["learning_rate"],
        eps=opt["epsilon"], betas=(opt["beta_1"], opt["beta_2"]))
    controls = config["training"]
    best_score = float("inf")
    anchor = float("inf")
    stale = 0
    history = []
    started = time.perf_counter()
    limit = final_epochs if final_epochs is not None else controls["max_epochs"]
    for epoch in range(1, limit + 1):
        model.train()
        for idx in torch.randperm(len(x)).split(backbone["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            z, m = model(x[idx])
            data_loss = torch.mean((z - y[idx]) ** 2)
            physics_loss = torch.mean((z - (intercept[idx] + m * slope[idx])) ** 2)
            penalty = sum(p.square().sum() for name, p in model.named_parameters() if name.endswith("weight"))
            objective = data_loss + candidate["lambda_physics"] * physics_loss + backbone["l2"] * penalty
            if not torch.isfinite(objective):
                raise ValueError("Nonfinite joint objective.")
            objective.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            z, m = model(x)
            physical = intercept + m * slope
            row = {"epoch": epoch,
                "training_rmsle": rmsle_base10(fitting.y.to_numpy(), z.numpy()),
                "training_data_loss": float(torch.mean((z - y) ** 2)),
                "training_physics_loss": float(torch.mean((z - physical) ** 2)),
                **exponent_statistics(m.numpy())}
            if not torch.isfinite(m).all() or not (m > 0).all():
                raise ValueError("The exponent must be finite and strictly positive.")
            if validation is not None:
                zv, mv = model(xv)
                row.update({"validation_rmsle": rmsle_base10(validation.y.to_numpy(), zv.numpy()),
                    "validation_data_loss": float(torch.mean((zv - yv) ** 2)),
                    "validation_physics_loss": float(torch.mean((zv - physical_log10(validation, mv)) ** 2))})
        history.append(row)
        if validation is not None:
            score = row["validation_rmsle"]
            if score < best_score:
                best_score = score
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                best_row = row.copy()
            if score < anchor - controls["min_delta"]:
                anchor = score
                stale = 0
            else:
                stale += 1
            if stale >= controls["patience"]:
                break
    if validation is None:
        best_epoch = epoch
        best_row = history[-1]
    else:
        model.load_state_dict(best_state)
    elapsed = time.perf_counter() - started
    checkpoint = {"state_dict": model.state_dict(), "candidate": candidate, "configuration": config,
        "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
        "seed": seed, "epochs": best_epoch, "log_features": config["log_features"],
        "target_transform": "log10"}
    predictions = None
    if validation is not None:
        with torch.no_grad():
            z, m = model(xv)
        if not np.isclose(rmsle_base10(validation.y.to_numpy(), z.numpy()), best_score, rtol=1e-6):
            raise ValueError("Restored physics checkpoint differs from the selected score.")
        predictions = pd.DataFrame({"ID": validation.ID.to_numpy(), "observed_y": validation.y.to_numpy(),
            "predicted_y": np.power(10., z.numpy().astype(float)), "m": m.numpy(),
            "physical_log10_y": physical_log10(validation, m).numpy()})
    result = {"best_epoch": best_epoch, "epochs_run": epoch, "seconds": elapsed,
        "hit_epoch_limit": epoch == controls["max_epochs"] if final_epochs is None else False,
        **{k: v for k, v in best_row.items() if k != "epoch"}}
    return model, scaler, pd.DataFrame(history), checkpoint, predictions, result


def validate_physics_implementation(training, config):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config["seed"])
    candidate = {"m_hidden_sizes": [8], "m_inputs": "all", "learning_rate": .001, "lambda_physics": .1}
    model = PhysicsInformedNetwork(candidate, config)
    subset = training.iloc[:32]
    scaler = StandardScaler().fit(input_values(subset, config["backbone"], config))
    x = torch.tensor(scaler.transform(input_values(subset, config["backbone"], config)), dtype=torch.float32)
    z, m = model(x)
    physical = physical_log10(subset, m)
    # Independently evaluate the original equation for fixed positive exponents.
    fixed = np.linspace(.2, 3., len(subset))
    direct = subset.x2.to_numpy() ** 2 / (72.2 * subset.x1.to_numpy() ** (-fixed))
    analytic = physical_log10(subset, torch.tensor(fixed, dtype=torch.float64)).numpy()
    np.testing.assert_allclose(np.log10(direct), analytic, rtol=1e-12, atol=1e-12)
    m.retain_grad()
    residual = torch.mean((z - physical_log10(subset, m)) ** 2)
    residual.backward()
    predictor_grad = sum(float(p.grad.abs().sum()) for p in model.predictor.parameters() if p.grad is not None)
    exponent_grad = sum(float(p.grad.abs().sum()) for p in model.exponent.parameters() if p.grad is not None)
    if predictor_grad <= 0 or exponent_grad <= 0 or float(m.grad.abs().sum()) <= 0:
        raise ValueError("Physics gradients must reach predictor, exponent network, and m.")
    rmsle_base10(subset.y.to_numpy(), z.detach().numpy())
    if not torch.isfinite(m).all() or not (m > 0).all():
        raise ValueError("Invalid exponent.")
    return {"original_equation_matches_log_form": True, "positive_finite_predictions_and_m": True,
        "predictor_physics_gradient_l1": predictor_grad, "exponent_physics_gradient_l1": exponent_grad,
        "m_physics_gradient_l1": float(m.grad.abs().sum()), "test_data_used": False}


def save_physics_table(records, output):
    results = pd.DataFrame(records)
    rows = []
    for cid, group in results.groupby("config_id", sort=True):
        if len(group) != 5 or set(group.fold) != set(range(1, 6)):
            raise ValueError("Five completed folds required per physics candidate.")
        first = group.iloc[0]
        row = {"config_id": int(cid), "m_hidden_sizes": first.m_hidden_sizes,
            "m_inputs": first.m_inputs, "learning_rate": first.learning_rate,
            "lambda_physics": first.lambda_physics}
        row.update({f"fold_{int(r.fold)}_rmsle": r.validation_rmsle for r in group.itertuples()})
        row.update({"mean_validation_rmsle": group.validation_rmsle.mean(),
            "std_validation_rmsle": group.validation_rmsle.std(ddof=1),
            "validation_physics_rmse_log10": float(np.sqrt(group.validation_physics_loss.mean())),
            "mean_training_m_at_best": group.m_mean.mean(),
            "total_training_seconds": group.seconds.sum(),
            "folds_hitting_epoch_limit": int(group.hit_epoch_limit.sum())})
        rows.append(row)
    table = pd.DataFrame(rows).sort_values(["mean_validation_rmsle", "config_id"]).reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    table["selected"] = table["rank"] == 1
    table.to_csv(output / "configuration_results.csv", index=False)
    headers = list(table.columns)
    lines = ["# Physics-informed five-fold validation results", "",
        "Selection uses mean base-10 RMSLE with log(1+y). Standard deviation uses ddof=1. Test data are not evaluated.", "",
        "| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in table.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(f"{x:.6g}" if isinstance(x, float) else str(x) for x in row) + " |")
    (output / "configuration_results.md").write_text("\n".join(lines) + "\n")
    return table


def run_physics_search(workbook, assignments_path, config, output):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output = Path(output)
    for name in ["histories", "checkpoints", "oof_predictions", "source"]:
        (output / name).mkdir(parents=True, exist_ok=True)
    df = load_workbook(workbook)
    a = pd.read_csv(assignments_path, dtype={"validation_fold": "Int64"})
    if a.ID.duplicated().any() or set(a.ID) != set(df.ID):
        raise ValueError("Partition IDs do not match workbook.")
    a = a.set_index("ID").loc[df.ID].reset_index()
    if not np.array_equal(a.predictor_group, df.predictor_group):
        raise ValueError("Predictor groups differ from saved assignments.")
    validate_assignments(df, a)
    training = df.loc[a.partition == "train"].copy()
    training["validation_fold"] = a.loc[a.partition == "train", "validation_fold"]
    checks = validate_physics_implementation(training, config)
    checks["training_observations_above_positive_m_upper_bound"] = int((training.y >= training.x2 ** 2 / 72.2).sum())
    checks["training_observations"] = len(training)
    (output / "implementation_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    keys = list(config["search_space"])
    grid = list(itertools.product(*(config["search_space"][k] for k in keys)))
    indices = np.random.default_rng(config["seed"]).choice(len(grid), config["n_configs"], replace=False)
    candidates = [{"config_id": i + 1, **dict(zip(keys, grid[int(idx)]))} for i, idx in enumerate(indices)]
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest() if "__file__" in globals() else "embedded_notebook"
    metadata = {"configuration": config, "candidates": candidates,
        "workbook_sha256": hashlib.sha256(Path(workbook).read_bytes()).hexdigest(),
        "assignments_sha256": hashlib.sha256(Path(assignments_path).read_bytes()).hexdigest(),
        "implementation_sha256": source_hash,
        "torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version(),
        "device": "cpu", "threads": 1, "test_evaluated": False}
    path = output / "search_manifest.json"
    if path.exists() and json.loads(path.read_text()) != metadata:
        raise ValueError("Existing physics results differ; choose a new output directory.")
    path.write_text(json.dumps(metadata, indent=2) + "\n")
    if "__file__" in globals():
        (output / "source/train_physics_informed.py").write_bytes(Path(__file__).read_bytes())
    records = []
    started = time.perf_counter()
    for candidate in candidates:
        for fold in range(1, 6):
            stem = f"config_{candidate['config_id']:02d}_fold_{fold}"
            path = output / "histories" / f"{stem}.json"
            if path.exists():
                record = json.loads(path.read_text())
                if not all((output / sub / f"{stem}{ext}").exists() for sub, ext in
                    [("histories", ".csv"), ("checkpoints", ".pt"), ("oof_predictions", ".csv")]):
                    raise ValueError("Incomplete physics cache.")
            else:
                fitting = training.loc[training.validation_fold != fold]
                validation = training.loc[training.validation_fold == fold]
                if set(fitting.predictor_group) & set(validation.predictor_group):
                    raise ValueError("Predictor-group leakage.")
                _, _, history, checkpoint, predictions, result = physics_fit(
                    fitting, validation, candidate, config, config["seed"] + fold)
                predictions["fold"] = fold
                history.to_csv(output / "histories" / f"{stem}.csv", index=False)
                torch.save(checkpoint, output / "checkpoints" / f"{stem}.pt")
                predictions.to_csv(output / "oof_predictions" / f"{stem}.csv", index=False)
                record = {**candidate, "fold": fold, **result}
                path.write_text(json.dumps(record, indent=2) + "\n")
            records.append(record)
            pd.DataFrame(records).to_csv(output / "fold_results.csv", index=False)
            print(f"Physics config {candidate['config_id']:02d}/{len(candidates)}, fold {fold}: "
                f"RMSLE={record['validation_rmsle']:.6f}, best epoch={record['best_epoch']}, "
                f"epochs={record['epochs_run']}", flush=True)
    table = save_physics_table(records, output)
    winner = int(table.iloc[0].config_id)
    selected = next(c for c in candidates if c["config_id"] == winner)
    results = pd.DataFrame(records)
    winning = results[results.config_id == winner].sort_values("fold")
    selection = {"candidate": selected, "configuration": config,
        "mean_validation_rmsle": float(table.iloc[0].mean_validation_rmsle),
        "std_validation_rmsle": float(table.iloc[0].std_validation_rmsle),
        "fold_best_epochs": winning.best_epoch.tolist(),
        "proposed_final_refit_epochs": int(np.median(winning.best_epoch)),
        "refit_rule": "Median of five selected best epochs; no test data used",
        "training_seconds_total": float(results.seconds.sum()),
        "search_invocation_seconds": time.perf_counter() - started,
        "completed_fits": len(records), "test_evaluated": False,
        "selected_folds_hitting_epoch_limit": int(winning.hit_epoch_limit.sum())}
    (output / "selected_configuration.json").write_text(json.dumps(selection, indent=2) + "\n")
    oof = pd.concat([pd.read_csv(output / "oof_predictions" / f"config_{winner:02d}_fold_{fold}.csv")
        for fold in range(1, 6)], ignore_index=True)
    if set(oof.ID) != set(training.ID) or oof.ID.duplicated().any():
        raise ValueError("Physics OOF coverage error.")
    oof.to_csv(output / "selected_oof_predictions.csv", index=False)
    print(json.dumps(selection, indent=2), flush=True)
    return table, selection


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path)
    parser.add_argument("--assignments", type=Path, default=Path("outputs/partitions/partition_assignments.csv"))
    parser.add_argument("--config", type=Path, default=Path("configs/physics_informed_search.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/physics_informed_search"))
    args = parser.parse_args()
    run_physics_search(args.workbook, args.assignments, json.loads(args.config.read_text()), args.output)
