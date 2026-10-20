"""Reproducible five-fold random search; the test partition is never evaluated."""
import argparse
import copy
import hashlib
import itertools
import json
import platform
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.preprocessing import StandardScaler

from preprocess import FEATURES, load_workbook
from create_partitions import validate_assignments


def rmsle_base10(y, log_prediction):
    # Invert log10(y) outputs without clipping; calculate the metric in float64.
    with np.errstate(over="raise", under="raise", invalid="raise"):
        prediction = np.power(10.0, np.asarray(log_prediction, dtype=np.float64))
    if not np.isfinite(prediction).all() or (prediction <= 0).any():
        raise ValueError("Invalid predictions: no silent clipping is permitted.")
    return float(np.sqrt(np.mean((np.log10(1.0 + prediction) - np.log10(1.0 + y)) ** 2)))


def input_values(rows, candidate, config):
    values = rows[FEATURES].copy()
    if candidate["log_inputs"]:
        names = config["log_features"]
        if (values[names] <= 0).any().any():
            raise ValueError("Selected logarithmic inputs must be positive.")
        values[names] = np.log10(values[names])
    return values


def build_network(candidate):
    layers = []
    incoming = len(FEATURES)
    for _ in range(candidate["n_layers"]):
        layer = torch.nn.Linear(incoming, candidate["width"])
        if candidate["activation"] == "relu":
            torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
        else:
            torch.nn.init.xavier_uniform_(layer.weight, gain=torch.nn.init.calculate_gain("tanh"))
        torch.nn.init.zeros_(layer.bias)
        layers.extend([layer, torch.nn.ReLU() if candidate["activation"] == "relu" else torch.nn.Tanh()])
        incoming = candidate["width"]
    output = torch.nn.Linear(incoming, 1)
    torch.nn.init.xavier_uniform_(output.weight)
    torch.nn.init.zeros_(output.bias)
    return torch.nn.Sequential(*layers, output)


def fit_fold(fitting, validation, candidate, config, fold):
    # The same fold seed is used for every candidate; all randomness is explicit.
    seed = config["seed"] + fold
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    scaler = StandardScaler().fit(input_values(fitting, candidate, config))
    x_fit = torch.tensor(scaler.transform(input_values(fitting, candidate, config)), dtype=torch.float32)
    x_val = torch.tensor(scaler.transform(input_values(validation, candidate, config)), dtype=torch.float32)
    y_fit_raw = fitting.y.to_numpy(dtype=float)
    y_val_raw = validation.y.to_numpy(dtype=float)
    y_fit = torch.tensor(np.log10(y_fit_raw), dtype=torch.float32)
    y_val = torch.tensor(np.log10(y_val_raw), dtype=torch.float32)
    model = build_network(candidate)
    optimizer_config = config["optimizer"]
    optimizer = torch.optim.Adam(model.parameters(), lr=candidate["learning_rate"],
        eps=optimizer_config["epsilon"], betas=(optimizer_config["beta_1"], optimizer_config["beta_2"]))
    controls = config["training"]
    best_score = float("inf")
    patience_anchor = float("inf")
    stale = 0
    history = []
    start = time.perf_counter()
    for epoch in range(1, controls["max_epochs"] + 1):
        model.train()
        permutation = torch.randperm(len(x_fit))
        for indices in permutation.split(candidate["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x_fit[indices]).squeeze(-1)
            loss = torch.mean((prediction - y_fit[indices]) ** 2)
            # Explicit L2 on weights only, not biases; no separate weight decay.
            penalty = sum(p.square().sum() for name, p in model.named_parameters() if name.endswith("weight"))
            objective = loss + candidate["l2"] * penalty
            if not torch.isfinite(objective):
                raise ValueError("Nonfinite training objective.")
            objective.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            fit_prediction = model(x_fit).squeeze(-1).numpy()
            val_prediction = model(x_val).squeeze(-1).numpy()
        training_score = rmsle_base10(y_fit_raw, fit_prediction)
        validation_score = rmsle_base10(y_val_raw, val_prediction)
        history.append({"epoch": epoch, "training_rmsle": training_score,
            "validation_rmsle": validation_score,
            "training_log10_mse": float(np.mean((fit_prediction - y_fit.numpy()) ** 2)),
            "validation_log10_mse": float(np.mean((val_prediction - y_val.numpy()) ** 2))})
        # Checkpoint the absolute minimum; min_delta controls patience only.
        if validation_score < best_score:
            best_score = validation_score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_training_score = training_score
        if validation_score < patience_anchor - controls["min_delta"]:
            patience_anchor = validation_score
            stale = 0
        else:
            stale += 1
        if stale >= controls["patience"]:
            break
    model.load_state_dict(best_state)
    with torch.no_grad():
        restored = model(x_val).squeeze(-1).numpy()
    if not np.isclose(rmsle_base10(y_val_raw, restored), best_score, rtol=1e-6):
        raise ValueError("Restored checkpoint does not match its recorded score.")
    checkpoint = {"state_dict": best_state, "candidate": candidate,
        "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
        "log_features": config["log_features"] if candidate["log_inputs"] else [],
        "target_transform": config["target_transform"], "fold": fold, "seed": seed}
    result = {"fold": fold, "validation_rmsle": best_score,
        "training_rmsle_at_best_epoch": best_training_score,
        "best_epoch": best_epoch, "epochs_run": epoch,
        "hit_epoch_limit": epoch == controls["max_epochs"],
        "seconds": time.perf_counter() - start}
    predictions = pd.DataFrame({"ID": validation.ID.to_numpy(), "fold": fold,
        "observed_y": y_val_raw, "predicted_y": np.power(10.0, restored.astype(float))})
    return result, pd.DataFrame(history), checkpoint, predictions


def run_search(workbook, assignments_path, config, output):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output = Path(output)
    for directory in [output, output / "histories", output / "checkpoints", output / "oof_predictions"]:
        directory.mkdir(parents=True, exist_ok=True)
    df = load_workbook(workbook)
    assignments = pd.read_csv(assignments_path, dtype={"validation_fold": "Int64"})
    # Reorder by ID and verify the saved groups still match the supplied workbook.
    if assignments.ID.duplicated().any() or set(assignments.ID) != set(df.ID):
        raise ValueError("Saved assignments do not match workbook IDs.")
    assignments = assignments.set_index("ID").loc[df.ID].reset_index()
    if not np.array_equal(assignments.predictor_group, df.predictor_group):
        raise ValueError("Saved predictor groups do not match the workbook.")
    validate_assignments(df, assignments)
    training = df.loc[assignments.partition == "train"].copy()
    training["validation_fold"] = assignments.loc[assignments.partition == "train", "validation_fold"]
    keys = list(config["search_space"])
    grid = list(itertools.product(*(config["search_space"][key] for key in keys)))
    sampled = np.random.default_rng(config["seed"]).choice(len(grid), config["n_configs"], replace=False)
    candidates = [{"config_id": i + 1, **dict(zip(keys, grid[int(index)]))} for i, index in enumerate(sampled)]
    metadata = {"configuration": config, "candidates": candidates,
        "workbook_sha256": hashlib.sha256(Path(workbook).read_bytes()).hexdigest(),
        "assignments_sha256": hashlib.sha256(Path(assignments_path).read_bytes()).hexdigest(),
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest() if "__file__" in globals() else "embedded_notebook",
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "device": "cpu", "threads": 1,
        "test_evaluated": False}
    metadata_path = output / "search_manifest.json"
    if metadata_path.exists() and json.loads(metadata_path.read_text()) != metadata:
        raise ValueError("Existing results belong to a different run. Use a new output directory.")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    records = []
    started = time.perf_counter()
    for candidate in candidates:
        for fold in range(1, config["n_folds"] + 1):
            stem = f"config_{candidate['config_id']:02d}_fold_{fold}"
            record_path = output / "histories" / f"{stem}.json"
            if record_path.exists():
                record = json.loads(record_path.read_text())
                required = [output / "histories" / f"{stem}.csv", output / "checkpoints" / f"{stem}.pt",
                    output / "oof_predictions" / f"{stem}.csv"]
                if not all(p.exists() for p in required):
                    raise ValueError("Incomplete cached fold outputs; use a new output directory.")
            else:
                fitting = training.loc[training.validation_fold != fold]
                validation = training.loc[training.validation_fold == fold]
                if set(fitting.predictor_group) & set(validation.predictor_group):
                    raise ValueError("Predictor group leakage.")
                result, history, checkpoint, predictions = fit_fold(fitting, validation, candidate, config, fold)
                record = {**candidate, **result}
                history.to_csv(output / "histories" / f"{stem}.csv", index=False)
                torch.save(checkpoint, output / "checkpoints" / f"{stem}.pt")
                predictions.to_csv(output / "oof_predictions" / f"{stem}.csv", index=False)
                record_path.write_text(json.dumps(record, indent=2) + "\n")
            records.append(record)
            pd.DataFrame(records).to_csv(output / "fold_results.csv", index=False)
            print(f"Config {candidate['config_id']:02d}/{len(candidates)}, fold {fold}: "
                  f"RMSLE={record['validation_rmsle']:.6f}, best epoch={record['best_epoch']}, "
                  f"run epochs={record['epochs_run']}", flush=True)
    results = pd.DataFrame(records)
    ranking = results.groupby("config_id").agg(
        mean_validation_rmsle=("validation_rmsle", "mean"),
        std_validation_rmsle=("validation_rmsle", "std"),
        mean_best_epoch=("best_epoch", "mean"),
        folds=("fold", "count"), total_seconds=("seconds", "sum"),
        folds_hitting_epoch_limit=("hit_epoch_limit", "sum")).reset_index()
    if len(results) != config["n_configs"] * config["n_folds"] or not (ranking.folds == 5).all():
        raise ValueError("Search incomplete.")
    ranking = ranking.sort_values(["mean_validation_rmsle", "config_id"])
    ranking.to_csv(output / "search_ranking.csv", index=False)
    winning_id = int(ranking.iloc[0].config_id)
    winning = next(c for c in candidates if c["config_id"] == winning_id)
    selected_folds = results[results.config_id == winning_id]
    selection = {"candidate": winning,
        "mean_validation_rmsle": float(ranking.iloc[0].mean_validation_rmsle),
        "std_validation_rmsle": float(ranking.iloc[0].std_validation_rmsle),
        "std_definition": "sample standard deviation across five folds (ddof=1)",
        "fold_best_epochs": selected_folds.best_epoch.tolist(),
        "proposed_final_refit_epochs": int(np.median(selected_folds.best_epoch)),
        "refit_rule": "Median of selected configuration's five best epochs; no test data used",
        "training_seconds_total": float(results.seconds.sum()),
        "current_invocation_seconds": time.perf_counter() - started,
        "completed_fits": len(results), "final_refit_performed": False,
        "test_evaluated": False,
        "selected_folds_hitting_epoch_limit": int(selected_folds.hit_epoch_limit.sum())}
    (output / "selected_configuration.json").write_text(json.dumps(selection, indent=2) + "\n")
    oof = pd.concat([pd.read_csv(output / "oof_predictions" / f"config_{winning_id:02d}_fold_{fold}.csv")
        for fold in range(1, 6)], ignore_index=True)
    if len(oof) != len(training) or set(oof.ID) != set(training.ID) or oof.ID.duplicated().any():
        raise ValueError("Selected out-of-fold predictions do not cover training exactly once.")
    oof.to_csv(output / "selected_oof_predictions.csv", index=False)
    # Representative fold fixed in advance: fold 1, rather than the best-looking fold.
    curve = pd.read_csv(output / "histories" / f"config_{winning_id:02d}_fold_1.csv")
    figure, axis = plt.subplots(figsize=(6, 3.6))
    axis.plot(curve.epoch, curve.training_rmsle, label="Training")
    axis.plot(curve.epoch, curve.validation_rmsle, label="Validation")
    best_epoch = int(selected_folds.loc[selected_folds.fold == 1, "best_epoch"].iloc[0])
    axis.axvline(best_epoch, color="gray", linestyle="--", label="Selected epoch")
    axis.set(xlabel="Epoch", ylabel="RMSLE (base 10, log1p)", title="Data-driven network: selected configuration, fold 1")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "selected_fold_1_learning_curve.png", dpi=180)
    plt.close(figure)
    print(json.dumps(selection, indent=2), flush=True)
    return ranking, selection


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path)
    parser.add_argument("--assignments", type=Path, default=Path("outputs/partitions/partition_assignments.csv"))
    parser.add_argument("--config", type=Path, default=Path("configs/data_driven_search.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/data_driven_search"))
    arguments = parser.parse_args()
    run_search(arguments.workbook, arguments.assignments,
        json.loads(arguments.config.read_text()), arguments.output)
