"""Freeze selected settings, refit on full training data, then evaluate once."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from preprocess import load_workbook
from create_partitions import validate_assignments
from train_data_driven import build_network, input_values, rmsle_base10
from train_physics_informed import physics_fit, physical_log10, exponent_statistics


def original_scale_metrics(y, prediction):
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if not np.isfinite(prediction).all() or not (prediction > 0).all():
        raise ValueError("Invalid predictions; metrics must not silently clip values.")
    r2 = 1.0 - np.sum((y - prediction) ** 2) / np.sum((y - y.mean()) ** 2)
    return {"r2": float(r2), "rmsle": rmsle_base10(y, np.log10(prediction))}


def refit_data_driven(training, candidate, config, epochs, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    scaler = StandardScaler().fit(input_values(training, candidate, config))
    x = torch.tensor(scaler.transform(input_values(training, candidate, config)), dtype=torch.float32)
    y = torch.tensor(np.log10(training.y.to_numpy()), dtype=torch.float32)
    model = build_network(candidate)
    opt = config["optimizer"]
    optimizer = torch.optim.Adam(model.parameters(), lr=candidate["learning_rate"],
        betas=(opt["beta_1"], opt["beta_2"]), eps=opt["epsilon"])
    history = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        for idx in torch.randperm(len(x)).split(candidate["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            z = model(x[idx]).squeeze(-1)
            penalty = sum(p.square().sum() for name, p in model.named_parameters() if name.endswith("weight"))
            loss = torch.mean((z - y[idx]) ** 2) + candidate["l2"] * penalty
            if not torch.isfinite(loss): raise ValueError("Nonfinite refit objective.")
            loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            z = model(x).squeeze(-1).numpy()
        history.append({"epoch": epoch, "training_data_loss": float(np.mean((z - y.numpy()) ** 2)),
            "training_rmsle": rmsle_base10(training.y.to_numpy(), z)})
    checkpoint = {"state_dict": model.state_dict(), "candidate": candidate,
        "configuration": config, "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(), "epochs": epochs, "seed": seed}
    return model, scaler, pd.DataFrame(history), checkpoint, time.perf_counter() - started


def predict_final(model, scaler, rows, candidate, config, physics=False):
    # Targets are never passed to either trained network, including the m component.
    x = torch.tensor(scaler.transform(input_values(rows, candidate, config)), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        result = model(x)
        if physics:
            z, m = result
            physical = physical_log10(rows, m).numpy()
            m = m.numpy().astype(float)
            if not np.isfinite(m).all() or not (m > 0).all(): raise ValueError("Invalid final exponent.")
        else:
            z = result.squeeze(-1)
    z = z.numpy().astype(float)
    with np.errstate(over="raise", under="raise", invalid="raise"):
        prediction = np.power(10., z)
    if not np.isfinite(prediction).all() or not (prediction > 0).all(): raise ValueError("Invalid final prediction.")
    return {"prediction": prediction, **({"m": m, "physical_log10_y": physical} if physics else {})}


def save_figure(figure, stem):
    figure.tight_layout()
    figure.savefig(str(stem) + ".png", dpi=180)
    figure.savefig(str(stem) + ".pdf")
    plt.close(figure)


def generate_comparison_figures(predictions, metrics, dd_output, physics_output, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    dd = json.loads((dd_output / "selected_configuration.json").read_text())
    pi = json.loads((physics_output / "selected_configuration.json").read_text())
    selections = [("Data-driven", dd_output, dd), ("Physics-informed", physics_output, pi)]
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.8), sharey=True)
    for axis, (label, folder, selected) in zip(axes, selections):
        cid = selected["candidate"]["config_id"]
        h = pd.read_csv(folder / "histories" / f"config_{cid:02d}_fold_1.csv")
        axis.plot(h.epoch, h.training_rmsle, label="Training", linewidth=1)
        axis.plot(h.epoch, h.validation_rmsle, label="Validation", linewidth=1)
        axis.axvline(selected["fold_best_epochs"][0], color="gray", linestyle="--", linewidth=1)
        axis.set(title=f"{label}: CV fold 1", xlabel="Epoch", ylabel="RMSLE (base 10)")
        axis.legend(fontsize=8)
    save_figure(fig, output / "training_validation_curves")
    fig, axes = plt.subplots(2, 5, figsize=(15, 5), sharey=True, sharex=True)
    for row, (label, folder, selected) in enumerate(selections):
        for fold in range(1, 6):
            cid = selected["candidate"]["config_id"]
            h = pd.read_csv(folder / "histories" / f"config_{cid:02d}_fold_{fold}.csv")
            axis = axes[row, fold - 1]
            axis.plot(h.epoch, h.training_rmsle, label="Training", linewidth=1)
            axis.plot(h.epoch, h.validation_rmsle, label="Validation", linewidth=1)
            axis.axvline(selected["fold_best_epochs"][fold-1], color="gray", linestyle="--", linewidth=1)
            axis.set(title=f"{label}, fold {fold}", xlabel="Epoch")
            if fold == 1: axis.set_ylabel("RMSLE (base 10)")
            if fold == 5: axis.legend(fontsize=7)
    save_figure(fig, output / "all_folds_learning_curves")
    cid = pi["candidate"]["config_id"]
    history = pd.read_csv(physics_output / "histories" / f"config_{cid:02d}_fold_1.csv")
    def diagnostics(history, stem, label, selected_epoch=None):
        fig, axes = plt.subplots(1, 3, figsize=(9, 2.5))
        for axis, kind in zip(axes[:2], ["data", "physics"]):
            axis.plot(history.epoch, history[f"training_{kind}_loss"], label="Training", linewidth=1)
            if f"validation_{kind}_loss" in history:
                axis.plot(history.epoch, history[f"validation_{kind}_loss"], label="Validation", linewidth=1)
            axis.set(title=f"{kind.capitalize()} loss", xlabel="Epoch", yscale="log", ylabel="Log10-space MSE")
            axis.legend(fontsize=7)
        axes[2].plot(history.epoch, history.m_mean, label="Mean", linewidth=1)
        axes[2].plot(history.epoch, history.m_median, label="Median", linewidth=1)
        axes[2].fill_between(history.epoch, history.m_q10, history.m_q90, alpha=.2, label="10–90%")
        axes[2].set(title="Exponent m", xlabel="Epoch", ylabel="Exponent m")
        axes[2].legend(fontsize=7)
        if selected_epoch is not None:
            for axis in axes: axis.axvline(selected_epoch, color="gray", linestyle="--", linewidth=.8)
        fig.suptitle(label, fontsize=10)
        save_figure(fig, output / stem)
    diagnostics(history, "physics_training_diagnostics", "Physics-informed model: CV fold 1", pi["fold_best_epochs"][0])
    diagnostics(pd.read_csv(output.parent / "physics_informed_refit_history.csv"),
        "physics_final_refit_diagnostics", "Physics-informed model: full-training refit")
    fig, axes = plt.subplots(2, 2, figsize=(6, 5.5), sharex=True, sharey=True)
    names = ["data_driven_prediction", "physics_informed_prediction"]
    low = min(predictions.observed_y.min(), predictions[names].min().min()) * .7
    high = max(predictions.observed_y.max(), predictions[names].max().max()) * 1.3
    for row, (label, column) in enumerate(zip(["Data-driven", "Physics-informed"], names)):
        for col, part in enumerate(["train", "test"]):
            axis = axes[row, col]
            subset = predictions[predictions.partition == part]
            axis.scatter(subset.observed_y, subset[column], s=7, alpha=.4)
            axis.plot([low, high], [low, high], "k--", linewidth=.8)
            axis.set(xscale="log", yscale="log", xlim=(low, high), ylim=(low, high),
                title=f"{label}: {'training' if part=='train' else 'testing'}", xlabel="Observed y", ylabel="Predicted y")
            axis.set_aspect("equal", adjustable="box")
            score = metrics[(metrics.model == label) & (metrics.partition == part)].iloc[0]
            axis.text(.04, .95, f"R²={score.r2:.3f}\nRMSLE={score.rmsle:.3f}",
                transform=axis.transAxes, va="top", fontsize=8)
    save_figure(fig, output / "observed_vs_predicted")
    table = pd.read_csv(physics_output / "configuration_results.csv")
    fig, axis = plt.subplots(figsize=(5, 3.5))
    from matplotlib.colors import LogNorm
    points = axis.scatter(table.validation_physics_rmse_log10, table.mean_validation_rmsle,
        c=table.lambda_physics, norm=LogNorm(vmin=.01, vmax=10), cmap="viridis", s=40)
    winner = table[table.selected].iloc[0]
    axis.scatter([winner.validation_physics_rmse_log10], [winner.mean_validation_rmsle],
        marker="*", s=180, facecolors="none", edgecolors="red", label="Selected")
    axis.set(xlabel="Validation physics RMSE (log10 units)", ylabel="Mean validation RMSLE", title="Accuracy–physics tradeoff: evaluated candidates")
    axis.legend(fontsize=8)
    fig.colorbar(points, ax=axis, label="Physics weight λ")
    save_figure(fig, output / "accuracy_physics_tradeoff")


def finalize_models(workbook, assignments_path, dd_output, physics_output, output):
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    dd_output = Path(dd_output); physics_output = Path(physics_output)
    dd = json.loads((dd_output / "selected_configuration.json").read_text())
    pi = json.loads((physics_output / "selected_configuration.json").read_text())
    if dd["selected_folds_hitting_epoch_limit"] or pi["selected_folds_hitting_epoch_limit"]:
        raise ValueError("Investigate selected-fold convergence before evaluating the test set.")
    config_dd = json.loads((dd_output / "search_manifest.json").read_text())["configuration"]
    config_pi = pi["configuration"]
    lock = {"data_driven_selection_sha256": hashlib.sha256(json.dumps({"candidate": dd["candidate"], "configuration": config_dd, "epochs": dd["proposed_final_refit_epochs"]}, sort_keys=True).encode()).hexdigest(),
        "physics_selection_sha256": hashlib.sha256(json.dumps({"candidate": pi["candidate"], "configuration": config_pi, "epochs": pi["proposed_final_refit_epochs"]}, sort_keys=True).encode()).hexdigest(),
        "workbook_sha256": hashlib.sha256(Path(workbook).read_bytes()).hexdigest(),
        "assignments_sha256": hashlib.sha256(Path(assignments_path).read_bytes()).hexdigest(),
        "data_driven_epochs": dd["proposed_final_refit_epochs"],
        "physics_informed_epochs": pi["proposed_final_refit_epochs"],
        "refit_seed": 42, "duration_rule": "Median selected CV best epochs; no early stopping or test monitoring in refit"}
    lock_path = output / "selection_lock.json"
    if lock_path.exists():
        if json.loads(lock_path.read_text()) != lock:
            raise ValueError("Finalized selections are locked; do not retune after testing.")
        if (output / "completion.json").exists():
            print("Using completed, locked evaluation; no retraining or retuning.")
            return pd.read_csv(output / "metrics.csv")
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")
    (output / "final_settings.json").write_text(json.dumps({"data_driven": {"candidate": dd["candidate"], "configuration": config_dd}, "physics_informed": {"candidate": pi["candidate"], "configuration": config_pi}, "selection_lock": lock}, indent=2) + "\n")
    df = load_workbook(workbook)
    a = pd.read_csv(assignments_path, dtype={"validation_fold": "Int64"})
    if a.ID.duplicated().any() or set(a.ID) != set(df.ID): raise ValueError("Partition IDs mismatch.")
    a = a.set_index("ID").loc[df.ID].reset_index()
    if not np.array_equal(a.predictor_group, df.predictor_group): raise ValueError("Predictor groups mismatch.")
    validate_assignments(df, a)
    training = df.loc[a.partition == "train"]
    dd_model, dd_scaler, dd_history, dd_checkpoint, dd_seconds = refit_data_driven(
        training, dd["candidate"], config_dd, lock["data_driven_epochs"], 42)
    pi_model, pi_scaler, pi_history, pi_checkpoint, _, pi_result = physics_fit(
        training, None, pi["candidate"], config_pi, 42, final_epochs=lock["physics_informed_epochs"])
    dd_history.to_csv(output / "data_driven_refit_history.csv", index=False)
    pi_history.to_csv(output / "physics_informed_refit_history.csv", index=False)
    torch.save(dd_checkpoint, output / "data_driven_final.pt")
    torch.save(pi_checkpoint, output / "physics_informed_final.pt")
    # Both models and durations are now fixed. Score training and test once below.
    dd_pred = predict_final(dd_model, dd_scaler, df, dd["candidate"], config_dd)
    pi_pred = predict_final(pi_model, pi_scaler, df, config_pi["backbone"], config_pi, physics=True)
    predictions = pd.DataFrame({"ID": df.ID, "predictor_group": df.predictor_group,
        "partition": a.partition, "observed_y": df.y,
        "data_driven_prediction": dd_pred["prediction"], "physics_informed_prediction": pi_pred["prediction"],
        "m": pi_pred["m"], "physical_log10_y": pi_pred["physical_log10_y"]})
    predictions["physics_residual_log10"] = np.log10(predictions.physics_informed_prediction) - predictions.physical_log10_y
    predictions.to_csv(output / "final_predictions.csv", index=False)
    records = []; summaries = []
    for part in ["train", "test"]:
        subset = predictions[predictions.partition == part]
        for name, column in [("Data-driven", "data_driven_prediction"), ("Physics-informed", "physics_informed_prediction")]:
            records.append({"model": name, "partition": part, "n": len(subset),
                **original_scale_metrics(subset.observed_y, subset[column])})
        summaries.append({"partition": part, "n": len(subset), **exponent_statistics(subset.m.to_numpy()),
            "physics_rmse_log10": float(np.sqrt(np.mean(subset.physics_residual_log10 ** 2)))})
    metrics = pd.DataFrame(records)
    metrics.to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "learned_m_summary.csv", index=False)
    comparison = []
    for label, selected in [("Data-driven", dd), ("Physics-informed", pi)]:
        rows = metrics[metrics.model == label].set_index("partition")
        comparison.append({"Model": label + " neural network",
            "Training R2": rows.loc["train", "r2"], "Training RMSLE": rows.loc["train", "rmsle"],
            "Testing R2": rows.loc["test", "r2"], "Testing RMSLE": rows.loc["test", "rmsle"],
            "Validation RMSLE mean": selected["mean_validation_rmsle"],
            "Validation RMSLE std": selected["std_validation_rmsle"]})
    pd.DataFrame(comparison).to_csv(output / "model_comparison.csv", index=False)
    table = pd.read_csv(physics_output / "configuration_results.csv")
    table.groupby("lambda_physics").agg(configurations=("config_id", "count"),
        mean_validation_rmsle=("mean_validation_rmsle", "mean"),
        mean_physics_rmse=("validation_physics_rmse_log10", "mean")).to_csv(output / "lambda_tradeoff_summary.csv")
    (output / "refit_runtime.json").write_text(json.dumps({"data_driven_seconds": dd_seconds,
        "physics_informed_seconds": pi_result["seconds"]}, indent=2) + "\n")
    generate_comparison_figures(predictions, metrics, dd_output, physics_output, output / "figures")
    completion = {"final_models_refitted": True, "test_evaluation_completed": True,
        "test_used_for_selection": False, "selections_changed_after_test": False,
        "all_predictions_positive_finite": True, "all_exponents_positive_finite": True,
        "training_samples": len(training), "testing_samples": int((a.partition == "test").sum())}
    (output / "completion.json").write_text(json.dumps(completion, indent=2) + "\n")
    print(pd.DataFrame(comparison).to_string(index=False), flush=True)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path)
    parser.add_argument("--assignments", type=Path, default=Path("outputs/partitions/partition_assignments.csv"))
    parser.add_argument("--data-driven", type=Path, default=Path("outputs/data_driven_search"))
    parser.add_argument("--physics", type=Path, default=Path("outputs/physics_informed_search"))
    parser.add_argument("--output", type=Path, default=Path("outputs/final_evaluation"))
    args = parser.parse_args()
    finalize_models(args.workbook, args.assignments, args.data_driven, args.physics, args.output)
