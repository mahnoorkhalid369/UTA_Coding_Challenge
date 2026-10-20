"""Inspect the supplied workbook without fitting transforms on held-out data."""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURES = [f"x{i}" for i in range(1, 10)]


def load_workbook(path):
    # Row 1 contains merged section headings; row 2 contains feature names.
    df = pd.read_excel(path, sheet_name="sorted all", header=1)
    if df.shape[1] != 11 or list(df.columns[1:]) != FEATURES + ["y"]:
        raise ValueError("Unexpected workbook columns; inspect the source header.")
    df.columns = ["ID"] + FEATURES + ["y"]
    df = df.apply(pd.to_numeric, errors="raise")
    if not np.isfinite(df.to_numpy()).all():
        raise ValueError("Missing or nonfinite values require explicit investigation.")
    if df.ID.duplicated().any() or (df.ID % 1 != 0).any():
        raise ValueError("IDs must be unique integers.")
    if (df[["x1", "x2", "y"]] <= 0).any().any():
        raise ValueError("The physical relation and log target require positive x1, x2, y.")
    df["ID"] = df.ID.astype("int64")
    # Exact predictor matches share an ID regardless of their observed target.
    df["predictor_group"] = df.groupby(FEATURES, sort=True, dropna=False).ngroup()
    return df


class FoldPreprocessor:
    """Fit only on a fold's training rows; preserve original units separately.

    Raw features are standardized, including features that can be zero.
    The target is represented as log10(y), with no learned target statistics.
    """

    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, training_rows):
        self.scaler.fit(training_rows[FEATURES])
        return self

    def transform_inputs(self, rows):
        return self.scaler.transform(rows[FEATURES])

    @staticmethod
    def transform_target(rows):
        values = rows["y"].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("Log target requires finite positive values.")
        return np.log10(values)

    @staticmethod
    def inverse_target(log_predictions):
        with np.errstate(over="raise", under="raise", invalid="raise"):
            predictions = np.power(10.0, np.asarray(log_predictions, dtype=float))
        if not np.isfinite(predictions).all() or (predictions <= 0).any():
            raise ValueError("Predictions must be finite and strictly positive.")
        return predictions


def prepare(path, output):
    df = load_workbook(path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    groups = df.groupby("predictor_group")
    audit = {
        # Total number of observations.
        "samples": len(df),
        # Names of the nine model input columns.
        "predictors": FEATURES,
        # Total number of missing cells across all columns.
        "missing_cells": int(df.isna().sum().sum()),
        # Total number of NaN or positive/negative infinity cells.
        "nonfinite_cells": int((~np.isfinite(df.to_numpy())).sum()),
        # Number of distinct combinations of all nine input features.
        "unique_predictor_groups": int(groups.ngroups),
        # All rows sharing their feature combination with another row, including the first.
        "rows_in_repeated_predictor_groups": int(df.duplicated(FEATURES, keep=False).sum()),
        # Rows repeating a feature combination after its first occurrence.
        "extra_rows_with_repeated_predictors": int(df.duplicated(FEATURES).sum()),
        # Feature groups containing more than one distinct target value.
        "groups_with_different_targets": int((groups.y.nunique() > 1).sum()),
        # Extra rows with identical features AND target, ignoring sample ID.
        "repeated_records_excluding_id": int(df.duplicated(FEATURES + ["y"]).sum()),
        # No observations are removed by this preprocessing step.
        "removed_rows": 0,
        # Scaling is deferred until training partitions have been constructed.
        "scaling_fitted": False,
    }
    df.to_csv(output / "validated_data.csv", index=False, float_format="%.17g")
    df.describe().to_csv(output / "descriptive_statistics.csv")
    repeated = df[df.duplicated(FEATURES, keep=False)].sort_values(["predictor_group", "ID"])
    repeated.to_csv(output / "repeated_predictors.csv", index=False, float_format="%.17g")
    (output / "data_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/preprocessing"))
    args = parser.parse_args()
    prepare(args.workbook, args.output)
