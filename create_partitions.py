"""Save shared, group-disjoint holdout and five-fold assignments (seed 42)."""
import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from preprocess import load_workbook

SEED = 42


def create_assignments(df):
    # The fraction applies to predictor groups, so row counts may depart from 80/20.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_indices, test_indices = next(splitter.split(df, groups=df.predictor_group))
    assignments = df[["ID", "predictor_group"]].copy()
    assignments["partition"] = "train"
    assignments.loc[test_indices, "partition"] = "test"
    # Fold numbers indicate when a training row is used for validation.
    # Test rows have no validation fold and must never enter cross-validation.
    assignments["validation_fold"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    training = df.iloc[train_indices]
    folds = GroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (_, validation_indices) in enumerate(
        folds.split(training, groups=training.predictor_group), start=1
    ):
        assignments.loc[training.index[validation_indices], "validation_fold"] = fold
    validate_assignments(df, assignments)
    return assignments


def validate_assignments(df, assignments):
    if len(assignments) != len(df) or not assignments.ID.equals(df.ID):
        raise ValueError("Assignments must preserve every sample ID exactly once.")
    if assignments.groupby("predictor_group").partition.nunique().max() != 1:
        raise ValueError("A predictor group crosses the train/test boundary.")
    training = assignments[assignments.partition == "train"]
    testing = assignments[assignments.partition == "test"]
    if training.validation_fold.isna().any() or testing.validation_fold.notna().any():
        raise ValueError("Only training samples must have validation fold assignments.")
    if set(training.validation_fold) != set(range(1, 6)):
        raise ValueError("Exactly five nonempty validation folds are required.")
    if training.groupby("predictor_group").validation_fold.nunique().max() != 1:
        raise ValueError("A predictor group crosses validation folds.")


def prepare(workbook, output):
    df = load_workbook(workbook)
    assignments = create_assignments(df)
    # Recreate assignments to verify deterministic behavior without using targets.
    pd.testing.assert_frame_equal(assignments, create_assignments(df))
    training = assignments[assignments.partition == "train"]
    testing = assignments[assignments.partition == "test"]
    summary = {
        "seed": SEED,
        "holdout_method": "GroupShuffleSplit(test_size=0.2)",
        "fold_method": "GroupKFold(n_splits=5, shuffle=True)",
        "group_definition": "Exact equality of x1 through x9; target excluded",
        "total_samples": len(df),
        "training_samples": len(training),
        "testing_samples": len(testing),
        "training_fraction": len(training) / len(df),
        "testing_fraction": len(testing) / len(df),
        "training_groups": training.predictor_group.nunique(),
        "testing_groups": testing.predictor_group.nunique(),
        "folds": [
            {"fold": fold,
             "training_samples": int((training.validation_fold != fold).sum()),
             "validation_samples": int((training.validation_fold == fold).sum())}
            for fold in range(1, 6)
        ],
        "checks": {
            "all_samples_assigned_once": True,
            "no_train_test_group_overlap": True,
            "no_cross_fold_group_overlap": True,
            "reproducible_assignments": True,
        },
        "scaling_fitted": False,
        "models_trained": False,
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(output / "partition_assignments.csv", index=False)
    (output / "partition_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/partitions"))
    args = parser.parse_args()
    prepare(args.workbook, args.output)
