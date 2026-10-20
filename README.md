# Coding challenge: preprocessing

Run with Python 3.12 from this directory:

```sh
python -m pip install -r requirements.txt
python preprocess.py /path/to/Database.xls
```

Outputs in `outputs/preprocessing/` include validated observations, descriptive
statistics, repeated predictor records, and a machine-readable audit. The source
workbook is never modified. All observations are retained, including repeated
records and extreme values. The second workbook row supplies column names.
Missing, nonnumeric, nonfinite, invalid ID, or nonpositive physical inputs/target
values trigger explicit errors rather than silent imputation or removal.

`ID` and `predictor_group` are metadata, never model inputs. Exact matches across
all nine predictors share a group so subsequent train/test and cross-validation
partitions can keep them together, including matches with different targets.

`FoldPreprocessor` standardizes the nine raw features, supporting zeros without
an arbitrary log offset. Its scaler must be fitted separately on each fold's
training portion and applied to that fold's validation portion. Fit a new scaler
on the full training portion for final refitting. No scaler is fitted by the
inspection command; partitions and model training are subsequent steps.

The initial target representation is `log10(y)` because y is positive and spans
several orders of magnitude. Its inverse, `10**prediction`, enforces positive
predictions; invalid numerical results raise errors rather than being clipped.
This transformation is a modeling choice, not the definition of RMSLE. Metrics
must later be computed explicitly on inverse-transformed predictions.

Keep original x1 and x2 values for the physics loss in the supplied units:
`y = x2**2 / (72.2 * x1**(-m))`. Do not substitute standardized values into
this equation. Model evaluation and tuning have not yet been performed.

This preprocessing implementation and documentation were prepared with OpenAI
Codex assistance. Review and disclose this contribution in the final submission.

## Shared train/test split and validation folds

```sh
python create_partitions.py /path/to/Database.xls
```

`outputs/partitions/partition_assignments.csv` links each sample ID and predictor
group to `train` or `test`. Training rows have a `validation_fold` from 1 to 5;
test rows have a blank fold. Use these same saved assignments for both models.
Join by ID rather than assuming row order when loading assignments later.

The holdout uses `GroupShuffleSplit(test_size=0.2, random_state=42)`: the 20%
fraction applies to distinct predictor groups, so sample fractions can differ
slightly from 80/20. Within training, `GroupKFold(n_splits=5, shuffle=True,
random_state=42)` creates group-disjoint folds. Neither operation uses target
values. `partition_summary.json` records actual sample counts and checks for
complete coverage, group separation, and deterministic regeneration.

For validation fold k, use training-partition rows with `validation_fold != k`
for fitting and those with `validation_fold == k` for validation. Fit a fresh
`FoldPreprocessor` on those fitting rows only. Test rows are excluded throughout
cross-validation. This step neither fits scaling nor trains a model.
