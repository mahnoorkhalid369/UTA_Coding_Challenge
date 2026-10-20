# Coding challenge: data preparation and neural-network selection

## All-in-one Colab notebook

Open `UTA_Coding_Challenge.ipynb` in Google Colab and run its cells in order.
Upload the supplied `Database.xls` when prompted. The notebook includes all
completed preprocessing, splitting, and data-driven hyperparameter search code,
and packages analysis results as `challenge_progress.zip` for download.
Outputs are organized under `outputs/preprocessing/`, `outputs/partitions/`, and
`outputs/data_driven_search/`. The academic Word report is maintained separately;
the notebook neither generates nor packages it. Download analysis outputs before
closing a temporary Colab runtime. Physics-informed training and final test
evaluation remain subsequent steps. Notebook code was exercised in the cloud
workspace; Colab's interactive upload/download flow is not yet tested.

The standalone Python scripts below remain available for command-line use.

Run with Python 3.12 from this directory:

```sh
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
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

## Data-driven hyperparameter search

```sh
python train_data_driven.py /path/to/Database.xls
```

The script uses `configs/data_driven_search.json` and the saved partition CSV.
Twenty distinct combinations are sampled from 1,296 possibilities using seed 42,
with five fold fits per candidate. The notebook documents the ranges and rationale.
The objective is mean squared log10-target error plus explicit L2 on weight
matrices only. Adam has no separate weight decay. Optional selected-feature logs
are followed by fold-fitted standardization. Predictions are inverse-transformed
without clipping.

Selection and early stopping use base-10 RMSLE, explicitly defined as
`sqrt(mean((log10(1 + predicted_y) - log10(1 + observed_y))**2))`.
This differs from RMSE in log10(y) space. The epoch limit is 1,000, patience 50,
and minimum improvement for patience is 1e-5. The absolute lowest-score checkpoint
is retained. Rank candidates by mean fold RMSLE; report sample standard deviation
across folds (ddof=1). CPU runs use one PyTorch thread and deterministic algorithms.

Outputs include the search manifest and sampled candidates, all fold scores,
per-epoch histories, fitted scaler statistics and model checkpoints, validation
predictions linked to IDs, the ranked search, selected configuration, and a
representative fold-1 learning curve. Checkpoints are fold models, not final models
refitted on all training data. The predeclared final-refit duration is the median
of the selected candidate's five best epochs; refitting and test evaluation are
deferred. Selection scores are not an independent generalization estimate.

The code also generates `configuration_results.csv` and `configuration_results.md`:
one row per candidate with all hyperparameters, five fold RMSLE values, mean,
sample standard deviation, rank, selected status, and training time. The notebook
displays this table. The Markdown file is readable directly on GitHub; the CSV
preserves numerical precision for analysis. They are calculated from measured
fold results, not manually entered. The initial training source is archived under
`outputs/data_driven_search/source/` to match the original manifest's source hash.

Completed folds can be resumed when code, configuration, data, and assignments
match the saved manifest. Use a new output directory (`--output`) after changes.
Notebook and standalone-script manifests distinguish their entry points; use a
fresh directory when switching between them. Do not load untrusted checkpoints.
