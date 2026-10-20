# Coding challenge: data preparation and neural-network selection

## All-in-one Colab notebook

Open `UTA_Coding_Challenge.ipynb` in Google Colab and run its cells in order.
Upload the supplied `Database.xls` when prompted. The notebook includes all
preprocessing, splitting, both neural-network searches, final refitting and evaluation code,
and packages analysis results as `challenge_progress.zip` for download.
Outputs are organized under `outputs/preprocessing/`, `outputs/partitions/`, and
`outputs/data_driven_search/`, `outputs/physics_informed_search/`, and
`outputs/final_evaluation/`. The academic Word/PDF report is maintained separately;
the notebook neither generates nor packages it. Download analysis outputs before
closing a temporary Colab runtime. Model settings and refit durations are fixed
before test evaluation. Notebook code was exercised in the cloud workspace; Colab's interactive upload/download flow is not yet tested.

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
this equation. Both searches and final evaluation retain this original-unit convention.

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
of the selected candidate's five best epochs; final refitting and evaluation are
performed separately after both selections. Selection scores are not an independent generalization estimate.

`selected_all_folds_learning_curves.png` and its PDF version display training and
validation RMSLE for all five folds of the selected configuration, on shared axes.
Dashed lines identify each fold's selected epoch. Curves end when that fold stops;
unequal-length runs are not padded or averaged. The notebook displays this figure.

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


## Physics-informed search and final evaluation

```sh
python train_physics_informed.py /path/to/Database.xls
python finalize_models.py /path/to/Database.xls
```

Physics settings are in `configs/physics_informed_search.json`; their initial
status records the definition stage, while actual completion is recorded in
output manifests and `outputs/final_evaluation/completion.json`. The prediction
backbone follows the selected data-driven architecture. A tanh network with a
softplus output learns a positive, sample-dependent exponent from features alone.
Both components train jointly, using data MSE plus lambda times the physical
residual MSE in log10 units, plus L2 on both networks' weights. The physical
relation is `log10(y_phys) = 2*log10(x2) - log10(72.2) + m*log10(x1)` in original
units. Twenty of 72 exponent/learning-rate/lambda combinations are evaluated
using the same five folds. The implementation checks equation equivalence,
positivity, and gradients to both components before searching.

`outputs/physics_informed_search/` contains full candidate tables, separate loss
histories, m distributions on fixed training inputs, all fold checkpoints and
validation predictions, selected settings, runtime, and implementation checks.
All observations are retained, including training records incompatible with
exact agreement under the positive-m equation.

Final refitting uses fresh models and scalers on all training observations,
seed 42, and each model's median selected CV best epoch. No early stopping or
test monitoring occurs during refitting. The selection lock excludes mutable
runtime fields and prevents changing scientific settings after evaluation.
Outputs include `model_comparison.csv` (required metrics and CV summaries),
`final_predictions.csv` (IDs, partitions, predictions, m), `learned_m_summary.csv`,
final checkpoints, refit histories/runtime, and PNG/PDF figures under `figures/`.
R2 uses original-scale residuals; base-10 RMSLE includes +1. Predictions are
positive without clipping. Test error is never plotted against epoch.

## Reproduce from the repository snapshot

Saved outputs preserve the original experiment. Use a fresh output directory
when regenerating them with the current source; historical source hashes and
entry points can differ because plotting/export code was added afterward.
For the notebook, set `OUTPUT_ROOT = Path("outputs/reproduction")` in the initial
path cell before running from top to bottom. In a fresh Colab runtime the default
output folder also works. For the standalone workflow:

```sh
python preprocess.py /path/to/Database.xls --output outputs/reproduction/preprocessing
python create_partitions.py /path/to/Database.xls --output outputs/reproduction/partitions
python train_data_driven.py /path/to/Database.xls --assignments outputs/reproduction/partitions/partition_assignments.csv --output outputs/reproduction/data_driven_search
python train_physics_informed.py /path/to/Database.xls --assignments outputs/reproduction/partitions/partition_assignments.csv --output outputs/reproduction/physics_informed_search
python finalize_models.py /path/to/Database.xls --assignments outputs/reproduction/partitions/partition_assignments.csv --data-driven outputs/reproduction/data_driven_search --physics outputs/reproduction/physics_informed_search --output outputs/reproduction/final_evaluation
```

These commands reproduce a fixed experiment. Do not use the known test results
for further selection. CPU runs use one thread; runtime can differ across
machines. The separately maintained report documents interpretation, limitations,
and AI assistance; no report-generation code is included in the notebook.

## Academic report

The updated Word manuscript is `reports/preprocessing_progress_report.docx`;
`reports/final_report.pdf` is its three-page PDF export, including the required
metrics, figures, learned exponent summaries, runtime, interpretation, and limitations.
The report is maintained outside the notebook.
