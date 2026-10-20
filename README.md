# Neural-network coding challenge

Compare a data-driven network with a network that jointly learns a positive,
sample-dependent exponent in the supplied physical equation.

## Start in Colab

1. Open `UTA_Coding_Challenge.ipynb` in Google Colab.
2. Run the dependency cell. Restart the runtime if Colab requests it.
3. Upload the supplied `Database.xls` when prompted, then run cells in order.

Sections **A–E** cover shared data/metrics, the data-driven model, the
physics-informed model, final comparison, and references. Architecture, training,
and search have separate cells for each model. The academic report is maintained
outside the notebook; there is no report-generation or ZIP-export cell.

A fresh Colab runtime can use the default `OUTPUT_ROOT = Path("outputs")`.
When rerunning a repository checkout containing saved results, first change it
to a fresh folder, such as `Path("outputs/reproduction")`. Save needed files
before ending a temporary Colab runtime. The interactive Colab upload interface
has not been tested from this workspace.

## Reproduce with Python scripts

Use Python 3.12 and the supplied workbook. Install the pinned CPU dependencies:

```sh
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Run the fixed experiment into a fresh output folder:

```sh
python preprocess.py /path/to/Database.xls --output outputs/reproduction/preprocessing
python create_partitions.py /path/to/Database.xls --output outputs/reproduction/partitions
python train_data_driven.py /path/to/Database.xls --assignments outputs/reproduction/partitions/partition_assignments.csv --output outputs/reproduction/data_driven_search
python train_physics_informed.py /path/to/Database.xls --assignments outputs/reproduction/partitions/partition_assignments.csv --output outputs/reproduction/physics_informed_search
python finalize_models.py /path/to/Database.xls --assignments outputs/reproduction/partitions/partition_assignments.csv --data-driven outputs/reproduction/data_driven_search --physics outputs/reproduction/physics_informed_search --output outputs/reproduction/final_evaluation
```

The searches contain 20 configurations × five folds each: **200 fold fits total**.
The original run took approximately 14 minutes of fold-training time on one CPU
thread. Runtime varies by machine. The notebook contains the same model and
training implementations as the scripts.

## Protocol and metric conventions

- Seed 42; identical grouped partitions and five folds for both models.
- Keep matching predictor combinations together. ID and group number are metadata.
- Retain all observations; fit input transformations on fitting rows only.
- Train on `log10(y)` MSE; inverse-transform predictions without clipping.
- Select and early-stop using base-10 RMSLE:
  `sqrt(mean((log10(1 + predicted_y) - log10(1 + observed_y))**2))`.
  This differs from RMSE in `log10(y)` space.
- R² uses original-scale residuals. Fold SD uses `ddof=1` and is not a confidence interval.
- Refit on all training rows for the median selected CV best epoch, seed 42.
  Fix both configurations and durations before evaluating test data.
- The physical equation uses original units:
  `log10(y_phys) = 2*log10(x2) - log10(72.2) + m*log10(x1)`.
  The exponent branch uses features alone and softplus ensures `m > 0`.
- Do not use the already-known test results for further selection.

## Files and experiment records

| Location | Contents |
|---|---|
| `configs/` | Predefined search spaces and fixed training controls |
| `outputs/preprocessing/` | Validated data, quality audit, repeated records, descriptive statistics |
| `outputs/partitions/` | Sample IDs, groups, train/test assignment, validation fold, split counts |
| `outputs/data_driven_search/` | All candidates, scores, histories, checkpoints, selected settings, out-of-fold predictions |
| `outputs/physics_informed_search/` | Same records plus separate losses, exponent histories, and equation/gradient checks |
| `outputs/final_evaluation/` | Final models, locked settings, metrics, ID-linked predictions/m, runtimes, verification records |
| `outputs/final_evaluation/figures/` | Required figures and supporting diagnostics in PNG/PDF |
| `reports/final_report.pdf` | Three-page academic report |
| `reports/preprocessing_progress_report.docx` | Editable version of the report |

Both search folders contain generated `configuration_results.csv` and `.md`
tables with all five fold scores, hyperparameters, mean, SD, rank, and runtime.
Final required metrics are in `model_comparison.csv`; final sample-level values
are in `final_predictions.csv`.

Historical manifests and source snapshots preserve the original execution.
A changed source hash or entry point requires a new output directory rather than
silently mixing cached runs. The final selection lock prevents overwriting an
existing evaluation with different settings. Finalization checkpoints differ from
fold checkpoints: only final models use the complete training set.

## Sources and assistance

The workbook and challenge PDF are supplied by the assessment. The implementation
uses NumPy, pandas, scikit-learn, PyTorch, and Matplotlib. Method references are
linked in notebook section E, including the official PyTorch Softplus/autograd
documentation and the original PINNs paper. This challenge uses an algebraic
constraint, rather than a PDE residual.

OpenAI Codex assisted with code, execution, interpretation, figures, and report
writing. The author must understand, verify, and disclose this assistance.

## Final verification

The notebook's full analysis was rerun in a fresh working directory with the
pinned dependencies. All 200 fold scores/epochs and all final predictions,
exponents, and comparison metrics reproduced the original records exactly.
See `REVIEW_NOTES.md` and `outputs/review/reproduction_check.json` for the scope,
corrected issues, evidence, and remaining limitations. This verifies deterministic
reproduction; it does not establish statistical robustness or validate the Colab UI.
