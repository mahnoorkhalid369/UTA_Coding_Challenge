# Final review notes

## Scope

Reviewed the report, notebook, scripts, and saved outputs against the supplied
challenge. This is an internal scientific and reproducibility review.

## Issues corrected

- Restored missing `GroupKFold` / `GroupShuffleSplit` imports in the reorganized
  notebook. A fresh run caught this defect before model selection could start.
- Reworded the report's headline and interpretation: the physics-informed model
  had slightly lower test error; this comparison does not isolate a causal
  benefit of the physics penalty.
- Made the log(y) training-loss versus log(1+y) selection-metric difference more
  explicit, alongside limited search coverage and future matched-weight/seed studies.
- Removed unused code/imports and stale `planned_not_trained` metadata from the
  current configuration. Original execution manifests remain unchanged.
- Shortened reproduction instructions and retained the requested removal of
  notebook section F. Report generation remains outside the notebook.

## Verification completed

- Executed all analysis cells sequentially in a new Python process and fresh
  working directory, using the supplied workbook and verified pinned dependencies.
- Repeated 100 data-driven and 100 physics-informed fold fits with the predefined
  settings. All fold scores, best epochs, and stopping epochs matched the original
  experiment. Maximum score difference: **0.0**.
- Repeated fixed-duration final refitting and evaluation. All predictions and
  learned exponents matched the original records. Maximum difference: **0.0**.
- Confirmed identical partitions and the required final comparison table.
- Static analysis of the extracted notebook and five analysis scripts passed.
- Confirmed the report has three pages, all required figure groups, unchanged
  metrics/runtime, and AI-assistance disclosure.
- Checked that metadata cleanup preserves every scientific configuration value
  and that equation/gradient diagnostics remain unchanged.

Machine-readable evidence: `outputs/review/reproduction_check.json`.
The repetition is verification of the fixed experiment, not a second selection
stage or evidence of robustness to new seeds/data.

## Limitations to defend

- Test R² remains modest: approximately 0.18 and 0.21. The small RMSLE improvement
  on one split does not establish a general advantage.
- The exponent network and predictor can adapt together. Learned m values are
  not independently measured physical parameters.
- The extra branch and training differences are not isolated by a matched
  ablation. The candidate tradeoff plot is exploratory.
- Twenty candidates per model and one initialization per fold provide limited
  coverage. Fold SD is not a confidence interval.
- A training-only physical compatibility check identifies 359 observations that
  exceed the equation's positive-m upper bound; these observations are retained.
- RMSLE uses the explicitly stated base-10 log(1+y) convention. It must not be
  compared directly with a metric defined using log(y) alone.

## Environment boundary

The exact installed package versions were verified; the dependency-installation
cell was not rerun during this check. The supplied workbook was placed in the
fresh directory, so Colab's browser upload interface was not exercised. The Python
analysis ran end to end; the interactive Colab UI remains unverified.
