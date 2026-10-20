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
