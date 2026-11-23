# M3 calibration summary (measured)

Source: `docs/results/m3_calibration.csv` from `python -m router.run_calibration`.

## Does calibration help routing?

**On the ID split, no.** LightGBM is already well calibrated; fixing the remaining ECE does not
raise the cost–quality curve.

| Router | Calibrator | ECE (test) | Route AUC |
|---|---|---|---|
| lightgbm | none | 0.0087 | **0.7571** |
| lightgbm | isotonic | 0.0078 | 0.7542 |
| lightgbm | temperature | 0.0088 | 0.7566 |
| knn | none | 0.0101 | 0.7559 |
| knn | isotonic | 0.0070 | 0.7534 |
| knn | temperature | 0.0074 | 0.7553 |

Val ECE goes to ~0 under isotonic (as expected: isotonic is fit on val), but test ECE barely moves
and routing AUC is flat or slightly worse. **Interview line:** calibration is necessary for
*probabilities as probabilities*, but our router only needs *ranking* of `q̂ − λĉ`, so small
monotone miscalibration often does not change decisions.

On some LOBO folds the picture is mixed (temperature helps HellaSwag LightGBM AUC 0.687 → 0.711;
isotonic can hurt MMLU). Default remains uncalibrated LightGBM.
