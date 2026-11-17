"""Reproduce the final results of the thesis from a fresh clone: ``python -m scripts.reproduce --out <dir>``.

Steps (each writes into ``<dir>`` and is skipped when its output already exists):

1. ``seals``      check that the protocols, seals and labels of the three final tests are the committed bytes;
2. ``forecast``   refit the forecasting heads on the meta-training data and regenerate, from the sealed pilot inputs,
                  the forecasts of OEH, the full head, the robust head, B2 and OEH without each evidence block;
                  every regenerated forecast that was sealed is compared with the sealed one;
3. ``registered`` rerun the registered analysis of each test on a copy of its files, with the regenerated forecasts
                  in place of the sealed ones, and compare every output with the committed result;
4. ``tables``     compute Tables 11-18 of the thesis and compare every printed cell.

Nothing under ``artifacts/`` is written. See the "Reproduce the results" section of the README.
"""
