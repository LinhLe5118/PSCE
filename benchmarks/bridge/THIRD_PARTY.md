# Third-party code used by the BRIDGE survival comparator

`expanded_published_survival_forecast.py`, `fresh_published_survival_forecast.py`
and `freeze_expanded_protocol.py` import the authors' TabularFM4Survival code from
`benchmarks/bridge/third_party/TabularFM4Survival/`. That directory is a separate
Git checkout and is not committed here. Recreate it with:

```bash
git clone https://github.com/dain20000222/TabularFM4Survival.git benchmarks/bridge/third_party/TabularFM4Survival
git -C benchmarks/bridge/third_party/TabularFM4Survival checkout b27d69c27a5c39b87822b1aebf838e3dd900000e
```

No other BRIDGE module or test depends on it.
