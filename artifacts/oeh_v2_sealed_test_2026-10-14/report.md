| Method | Internal | Colosseum 2029 | SPIBB-2 | Forest | Pooled external |
|---|---:|---:|---:|---:|---:|
| OEH v2 | 0.0476 / 0.562 | 0.1080 / 0.833 | 0.0189 / 0.358 | 0.1097 / 0.790 | 0.0662 / 0.599 |
| OEH | 0.0475 / 0.563 | 0.1153 / 0.951 | 0.0188 / 0.352 | 0.1216 / 0.941 | 0.0709 / 0.665 |
| B2 + legacy | 0.0497 / 0.579 | 0.1154 / 0.873 | 0.0190 / 0.354 | 0.1794 / 1.022 | 0.0832 / 0.659 |
| B2 | 0.0565 / 0.613 | 0.1407 / 0.989 | 0.0252 / 0.420 | 0.2370 / 1.285 | 0.1063 / 0.782 |
| XGBoost-CF/18 | 0.0596 / 1.122 | 0.1436 / 1.256 | 0.0277 / 0.457 | 0.2535 / 1.447 | 0.1118 / 0.918 |
| B2 + LCE | 0.0544 / 0.597 | 0.1396 / 0.995 | 0.0249 / 0.400 | 0.1299 / 0.865 | 0.0832 / 0.686 |
| B2 + CB | 0.0560 / 0.607 | 0.1375 / 0.984 | 0.0224 / 0.391 | 0.2108 / 1.164 | 0.0984 / 0.741 |
| B2 + legacy, shrunk | 0.0498 / 0.580 | 0.1145 / 0.866 | 0.0190 / 0.357 | 0.1790 / 1.018 | 0.0829 / 0.657 |
| B2, shrunk | 0.0565 / 0.615 | 0.1393 / 0.976 | 0.0252 / 0.423 | 0.2356 / 1.275 | 0.1055 / 0.777 |

| Step | Baseline | Group | Metric | Mean improvement | Simultaneous LB | Verdict |
|---|---|---|---|---:|---:|---|
| step1 | B2 + legacy | external | brier | +0.0171 | +0.0107 | GO |
| step1 | B2 + legacy | external | log_loss | +0.0597 | +0.0344 | GO |
| step2 | OEH | internal_noninferiority | brier | +0.0009 | +0.0003 | GO |
| step2 | OEH | internal_noninferiority | log_loss | +0.0115 | +0.0045 | GO |
| step2 | OEH | external | brier | +0.0047 | +0.0028 | GO |
| step2 | OEH | external | log_loss | +0.0662 | +0.0398 | GO |
| step3 | B2 | internal | brier | +0.0089 | +0.0046 | GO |
| step3 | B2 | internal | log_loss | +0.0512 | +0.0294 | GO |
| step3 | XGBoost-CF/18 | internal | brier | +0.0120 | +0.0074 | GO |
| step3 | XGBoost-CF/18 | internal | log_loss | +0.5594 | +0.3428 | GO |
| step3 | lce_raw | internal | brier | +0.0604 | +0.0449 | GO |
| step3 | lce_raw | internal | log_loss | +0.6343 | +0.5185 | GO |
| step3 | B2 + LCE | internal | brier | +0.0069 | +0.0028 | GO |
| step3 | B2 + LCE | internal | log_loss | +0.0348 | +0.0134 | GO |
| step3 | cb_only | internal | brier | +0.0441 | +0.0327 | GO |
| step3 | cb_only | internal | log_loss | +0.2419 | +0.1939 | GO |
| step3 | B2 + CB | internal | brier | +0.0084 | +0.0043 | GO |
| step3 | B2 + CB | internal | log_loss | +0.0444 | +0.0238 | GO |
| step3 | B2 | external | brier | +0.0401 | +0.0259 | GO |
| step3 | B2 | external | log_loss | +0.1827 | +0.1231 | GO |
| step3 | XGBoost-CF/18 | external | brier | +0.0457 | +0.0297 | GO |
| step3 | XGBoost-CF/18 | external | log_loss | +0.3187 | +0.1879 | GO |
| step3 | lce_raw | external | brier | +0.0293 | +0.0101 | GO |
| step3 | lce_raw | external | log_loss | +0.2445 | +0.1282 | GO |
| step3 | B2 + LCE | external | brier | +0.0171 | +0.0088 | GO |
| step3 | B2 + LCE | external | log_loss | +0.0867 | +0.0478 | GO |
| step3 | cb_only | external | brier | +0.0512 | +0.0357 | GO |
| step3 | cb_only | external | log_loss | +0.3866 | +0.3087 | GO |
| step3 | B2 + CB | external | brier | +0.0322 | +0.0206 | GO |
| step3 | B2 + CB | external | log_loss | +0.1421 | +0.0954 | GO |

Decision regret (tau 0.8, lambda 100), multiples of |S||A|:

| method | colosseum2029 | forest | internal | spibb2 |
|---|---|---|---|---|
| always_skip | 24.59 | 39.63 | 22.42 | 19.72 |
| always_x50 | 120.52 | 108.8 | 116.63 | 108.61 |
| b2_matched | 24.86 | 35.81 | 16.47 | 9.07 |
| legacy | 22.88 | 28.15 | 15.07 | 7.04 |
| oeh_final | 20.46 | 19.45 | 13.22 | 7.99 |
| v2 | 21.06 | 20.03 | 13.12 | 7.99 |

Extended endpoint (rows labeled >x50 by v2's primary label):

| source_panel | 6 | 7 | 8 |
|---|---|---|---|
| colosseum2029 | 20 | 18 | 750 |
| forest | 4 | 4 | 421 |
| internal | 101 | 70 | 1646 |
| spibb2 | 8 | 1 | 216 |

Shapley (mean per-MDP loss improvement over B2):

| scope | metric | P | H | L |
|---|---|---|---|---|
| internal | brier | 0.00281 | 0.00244 | 0.00371 |
| internal | log_loss | 0.01625 | 0.015 | 0.01974 |
| colosseum2029 | brier | 0.00577 | 0.00864 | 0.01093 |
| colosseum2029 | log_loss | 0.02221 | -0.03342 | 0.04847 |
| spibb2 | brier | 0.00015 | 0.00058 | 0.0057 |
| spibb2 | log_loss | 0.01166 | 0.00043 | 0.05603 |
| forest | brier | 0.06924 | 0.02253 | 0.0237 |
| forest | log_loss | 0.31204 | -0.08653 | 0.11874 |
| pooled_external | brier | 0.01647 | 0.00775 | 0.01114 |
| pooled_external | log_loss | 0.07823 | -0.02857 | 0.06685 |
