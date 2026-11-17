# Development amendment: stochastic B0–B3 và learner selection

Ngày khóa trong workspace: **19/09/2026, trước lần chạy pipeline mới trong cùng
phiên làm việc**. Đây là development specification, không khẳng định Git history
đã chứng minh preregistration độc lập. Evaluation 120 MDP vẫn sealed.

## Dữ liệu và đơn vị thống kê

- Cohort: 15 development MDP, 5 behavior-policy families, tổng 75 groups.
- Pilot feature source: prefix `10|S||A|` của pilot stream trong
  `datasets/week1_development_validation`; không dùng training stream.
- Labels/utility: independent training streams tại
  `artifacts/week2/development_offline_rl_2026-08-22/`.
- Base MDP là cluster và fold unit; mọi policy family của cùng MDP nằm cùng fold.
- Grid requirement: `{1,2,5,10,20,50}|S||A|`; target `NR>=0.95`, suffix success
  `8/10`; right-censored cases được giữ trong survival likelihood.

## Feature sets lồng nhau

- **B0:** log pilot size và log `|S||A|`.
- **B1:** B0 + observed state/action coverage, visitation entropy, minimum
  positive cell count, trajectory discounted-return mean/std/IQR, completion rate.
- **B2:** B1 + count-aware conditional transition entropy, fraction low-count
  cells, fraction empirically stochastic cells, entropy split by coverage và
  trajectory-bootstrap uncertainty of return/rare-cell exposure.
- **B3:** B2 + cross-fitted EB-OEH profile summaries: finite/unsupported/unresolved
  fractions, minimum log requirement, selected QVI index, range/slope/curvature.

Không feature nào được đọc `P,R,V*,Q*`, exact return, training return hoặc
`N_emp`. Oracle/training quantities chỉ được dùng sau khi OOF prediction tồn tại.

## Cross-fitted EB-OEH

Trajectory IDs được chia deterministically thành hai folds. Với mỗi hướng:

1. model fold khóa empirical model, `Q^[k]`, greedy policy và top/runner pair;
2. evidence fold độc lập ước lượng Bellman-target gap, variance và behavior
   coverage;
3. lower-confidence gap bằng point gap trừ hai empirical-Bernstein radii;
4. Clopper–Pearson lower coverage đi vào requirement;
5. missing relevant action trả `unsupported`; nonpositive lower gap trả
   `unresolved`; không gán finite value;
6. đổi vai fold và lấy maximum requirement của hai hướng tại depth chỉ khi cả hai
   identified/admissible.

Đây là finite-sample conservative proxy, chưa phải exact alternative-MDP KL
optimizer hoặc theorem instance-optimal.

## Validation

- Outer grouped folds: 5; inner grouped folds: 4.
- Ridge grid khóa: `{1e-4,1e-3,1e-2,1e-1,1}`; chọn bằng inner validation mean
  survival log-likelihood.
- Scaling chỉ fit trên outer/inner training rows tương ứng.
- Primary predictive contrast: B2−B1 mean log-likelihood, paired cluster bootstrap
  10.000 draws; B3−B2 là OEH incremental ablation.
- Secondary: integrated Brier trên toàn grid.
- Development interpretation only; không dùng p-value để quyết định mở evaluation.

## Learner-selection utility

- Candidate learners: CFQI, tabular IQL, tabular BC.
- Primary fixed budget khóa: multiplier `10`.
- Với mỗi feature set, chọn learner có OOF predicted probability đạt requirement
  trước hoặc tại budget 10 cao nhất; deterministic tie order `CFQI, IQL, BC`.
- Comparator chính: best-fixed learner được chọn chỉ từ outer-training MDPs tại
  budget 10. Oracle-best per group chỉ là reference.
- Metric: mean normalized return trên toàn 75 groups, paired cluster CI theo MDP;
  không loại abstention. Nếu một feature row không hợp lệ, fallback về best-fixed.

## Gate

Development run chỉ đủ để quyết định implementation và scope. Confirmatory chỉ
được cân nhắc sau khi code/tests/artifact hoàn tất, effect-size threshold được
quy đổi từ chi phí và external replication được giải quyết. Nếu B2 không cải
thiện B1 hoặc selector không cải thiện best-fixed, kết luận đúng là negative
development result; không chỉnh feature/threshold rồi gọi lại cùng run là test mới.
