# Thử nghiệm OEH current trên benchmark ngoài Colosseum (14/10/2026)

Đây là thử nghiệm **khám phá**, chưa phải protocol được nhóm duyệt. Toàn bộ chạy trong
`E:\research\oeh_colosseum_trial`; repo `Devoted` chỉ được đọc (và `git pull`), không bị sửa.

## Thiết kế

- **MDP:** 32 MDP stochastic từ Colosseum (Conserva & Rauber, NeurIPS 2022), lấy theo sampler
  của tác giả (seed 2026): RiverSwim 8, DeepSea 6, FrozenLake 9, SimpleGrid 9. Ba họ
  MiniGrid-Empty/Rooms/Taxi bị loại vì sampler chỉ sinh |S||A| > 256 (ngoài miền huấn luyện);
  MG-DoorKey bị loại như tác giả. Áp bộ lọc dynamic range của dự án.
- **Log và nhãn:** bộ sinh, 5 behavior, pilot x10, 10 seed, BC/CFQI/IQL và luật ≥8/10 suffix
  của dự án. 28.800 job learner.
- **Forecaster:** đúng bảng meta-training 7 cohort đã niêm phong (720 MDP, 10.800 dòng) qua
  `expanded_meta` của runner cohort 7. Kiểm tra tái tạo: 3.600/3.600 dự báo cohort 7 khớp bản
  niêm phong (sai khác ≤ 7,5e-8); điểm cohort 7 trùng đến 6 chữ số.
- **So sánh (cùng `matched_forecast`):** OEH current (108 cột), B2 matched (18 cột),
  B2 + 3 thước đo độ khó Colosseum (diameter, value norm, sub-optimality gaps; đặc quyền).
- **Thống kê:** Brier tích hợp và log loss 7 lớp, trung bình theo MDP; bootstrap 10.000 lần theo
  MDP, cận dưới đồng thời max-t một phía 95%. Tiêu chí và amendment được khóa trước nhãn; dự báo
  được niêm phong (SHA-256) trước khi chạy learner.

## Kết quả chính (32 MDP)

| Phương pháp | Brier | Log loss |
|---|---:|---:|
| **OEH current** | **0,0986** | **0,8427** |
| B2 matched | 0,1261 | 0,9314 |
| B2 + độ khó Colosseum | 0,1361 | 0,9659 |

| So sánh | Brier: cải thiện / cận dưới | Log loss: cải thiện / cận dưới | Verdict |
|---|---|---|---|
| OEH vs B2 matched | 21,8% / **+0,0030** | 9,5% / **−0,0206** | **NO-GO** (log loss) |
| OEH vs B2 + độ khó Colosseum | 27,5% / +0,0120 | 12,8% / +0,0098 | **GO** |

Log loss so với B2 khi xét riêng một phép so: cận dưới −0,0068, P(gain ≤ 0) ≈ 0,07.

## Phân tích phụ (mô tả)

- **Sai số tuyệt đối gấp đôi trong họ:** OEH 0,099 Brier trên Colosseum so với 0,046 ở cohort 7
  (B2: 0,126 so với 0,055).
- **Lợi thế đến từ các ca >x50 (không đạt):** Brier OEH 0,049 so với B2 0,091. Ở các ca đạt được
  trong x1–x50, OEH 0,230 kém B2 0,217, giống điểm yếu đã nêu trong IMPROVEMENT_REVIEW.
- **Theo learner:** BC 0,049 so với 0,105 và IQL 0,111 so với 0,139 (OEH tốt hơn); CFQI 0,137 so
  với 0,134 (không hơn).
- **Theo họ:** OEH tốt hơn rõ ở SimpleGrid và DeepSea; xấp xỉ ở FrozenLake; RiverSwim lẫn lộn.
- **Phân biệt learner trên cùng MDP:** Spearman trong nhóm OEH 0,654, B2 0,583, B2+độ khó 0,540.
- **Thước đo độ khó Colosseum (chỉ phụ thuộc MDP):** Spearman với nhu cầu dữ liệu: diameter
  0,41–0,55, value norm −0,33 đến −0,45, gaps 0,07–0,29. Kỳ vọng của OEH đạt 0,71–0,88. Thêm 3
  thước đo này vào B2 còn làm B2 **kém đi**.
- **OEH-B không có mẫu posterior** ở 26/160 nhóm (RiverSwim 11, DeepSea 9, FrozenLake 6). Ở các
  nhóm này OEH kém B2 (0,018 so với 0,006 Brier).

## Sự cố kỹ thuật phát hiện

- **Lỗi expectile của IQL** (paper ghi "chưa từng xảy ra trên dữ liệu thật") đã xảy ra trên
  Colosseum: 3 nhóm khi tính feature và 4 MDP khi chạy learner. Đã áp đúng bản sửa của nhóm
  (`code/oeh/tabular.py`: giữ nguyên kết quả mỗi khi bản cũ chạy được, chỉ dùng nghiệm vi phạm
  nhỏ nhất khi bản cũ ném lỗi), dưới dạng monkeypatch, không sửa file.
- Colosseum chỉ chạy trên Python ≤ 3.10 và cần vá Windows (`os.sysconf`, đường dẫn dài); diameter
  vô hạn ở MDP không liên thông được mã hóa là 1e4 (chỉ ảnh hưởng comparator độ khó).

## Thời gian chạy (laptop 16 nhân, RTX 3050 Ti không dùng)

| Bước | Thời gian |
|---|---:|
| Xuất MDP Colosseum, tính độ khó | ~1 phút |
| Sinh log offline (1.760 buffer) | 1 phút |
| OEH-B 160 nhóm | 10 phút |
| Pilot feature và OEH-H | 16 phút |
| Fit forecaster và niêm phong dự báo | 6 phút |
| Learner lấy nhãn (28.800 job) | 9 phút |
| Chấm điểm | < 1 phút |
| **Tổng cho một benchmark ngoài** | **~45 phút** |

## Kết luận

OEH current **tốt hơn rõ** các thước đo độ khó đã công bố và **tốt hơn B2 về Brier** trên họ MDP
do người khác thiết kế. Theo đúng gate của paper (cả hai metric), so với B2 vẫn là **NO-GO** vì
cận dưới của log loss âm. Sai số trên MDP ngoài cao gấp khoảng 2 lần trong họ, lợi thế tập trung ở
việc nhận ra ca "không bao giờ đạt", còn ước lượng mức budget cụ thể và CFQI chưa cải thiện.

Giới hạn: 32 MDP (paper có 84), 4/7 họ Colosseum, một lần chạy, chưa có review của nhóm.
