# T4 — Spec của các ứng viên OEH đơn giản hóa (viết trước khi chạy, 15/10/2026)

Thẻ T4 trong [`docs/agent_kit/TASKS.md`](../agent_kit/TASKS.md); trả lời W3 ("đâu là phương pháp, đâu là phần dán thêm?"). Spec này và quy tắc chọn ở mục 3 được commit **trước** lần chạy đầu tiên.

## 1. Các thành phần (mỗi thành phần một phương trình)

Ký hiệu khối cột theo lineage M1: G1 = B2 (18 thống kê chung); **P** = mô phỏng hậu nghiệm OEH-B (G2, G3, G6); **H** = rủi ro lập kế hoạch OEH-H (G4, G5); **D** = chẩn đoán cross-fit trên pilot (G7, G8).

- **Head đơn** h_X: thành phần multiclass (XGBoost, sâu 3, 200 vòng) của quy trình matched, fit riêng theo learner trên 720 MDP meta-training với tập cột X. Đây là "head" duy nhất của M6.
- **Head hữu hạn** h_X^fin: cùng mô hình, fit trên các dòng có E < 6, rồi chuẩn hóa lại trên 6 lớp hữu hạn.
- **Cổng dịch chuyển** (tham số đã khóa của OEH: λ_max = 1, ε_max = 0,02, d0 = 3; d là khoảng cách 10-NN trong B2):

  g(p, q_O, q_R) = (1 − ε(d)) [(1 − p_6)((1 − λ(d)) q_O + λ(d) q_R), p_6] + ε(d) π_l,

  với p là luật của head đầy đủ, q_O là phần hữu hạn của head hữu hạn cùng tập cột, q_R là phần hữu hạn của head vững.

## 2. Ứng viên

| Ứng viên | Head đầy đủ | Cổng | Head vững |
|---|---|---|---|
| S1 | h_{B2+P} | không | — |
| S2 | h_{B2+P+H} | không | — |
| S3 | h_{B2+P} | có | h_{B2} |
| S4 | h_{B2+P+D} | có | h_{B2+D} |
| Tham chiếu: OEH | ensemble 108 cột (đã khóa) | có | ensemble B2 + D |
| Tham chiếu: OEH-F | ensemble 108 cột | không | — |

Mọi head fit trên cùng 720 MDP meta-training, theo thứ tự hàng đã niêm phong. Không có tham số nào được tune lại; cổng dùng đúng tham số đã khóa.

## 3. Quy tắc chọn (cố định trước khi chạy)

**Panel đã mở (11 panel):**
- trong generator: cohort 7, cohort 8, internal (kiểm định cuối);
- ngoài generator: Colosseum 2027, Colosseum 2028, SPIBB, Colosseum 2029, SPIBB-2, forest, FrozenLake, gambler.

**Điểm** của ứng viên c trên panel k: rel_k(c) = (IBS_k(c)/IBS_k(OEH) − 1 + LL_k(c)/LL_k(OEH) − 1)/2, với điểm trung bình theo MDP.

**Chấp nhận được** khi cả hai điều kiện đúng:
- trong generator: IBS và LL gộp theo MDP mỗi cái ≤ +1% so với OEH;
- ngoài generator: trung bình rel_k ≤ +1%.

**Khuyến nghị (tối đa 2):**
- ứng viên chấp nhận được **đơn giản nhất** (thứ tự S1 < S2 < S3 < S4);
- và ứng viên chấp nhận được có **điểm ngoài tốt nhất**, nếu khác ứng viên trên.
- Nếu không ứng viên nào chấp nhận được: khuyến nghị **giữ OEH**.

**Nested LOPO:** với mỗi panel k trong 11 panel, áp quy tắc trên 10 panel còn lại rồi ghi điểm của lựa chọn đó trên k (lựa chọn "giữ OEH" có điểm 0).

**Không đăng ký ứng viên nào trong thẻ này.** Kết quả dừng ở chốt H1.

## 4. Điểm vào và ví dụ

- Điểm vào của phương pháp cuối là `scripts/analysis/oeh_pipeline.py` (tái tạo 4 panel cuối đến 1e-15).
- Thêm `scripts/examples/forecast_one_pilot.py`: dựng một bundle nhỏ chỉ gồm một nhóm MDP–behavior của một panel đã mở, chạy điểm vào trên đó, in luật 7 lớp cho 3 learner.
