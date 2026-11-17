# T8 — Bản nháp protocol panel 6 (15/10/2026) — **dừng ở chốt H1**

Thẻ T8 của kế hoạch lên hội nghị (bộ agent kit, lưu ngoài repo). **Đây là bản nháp:** chưa đóng băng, chưa sinh dữ liệu panel, chưa chạy learner nào trên hai họ mới.
- Bản nháp: [`artifacts/week8/panel6_protocol_draft_2026-10-15/protocol_draft.json`](../../artifacts/week8/panel6_protocol_draft_2026-10-15/protocol_draft.json), dựng bởi `scripts/analysis/panel6_protocol_draft.py`.
- Adapter: `scripts/panel6/export_kernels.py` (chạy trong venv riêng) và `scripts/analysis/panel6_families.py`. Test: `tests/test_panel6_families.py`.
- Khả thi (chỉ trên mẫu phát triển, seed 1): [`artifacts/week8/panel6_feasibility_2026-10-15/`](../../artifacts/week8/panel6_feasibility_2026-10-15/).
- Cỡ mẫu: [`artifacts/week8/panel6_power_2026-10-15/`](../../artifacts/week8/panel6_power_2026-10-15/), script `scripts/analysis/panel6_power.py`.

## 1. Nguồn họ MDP đã kiểm tra

| Nguồn | Có bộ sinh instance thật | \|S\|\|A\| ≤ 256 | Chưa dùng | Kết luận |
|---|---|---|---|---|
| **MDP Playground 1.0.0** (Rajan et al., 2021; Apache-2.0), `RLToyEnv` rời rạc | Có, theo seed | Có (S = d·A, chọn tham số) | Có | **Đề xuất** |
| **labmaze 1.0.6** (bộ sinh mê cung của DeepMind Lab; Apache-2.0) | Có, theo seed (bố cục) | Có (33–60 ô × 4 hành động) | Có | **Đề xuất**; động lực học do nhóm đặt |
| Colosseum MiniGrid-Empty/Rooms, Taxi (sampler của tác giả) | Có | **Không**: MiniGrid-Empty có cỡ 6–19, nên ≥ 144 trạng thái × 3 = 432 | Có | Loại. Ép cỡ nhỏ thì chỉ còn vài bố cục |
| rlberry 0.7.3 (Chain, GridWorld, NRoom) | **Không** (chỉ có constructor tham số) | Có | Có | Loại; thêm nữa, gói ghim Gymnasium cũ |
| Garnet, Gymnasium CliffWalking | — | — | — | Thẻ đã loại |

**Họ đề xuất:**
- **`mdpp`:**
  - đồ thị kế tiếp nối tối đa trên d tập độc lập, mỗi tập A trạng thái;
  - nhiễu chuyển đều η ~ U(0; 0,3);
  - trạng thái có thưởng theo mật độ U(0,1; 0,5) (chuỗi dài 1, không trễ);
  - trạng thái kết thúc với mật độ 0 hoặc 0,25.
- **`labmaze`:**
  - mê cung 9–11 × 9–11, 1–3 phòng;
  - đi bốn hướng, đúng hướng với xác suất p ~ U(0,7; 0,95), còn lại chia đều cho ba hướng kia; tường chặn;
  - thưởng 1 khi vào ô đích; xuất phát đều trên các ô spawn.
- Cả hai được chuyển sang dạng liên tục như panel 5.

## 2. Adapter và vân tay kernel

- **Venv riêng:** MDP Playground ghim Gymnasium 1.4 và ale-py, nên cả hai thư viện chạy trong venv riêng (`D:\Tools\venvs\mdpp`) và xuất kernel `npz`. Môi trường dự án giữ Gymnasium 1.0.0 cho panel FrozenLake đã niêm phong.
- **Kernel MDP Playground được đối chiếu với chính `step()` của thư viện:**
  - 200 bước cho mỗi cặp (s, a);
  - p-value nhị thức chính xác, hiệu chỉnh Bonferroni: nhỏ nhất 0,027 trên 60 kernel;
  - support và thưởng khớp tuyệt đối.
- **Kernel labmaze** được dựng từ bố cục, có test đơn vị (tường, ô đích, spawn).
- **CI không cần venv:** exporter ghi bảng nguồn của mọi kernel phát triển vào `source_tables.json` (bảng kế tiếp, η, thưởng; bố cục mê cung). Test dựng lại cả 120 kernel từ đó và so vân tay với kết quả đã ghi. Hợp đồng CI cấm test bị skip.
- **Phân biệt với mọi panel trước, theo hai cách:**
  - **vân tay chính xác** theo quy ước của dự án, so với 599 vân tay trong `seen_kernels()`;
  - **chữ ký bất biến khi đánh lại nhãn** trạng thái và hành động, so với 884 kernel ngoài generator có trên máy (Colosseum 2027–2029, SPIBB, SPIBB-2, forest, FrozenLake, gambler).
- **Khả thi** (60 mẫu phát triển mỗi họ, numpy seed 1, không bao giờ dùng seed của panel): 60/60 mỗi họ nằm trong giới hạn cỡ, qua bộ lọc dynamic range, có chữ ký riêng và không trùng panel trước.
  - `mdpp`: S 8–64, SA 16–256.
  - `labmaze`: S 33–60, SA 132–240.

## 3. Câu hỏi và cổng: chỉ câu hỏi mới

Thứ tự cố định F1 → F2 → F3. Đơn vị là MDP (gộp hai họ). Kiểm định max-t một phía, SE gốc, 10 000 lần bootstrap, seed 20261216.

| Họ cổng | Đối chứng | Chỉ số | GO khi |
|---|---|---|---|
| **F1** (chính) | B2 + prefix/FQE (T2) | `ibs_ge`, `ll_ext` (endpoint ngoại suy, T1) | cả hai cận dưới > 0 |
| F2 | B2 | `ibs_ge`, `ll_ext` | cả hai cận dưới > 0 |
| F3 | B2 + prefix/FQE và B2 | regret quyết định, τ = 0,8, λ = 100 | cả hai cận dưới > 0 |

- **Không kiểm định lại** OEH so với head vững (B2 + D). Đó là tương phản panel 5 đã NO-GO; nó chỉ được báo cáo mô tả.
- **Báo cáo mô tả:**
  - OEH-F, head vững, XGBoost-CF/18 và /108, LCE, B2 + LCE, prefix/FQE thô, k-NN, logistic, S4 (T4), OEH không có OEH-H;
  - endpoint đầy đủ, nhãn τ = 0,99 (T6), từng họ, từng learner, theo khoảng cách, Murphy.
- LC-PFN và TabPFN chỉ vào nếu quyết định giấy phép của T3 có trước khi đóng băng.

## 4. Cỡ mẫu

Hiệu ứng lấy theo MDP trên các panel đã mở có dự báo OEH cuối (kiểm định cuối và panel 5; 300 MDP ngoài generator):
- Hiệu ứng chuẩn hóa của F1: 0,39–0,40 khi gộp. Theo từng panel: gambler 0,32, SPIBB-2 0,27, FrozenLake 1,3–1,5.
- Power được mô phỏng bằng cách rút lại vector hiệu ứng theo MDP, đúng quy tắc max-t.
- **"Xấu nhất"** là proxy bỏ một panel ra.

| Power (xấu nhất / gộp) | 96 MDP | 144 | **192** |
|---|---:|---:|---:|
| F1, hiệu ứng × 0,75 | 0,68 / 0,84 | 0,83 / 0,95 | **0,92 / 0,98** |
| F1, hiệu ứng × 0,5 | 0,35 / 0,51 | 0,51 / 0,69 | 0,62 / 0,80 |
| F2, hiệu ứng × 0,75 | 0,76 / 0,95 | 0,91 / 0,99 | 0,97 / 1,00 |
| F3, hiệu ứng × 0,75 | 0,24 / 0,46 | 0,37 / 0,66 | 0,50 / 0,79 |

- **Đề xuất: 96 MDP mỗi họ (192 tổng).** Đây là cỡ nhỏ nhất đạt power ≥ 0,9 cho F1 với hiệu ứng × 0,75 dưới proxy xấu nhất.
- Mỗi họ có 150 ứng viên. Chọn 96 ứng viên đầu theo thứ tự danh sách, qua giới hạn cỡ, bộ lọc dynamic range và kiểm tra trùng.
- Chi phí ước tính theo panel 5 khoảng 1–2 giờ trên máy này.
- **F3 thiếu power.**

## 5. Cần con người quyết định ở H1

1. **Họ MDP:** MDP Playground + labmaze (hai nguồn mới), hay phương án một nguồn (MDP Playground tách theo d = 1 và d ≥ 2), hay họ khác.
2. **Động lực học labmaze** do nhóm đặt (bộ sinh chỉ cho bố cục), giống phép chuyển của FrozenLake. Duyệt hay sửa.
3. F1 là cổng chính; thứ tự F1 → F2 → F3.
4. Quy tắc cỡ mẫu (mục tiêu power, hệ số co) và **96 MDP mỗi họ**.
5. **F3 thiếu power:** giữ làm cổng cuối hay chuyển sang mô tả.
6. **Rủi ro nhãn suy biến.** Kernel phát triển có dynamic range lớn (MDP Playground 7,4–16,6; labmaze 2,2–8,3; ngưỡng 0,2), nên nhãn có thể dồn về x1, giống FrozenLake dồn về ">x50".
   - Phương án A: chấp nhận rủi ro.
   - Phương án B: cho phép chạy thử learner trên mẫu phát triển (seed 1) trước khi đóng băng, có công bố.
7. LC-PFN/TabPFN có vào không (cần quyết định giấy phép T3 trước).
8. Có liệt kê S4 không (chỉ mô tả; T4 khuyến nghị không đăng ký).

**Sau khi duyệt:** T9 viết runner panel 6, đóng băng `protocol.json` + `protocol.sha256` dưới namespace riêng, sinh dữ liệu và niêm phong dự báo (chốt H2). Seed dữ liệu đặt ở 30 000 000 (mdpp) và 31 000 000 (labmaze); khi đóng băng sẽ kiểm tra để không trùng với manifest của mọi panel trước.
