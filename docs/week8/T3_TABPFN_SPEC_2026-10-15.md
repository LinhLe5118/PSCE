# T3 — Spec của baseline TabPFN v2 (viết trước khi chạy, 15/10/2026)

Thẻ T3 của kế hoạch lên hội nghị (bộ agent kit, lưu ngoài repo); trả lời W7. Quyết định H1 (15/10): **chỉ dùng TabPFN v2, bỏ LC-PFN**. Spec này cố định mọi bậc tự do trước lần chạy đầu tiên.

## 1. Mô hình và phiên bản

| | Giá trị |
|---|---|
| Gói | `tabpfn==9.2.0` (code Apache-2.0), `torch==2.14.1+cpu`, venv riêng `D:\Tools\venvs\tabpfn` |
| Trọng số | TabPFN **v2** (`ModelVersion.V2`): HuggingFace `Prior-Labs/TabPFN-v2-clf` revision `f851f2a3c941544733b712d8c0f96dfae9b28862`, file mặc định `tabpfn-v2-classifier-finetuned-zk73skhh.ckpt`. Repo không bị khóa (gated = false), nên không cần token |
| Giấy phép trọng số | Prior Labs License 1.1: Apache 2.0 cộng điều 10 (ghi nguồn khi phân phối trọng số hoặc sản phẩm chứa trọng số). Đánh giá rồi công bố kết quả là được phép. Không commit trọng số; lưu một bản giấy phép trong artifact; ghi "Built with PriorLabs-TabPFN" và trích Hollmann et al. (2025) ở nơi có kết quả |
| Thiết lập | `TabPFNClassifier.create_default_for_version(ModelVersion.V2, device="cpu", random_state=0)` (`n_estimators="auto"`), `TABPFN_ALLOW_CPU_LARGE_DATASET=1` |

Bản 2.5 trở lên bị loại vì bị khóa và có giấy phép phi thương mại.

## 2. Bài toán

- **Đặc trưng:** 18 thống kê B2 (`compare_difficulty.FEATURES_B2`), giống B2 matched.
- **Huấn luyện:** 720 MDP meta-training (`meta_panel(sort=False)`, hàng primary). **Mỗi learner một classifier** (BC, CFQI, IQL), nhãn là lớp yêu cầu 7 lớp.
- **Đầu ra:** luật 7 lớp. Lớp không xuất hiện trong nhãn huấn luyện của một learner nhận xác suất 0; khi chấm điểm áp sàn 1e-12 như mọi phương pháp.
- **Không chọn tham số nào**, nên mỗi panel đã mở là một kiểm tra ngoài mẫu; không cần LOPO.

## 3. Đánh giá

- **Panel:** 11 panel đã mở — cohort 7, 8; Colosseum 2027, 2028; SPIBB; 4 panel kiểm định cuối; FrozenLake, gambler.
- **Điểm:** endpoint đầy đủ và endpoint ngoại suy (bộ chấm T1), trung bình theo MDP.
- **So với:** B2, head vững, OEH-F, OEH, B2 + prefix/FQE (T2). Lợi thế kèm KTC 95% (`extrapolation_endpoint.gains`).
- **Panel 6:** protocol đã đăng ký TabPFN v2 trên B2 là phương pháp mô tả có điều kiện. Dự báo được thêm và niêm phong riêng **trước** bước nhãn (H3), bằng đúng spec này.

## 4. LC-PFN (bỏ, quyết định H1)

LC-PFN cần Python < 3.12 và torch ≤ 1.11, và khớp đường cong từ chỉ 4 prefix của pilot. Bài sẽ ghi lý do không dùng.
