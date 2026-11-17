# D3 — Kiểm toán “SOTA” cho ước lượng độ khó offline RL tabular stochastic

> **Trạng thái 09/10:** Các bảng kết quả BRIDGE bên dưới là ảnh chụp lịch sử tại thời điểm chỉ có arm rút gọn trên 14 MiniGrid. Kết quả full OEH trên 5 Atari đã hoàn tất và cho kết quả hỗn hợp; 5 MDP mới cùng bản full OEH trên 14 MiniGrid còn đang chạy. Dùng [ledger bằng chứng hiện hành](CURRENT_EVIDENCE_AND_RETIREMENT_2026-10-09.md) và [paper Results](../../paper/sections/06_results.tex) để lấy kết luận mới nhất. Rà soát tài liệu vẫn chưa xác minh được một hệ nguyên bản xuất trực tiếp cùng phân phối bảy lớp; không suy từ đó ra claim “vượt SOTA”.

**Ngày 08/10/2026.** Câu hỏi của nhóm được khóa trong [T1 v1.1.0](T1_DIFFICULTY_AWARE_PROTOCOL_v1.1.0_2026-10-05.md): từ pilot `D0`, dự báo phân phối lớp đầu tiên `E_l` mà learner `l` đạt ngưỡng return. Đây là độ khó **có điều kiện theo MDP, behavior, learner và ngưỡng**. Không thay nhãn bằng return, độ khó online hay cận sample complexity sau khi biết kết quả.

## Rà bài gốc và quyết định đối chứng

Tìm theo *offline RL difficulty/data-requirement prediction*, *dataset performance prediction*, *instance-dependent/gap-dependent sample complexity*, *dataset quality* và *effective horizon*; đối chiếu bài gốc/trang tác giả. Đây là **rà soát có phạm vi**, không phải chứng minh không tồn tại công trình khác. Không tìm được leaderboard hoặc phương pháp công bố nhận cùng pilot tabular và learner để xuất nguyên trạng `P(E_l=0…6)`. Vì vậy chưa có “SOTA hiện tại” cùng estimand để tuyên bố OEH phải vượt trên Brier/log loss.

| Công trình gốc | Output thực | Kiểu so sánh với `E_l` | 120 MDP | BRIDGE |
|---|---|---|---|---|
| [Arnob, Islam & Precup, 2021](https://arxiv.org/abs/2112.15578) | Đường cong thực nghiệm hiệu năng learner khi thay kích thước tập offline | Rất gần câu hỏi **bao nhiêu dữ liệu**, nhưng là phân tích hậu nghiệm sau khi train nhiều mức ngân sách; không nhận pilot để xuất phân phối dự báo `E_l`. | Không có forecaster nguyên bản để chạy trên pilot | Tương tự; một đường cong oracle tính từ nhãn test sẽ gây leakage |
| [Mateos-Melero et al., EWRL 2024](https://plg.uc3m.es/papers/mateos-melero-et-al-ewrl24.pdf), [GECCO 2025](https://plg.uc3m.es/papers/mateos-et-al-gecco2025.pdf) | CNN dự báo return của policy từ ảnh dataset cho learner/cấu hình cố định | Gần nhất về dự báo trước khi train, **khác đầu ra** và biểu diễn ảnh theo domain. Chuyển sang `E_l` là bản thích nghi; chưa xác minh được mã/weights tác giả dùng trực tiếp cho panel. | NO-GO chạy hệ nguyên bản trên MDP không có hình học ảnh đã định nghĩa | NO-GO hệ nguyên bản trên `E_l`; ảnh từ pilot và fit model mới là adaptation, nhãn test đã mở |
| [Schweighofer et al., CoLLAs 2022](https://proceedings.mlr.press/v199/schweighofer22a.html) | TQ/SACo: chất lượng quỹ đạo và độ phủ state–action chuẩn hóa | Score gốc, thử chuyển sang xếp hạng `E_l`; TQ/SACo cần reference từ MDP thật | Đã chấm trên 120 MDP; full OEH cùng hàng | Đã chấm 14/14 task; reduced OEH cùng hàng, full OEH NO-GO tài nguyên |
| [Swazinna et al., SSCI 2021](https://arxiv.org/abs/2111.13461) | ERI: tiềm năng cải thiện so với behavior | Score gốc, mục tiêu khác; giữ dấu nguyên bản và báo undefined | Đã chấm, 398/420 hàng primary xác định | Đã chấm, 42/70 pilot xác định |
| [Laidlaw et al., NeurIPS 2023](https://papers.nips.cc/paper_files/paper/2023/file/b8be628bf719550b560de8bec9456e0b-Paper-Conference.pdf), [ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/3e18f7152b5e1696e73722dc5121b6f2-Paper-Conference.pdf) | EH online/MDP thật, BRIDGE EPW/k-min | Oracle difficulty descriptor; không phải offline learner requirement | Không gọi descriptor tự tính là tác giả chạy nguyên bản | Đã chấm EPW/k-min công bố trên 14 task; reduced OEH cùng hàng |
| [Wang et al., NeurIPS 2022](https://papers.neurips.cc/paper_files/paper/2022/file/5f5f7b6080dcadced61cf5d96f7c6dde-Paper-Conference.pdf), [Li et al., AoS 2024](https://arxiv.org/abs/2204.05275) | Cận sample complexity phụ thuộc gap/coverage hoặc minimax của thuật toán dưới giả định lý thuyết | **Cận** cho bảo đảm của learner, không phải estimator phân phối thực nghiệm `E_l` của BC/CFQI/IQL | Không có dự báo nguyên bản để chấm cùng nhãn | Tương tự |

Các thuật toán CQL, IQL, BCQ, SPIBB, VI-LCB là **learner**; return của chúng không cùng loại với Brier của OEH. Chúng thuộc bảng chọn learner, không phải đối chứng cho mô hình ước lượng độ khó. BWD tiếp tục bị loại theo [T1 v1.3.0](T1_COMPARATOR_DECISION_v1.3.0_2026-10-08.md) vì action geometry nguyên bản không khớp discrete tabular.

## Kết quả đã chạy trên đúng nhãn `E_l`

### 120 MDP của nhóm

OEH **đầy đủ** được niêm phong trước nhãn; phép so sánh thứ hạng score tác giả trên 84 stochastic MDP được chấm lại trong [E3](E3_OEH_VS_PUBLISHED_METHODS_2026-10-08.md). Với TQ, Spearman của OEH/TQ trên cùng hàng là **BC 0,491/0,404; CFQI 0,727/0,343; IQL 0,699/0,603**; khoảng bootstrap chênh lệch đều dương trong cohort đã mở. ERI, SACo và điều kiện xác định/privileged được báo đầy đủ ở E3. Đây là phép **chuyển score gốc sang mục tiêu xếp hạng `E_l`**, không chứng minh thắng nhiệm vụ gốc của tác giả hoặc bảy lớp SOTA.

### BRIDGE bên thứ ba

Chỉ có `oeh_without_direct_oehb` **rút gọn**, không phải OEH đầy đủ. Mình chấm lại score gốc và reduced OEH trên cùng hàng/nhãn trong [artifact mới](../../artifacts/week8/bridge_reduced_vs_original_difficulty_exploratory_2026-10-08/rank_audit.csv). Các đầu vào được kiểm SHA-256 với manifest đã lưu; 210 dự báo = 14 task × 5 behavior × 3 learner. Với TQ, chỉ 55 hàng thuộc 11 task xác định:

| Learner | ρ TQ gốc | ρ reduced OEH cùng hàng | Δρ [bootstrap 95% theo task] |
|---|---:|---:|---:|
| BC | 0,482 | 0,498 | +0,016 [−0,001; 0,050] |
| CFQI | 0,555 | 0,762 | +0,207 [0,053; 0,317] |
| IQL | 0,577 | 0,578 | +0,002 [−0,002; 0,006] |

SACo xác định 70/70 pilot; reduced OEH có ρ cao hơn SACo trên cả ba learner ở phép xếp hạng này. ERI xác định 42/70; hướng ERI nguyên bản không có ý nghĩa “càng cao càng dễ đạt `E_l`”. EPW/k-min từ MDP thật là oracle; đầy đủ 15 hàng/CI/undefined nằm ở [CSV](../../artifacts/week8/bridge_reduced_vs_original_difficulty_exploratory_2026-10-08/rank_audit.csv). Tất cả khoảng BRIDGE là **mô tả hậu nghiệm**, vì nhãn đã mở và chỉ có 14 task. BC/IQL so TQ chưa có cận chênh lệch dương.

## Gate còn thiếu trước claim “vượt SOTA”

1. **Cùng estimand.** Cần phương pháp tác giả hoặc bản tái lập được khai báo chính xác có cùng input pilot và đầu ra xác suất `E_l`; hiện chưa xác minh được phương pháp nguyên bản nào như vậy. Mateos-style chuyển return qua budget là *adaptation*, phải fit/freeze và gọi đúng tên.
2. **OEH đầy đủ trên BRIDGE.** Lượt trước dừng ở 7/70 pilot trước khi mở nhãn do ước lượng vượt cap 72 giờ; việc dùng reduced arm không thay được phép kiểm full 108-feature. [Gate tài nguyên](../../artifacts/week8/bridge_resource_gate_addendum_2026-10-07.json) và [E1](E1_EXECUTION_RECORD_2026-10-07.md) lưu quyết định.
3. **Kiểm định mới.** 120 MDP và 14 BRIDGE task hiện đã được dùng để xem kết quả; mọi phương pháp chọn hoặc tinh chỉnh sau đó trên các panel này chỉ là exploratory. Muốn cận dương confirmatory trước một đối chứng ngoại nghiên cứu, phải khóa phương pháp, tuning, quyền truy cập và task mới trước khi mở nhãn.

Do đó chưa ghi “OEH vượt SOTA” trong paper. Bằng chứng hiện tại mạnh nhất là kết quả sealed trên 120 MDP so với **đối chứng nội bộ**, cộng phép đối chiếu score tác giả trên `E_l` theo thứ hạng và transfer BRIDGE rút gọn với kết quả lẫn chiều.
