# BRIDGE reduced: kết quả kiểm định độc lập
14 MDP BRIDGE do bên thứ ba công bố; 70 log offline và nhãn do nhóm sinh theo giao thức đã khóa.
Phương pháp đối chiếu: B2 matched với ablation OEH bỏ OEH-B trực tiếp; không suy ra hiệu năng của OEH đầy đủ.
Brier/log loss càng thấp càng tốt; chênh lệch âm nghĩa là nhánh OEH có lỗi thấp hơn B2.
## Điểm dự báo và quyết định budget
| Learner | Method | Brier | Log loss | >x50 | Budget chọn TB | Đạt mục tiêu |
|---|---|---:|---:|---:|---:|---:|
| behavior_cloning_tabular | b2_matched | 0.022425 | 0.169015 | 0.900 | 49.57 | 0.100 |
| behavior_cloning_tabular | oeh_without_direct_oehb | 0.044523 | 0.323741 | 0.900 | 50.00 | 0.100 |
| cfqi_tabular | b2_matched | 0.155727 | 1.630302 | 0.600 | 50.00 | 0.400 |
| cfqi_tabular | oeh_without_direct_oehb | 0.081931 | 1.037709 | 0.600 | 48.71 | 0.400 |
| iql_tabular | b2_matched | 0.025290 | 0.274767 | 0.871 | 49.00 | 0.129 |
| iql_tabular | oeh_without_direct_oehb | 0.015152 | 0.205251 | 0.871 | 45.43 | 0.114 |

## Chênh lệch ghép cặp theo MDP
CI bootstrap 95% lấy mẫu lại 14 MDP, giữ năm behavior của mỗi MDP cùng nhau.
| Learner | Metric | OEH − B2 | CI 95% |
|---|---|---:|---:|
| behavior_cloning_tabular | brier | +0.022098 | [+0.007196; +0.037803] |
| behavior_cloning_tabular | log_loss | +0.154725 | [+0.040160; +0.275078] |
| cfqi_tabular | brier | -0.073796 | [-0.096982; -0.049428] |
| cfqi_tabular | log_loss | -0.592592 | [-0.744394; -0.429175] |
| iql_tabular | brier | -0.010137 | [-0.018050; -0.002443] |
| iql_tabular | log_loss | -0.069516 | [-0.126440; -0.011804] |

Censoring `>x50`, chi phí tính và mọi kết quả âm được giữ nguyên. TQ/SACo dùng MDP thật thuộc bảng oracle riêng; không xem điểm gốc của chúng là forecaster bảy lớp cùng quyền truy cập.
