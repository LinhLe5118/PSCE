# OEH-v3: frozen target-return risk protocol

Ngày khóa: **20/09/2026**. Công thức, feature block, cohort generator và primary
endpoint trong tài liệu này phải được commit trước khi sinh fresh holdout.

## Mục tiêu và estimand

OEH-v3 dự báo lượng dữ liệu cần để policy QVI độ sâu `k` đạt sai số return chuẩn
hóa không quá `epsilon=0.05` trong **action support quan sát được của pilot**.
Đây là đại lượng dự báo độ khó, không phải confidence certificate và không phải
định lý instance-optimal KL. Không được diễn giải nó thành khả năng tối ưu trên
action chưa từng được behavior policy hỗ trợ.

Mô-đun nhận duy nhất transition pilot và phân phối trạng thái đầu. Nó không nhận
learner return, success label, training buffer, `N_emp`, optimal return thật hay
MDP evaluation đã khóa.

## Công thức

Với policy ứng viên ở độ sâu `k`, tại trạng thái `s` có ít nhất hai action trong
support, gọi `a*` và `a'` là top và runner-up trong mô hình pilot. Bellman target
được chuẩn hóa vào `[0,1]`:

```text
Z = (1-gamma) [ R/Rmax + gamma V_leaf(S') ]
Delta_s,k = (1-gamma) [ Q_k(s,a*) - Q_k(s,a') ]
d_s,a = n_s,a / N_pilot
v_tilde_s,a = max(sample_variance(Z_s,a), 1/[4(n_s,a+1)])
```

Variance floor giảm về 0 theo số mẫu và ngăn một quan sát duy nhất bị hiểu sai
là transition/reward tất định. Từ hai thành phần của empirical Bernstein, scale
phân biệt action là

```text
tau_s,k = max(
  2 [v_tilde_s,a*/d_s,a* + v_tilde_s,a'/d_s,a'] / Delta_s,k^2,
  7 [1/d_s,a* + 1/d_s,a'] / (3 Delta_s,k)
)
```

Đây là phần tương ứng với sample term của EH gốc: gap nhỏ, variance lớn hoặc
coverage thấp đều làm thời gian phân biệt tăng. Xác suất sai thứ tự được mô hình
hóa bằng tail đơn điệu

```text
p_s,k(N) = min(1, 2 exp[-N/tau_s,k]).
```

Ảnh hưởng của sai action lên return dùng performance-difference weighting:

```text
w_s,k = min(1,
  d_pi_k(s) [Q_k(s,a*)-Q_k(s,a')]
  / [(1-gamma) Range_support]
).
```

`r_plan(k)` là regret của policy QVI tại độ sâu `k` so với optimum trong cùng
support, chuẩn hóa bằng `Range_support`. Risk curve và requirement là

```text
R_k(N) = r_plan(k) + sum_s w_s,k p_s,k(N)
N_hat_k = inf { N >= 0 : R_k(N) <= epsilon }.
```

OEH giữ đúng dạng của effective horizon:

```text
OEH(k) = (k+1) + log_A N_hat_k,
```

trong đó `k+1` là QVI index theo quy ước của paper gốc. Trường hợp
`N_hat_k=0` là `epsilon_trivial`; adapter gán evidence term bằng 0 để tránh
`log(0)`, và không gọi đây là một requirement dương.

## Vì sao không còn lỗi bão hòa cũ

- Requirement được suy ra từ risk curve cho **future total sample size**; nó
  không hỏi pilot hiện tại đã tự chứng nhận gap hay chưa.
- Không có phép chặn `max(1, multiplier)` làm mọi trường hợp dồn về 1.
- State có đúng một action trong support không có bài toán xếp hạng action, nên
  đóng góp 0. V2 đã coi trường hợp này là `unsupported` và làm mất gần toàn bộ
  profile.
- Các state/action được cộng theo ảnh hưởng return; một state hiếm không tự
  động chi phối toàn bộ profile bằng phép `max`.
- Variance, behavior frequency, gap, occupancy và planning depth vẫn có vai trò
  riêng, đo được và kiểm thử đơn điệu.

## Cross-fit và feature block

Requirement dùng toàn bộ pilot vì đây là forecast plug-in. Chia trajectory làm
hai nửa chỉ tạo biến `policy_instability`: discounted occupancy mass mà policy
từ mỗi nửa khác policy full-pilot. Cách này không vứt rare action chỉ vì nó vắng
mặt trong một nửa dữ liệu.

B3 chỉ thêm bốn biến đã khóa trước:

1. `log1p(N_hat)` nhỏ nhất;
2. QVI index được chọn, chuẩn hóa theo chiều dài profile;
3. risk tại đúng kích thước pilot hiện tại;
4. policy instability giữa hai trajectory half.

Việc giảm từ 11 biến v2 xuống 4 biến v3 xử lý tỷ lệ số biến/số cụm MDP quá cao.

## Leakage firewall và fresh holdout

1. Unit test và mechanism smoke được phép dùng transition pilot cũ nhưng không
   được join với learner outcome cũ.
2. Commit khóa phải chứa công thức, feature block, generator 60 MDP, phép chia
   fold và endpoint thống kê.
3. `scripts/data/generate_oeh_v3_holdout.py` từ chối chạy nếu working tree bẩn
   và ghi commit khóa vào manifest.
4. Fresh cohort dùng 60 development MDP mới, đúng tỷ lệ family 16/20/16/8,
   namespace seed mới và không trùng bất kỳ evaluation instance nào.
5. Pilot và training dùng các namespace seed tách biệt đã đăng ký.
6. Feature được hoàn tất từ pilot trước khi join `N_emp` và return.
7. Nested CV group theo base MDP; mọi policy family của một MDP ở cùng fold.
8. Primary OEH endpoint duy nhất là paired cluster-bootstrap 95% CI của pooled
   mean log-likelihood `B3-B2`; pass khi `CI_low > 0`. Integrated Brier là
   secondary endpoint. Kết quả theo từng learner vẫn được báo cáo nhưng không
   tạo điều kiện bắt buộc cả ba learner đồng thời significant trên hai metric.
9. Selector B3 so với best-fixed được đánh giá riêng tại budget `10|S||A|`.
10. Fresh holdout là single-use. Nếu kết quả âm, không được sửa v3 rồi chạy lại
    trên cùng cohort.

## Mechanism smoke trước khi khóa

Trên 75 nhóm x10 của development cũ, chỉ đọc pilot transitions:

- 75/75 có depth được chọn hữu hạn;
- 19 là `epsilon_trivial` và 56 có requirement dương;
- 56/56 requirement dương là khác nhau, từ `16,846.2571` đến
  `220,595,742.1188`;
- profile có 797 giá trị `risk_at_pilot` và 322 giá trị
  `policy_instability` khác nhau sau khi làm tròn 12 chữ số.

Smoke này chỉ bác bỏ lỗi cơ học “0 identified” và saturation hằng số. Nó không
là bằng chứng predictive vì outcome cũ không được đọc trong bước này.

## Giới hạn tuyên bố

Không một estimator data-only nào có thể bảo đảm hoạt động trên mọi stochastic
tabular MDP khi behavior support bằng 0 ở action quan trọng. OEH-v3 giải quyết
điều này bằng cách định nghĩa estimand support-constrained và báo rõ phạm vi,
không bằng cách nội suy giả cho action chưa quan sát. Claim được phép là
predictive utility trên distribution benchmark nếu fresh holdout pass; claim về
full-MDP optimality, finite-sample certificate hoặc exact EH theorem vẫn bị cấm.
