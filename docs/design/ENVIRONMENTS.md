# Bộ môi trường MDP dạng bảng — bộ kiểm tra đường đi

Tài liệu này mô tả bộ fixture môi trường trong `PLAN_Final.md`. Mã nguồn nằm tại
`environments/tabular_environments.py` và chỉ phụ thuộc NumPy. Theo giao thức cập nhật ngày
16/08/2026, 14 cấu hình này dùng để kiểm tra đường đi và minh hoạ; benchmark
chính `120 + 15 + 3` được sinh bởi `environments/benchmark_environments.py`.

## Phạm vi đã triển khai

Có đúng 14 cấu hình thuộc đủ ba họ môi trường:

| Họ | Cấu hình | Số lượng |
|---|---|---:|
| Combination Lock | độ dài 2, 3, 4, 5, 6, 8, 10, 12 | 8 |
| GridWorld | kích thước 4×4, 6×6, 8×8 | 3 |
| Linear Chain | độ dài 5, 11, 21 | 3 |
| Tổng | | 14 |

Combination Lock là trục thí nghiệm chính. Factory `make_combination_lock(chain_length, ...)` nhận độ dài tùy ý từ 2 trở lên, nên nhóm không bị giới hạn ở tám cấu hình đăng ký sẵn.

## Giao diện chung

Mọi factory trả về một `TabularMDP` có các trường:

| Trường | Shape/kiểu | Ý nghĩa |
|---|---|---|
| `environment_id` | `str` | ID cấu hình |
| `P` | `(S, A, S)` | `P[s,a,s_next]`, xác suất chuyển trạng thái |
| `R` | `(S, A, S)` | `R[s,a,s_next]`, thưởng của chuyển tiếp |
| `expected_rewards` | `(S, A)` | thưởng kỳ vọng `sum(P * R, axis=2)` |
| `initial_state_distribution` | `(S,)` | phân phối trạng thái đầu |
| `terminal_states` | `tuple[int, ...]` | các trạng thái kết thúc |
| `gamma` | `float` | hệ số chiết khấu, mặc định `0.99` |
| `metadata` | `dict` | họ, kích thước, seed và tham số môi trường |

`R` dùng ba chiều để thưởng có thể phụ thuộc vào trạng thái kế tiếp. Thành viên chỉ hỗ trợ reward hai chiều có thể dùng `env.expected_rewards`.

Các bất biến được `env.validate()` kiểm tra:

```python
P.shape == R.shape == (S, A, S)
P[s, a, :].sum() == 1
P[s, a, s_next] >= 0
initial_state_distribution.sum() == 1
```

Terminal state là absorbing: mọi hành động giữ nguyên terminal và thưởng sau terminal bằng 0. Thưởng hoàn thành nằm trên chuyển tiếp đi vào terminal.

## Combination Lock

- Với độ dài `L`: có `L + 1` trạng thái và 2 hành động.
- Bắt đầu ở trạng thái `0`, terminal là `L`.
- Mỗi trạng thái có một hành động đúng; đúng thì tiến một bước, sai thì quay về `0`.
- Chuỗi hành động đúng được sinh theo seed và lưu ở `metadata["correct_actions"]`.
- Chỉ chuyển tiếp vào terminal nhận thưởng `1.0`; còn lại bằng `0.0`.

Độ dài chuỗi là biến chính kiểm soát độ khó. Cùng độ dài và seed luôn tạo cùng `P`, `R` và chuỗi đúng.

## GridWorld

- Có bốn hành động: up `0`, right `1`, down `2`, left `3`.
- Start ở `(0,0)`, goal ở góc dưới phải.
- Ô tường không được đưa vào không gian trạng thái.
- Đi ra biên hoặc vào tường khiến agent đứng yên.
- Chuyển vào goal nhận `1.0`; bước khác nhận `-0.01`.
- Ba sơ đồ tường cố định nằm trong `GRID_WALLS`.

Phiên bản này deterministic và không có xác suất trượt.

## Linear Chain

- Độ dài phải lẻ và từ 5 trở lên; agent bắt đầu ở giữa.
- Hành động `0` đi trái, `1` đi phải.
- Hai đầu là terminal.
- Đi vào đầu trái nhận `0.1`, đi vào đầu phải nhận `1.0`, còn lại nhận `0.0`.

## Danh sách ID

```text
combination_lock_length_2
combination_lock_length_3
combination_lock_length_4
combination_lock_length_5
combination_lock_length_6
combination_lock_length_8
combination_lock_length_10
combination_lock_length_12
gridworld_4x4
gridworld_6x6
gridworld_8x8
linear_chain_5
linear_chain_11
linear_chain_21
```

## Cách dùng

Tạo môi trường chuẩn bằng ID:

```python
from tabular_environments import make_env

env = make_env("combination_lock_length_8", seed=42)
env.validate()
print(env.P.shape, env.R.shape)
print(env.metadata)
```

Tạo Combination Lock với độ dài khác:

```python
from tabular_environments import make_combination_lock

env = make_combination_lock(chain_length=15, seed=123, gamma=0.95)
```

Các factory còn lại là `make_gridworld(...)` và `make_linear_chain(...)`. Registry `ENVIRONMENT_REGISTRY` cho phép code thí nghiệm duyệt toàn bộ 14 cấu hình mà không hard-code ID.

## Kiểm thử

Chạy:

```powershell
python -m environments.tabular_environments
```

Self-test kiểm tra đủ 14 cấu hình, shape và xác suất của `P/R`, terminal absorbing, khả năng tái lập theo seed và ngữ nghĩa đặc trưng của cả ba họ.

## Các giả định nhóm cần xác nhận trước thí nghiệm chính thức

1. `R` có shape `(S,A,S)` thay vì `(S,A)`.
2. `gamma = 0.99`.
3. Tất cả chuyển trạng thái deterministic.
4. Hành động sai trong Combination Lock quay về start và không có step penalty.
5. GridWorld có step cost `-0.01`, goal reward `1.0`, chưa có transition noise.
6. Linear Chain dùng hai terminal reward `0.1` và `1.0`.
7. Bộ 14 kích thước hiện tại phù hợp với ma trận thí nghiệm.

Nếu nhóm đổi một giả định, cần cập nhật đồng thời factory, self-test và tài liệu này.
