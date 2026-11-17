# Sinh dữ liệu offline RL

File chính: `generate_offline_datasets.py`.

- `collect_trajectories()` chạy tác nhân và ghi lại từng bước.
- `generate_benchmark_bundle()` tạo dữ liệu cho các môi trường, cách chọn hành động,
  luồng pilot/huấn luyện và các mức ngân sách.
- Mỗi file `.npz` chứa `state`, `action`, `reward`, `next_state`, `done`, `terminal`,
  `truncated`, `trajectory_id`, `timestep`.
- `size` là số bước được ghi, không phải số lượt. Các ngân sách nhỏ dùng phần đầu
  của cùng một nhật ký lớn hơn. Pilot và huấn luyện dùng seed riêng.

## Chạy thử sau khi clone

Chạy từ thư mục gốc của repo, với Python 3.12:

```shell
python -m pip install -r requirements-data.txt
python -m scripts.data.generate_offline_datasets --smoke-test
python -m scripts.data.generate_offline_datasets --run-mode review --split development --limit-instances 1 --pilot-seeds 0 --training-seeds 0 --max-size 100 --horizon 30 --deviation-reason "development data preview" --output datasets/data_preview
```

Lệnh thứ hai tạo dữ liệu trong một thư mục mới. Nếu thư mục đã có dữ liệu, chọn
một tên khác để tránh trộn các lần chạy. Thư mục `datasets/` vẫn được bỏ qua bởi Git.

Đọc vài bước đầu bằng NumPy:

```python
from pathlib import Path
import numpy as np

path = next(Path("datasets/data_preview").glob("*.npz"))
with np.load(path, allow_pickle=False) as data:
    print("File:", path)
    print("Số bước:", len(data["state"]))
    for i in range(min(5, len(data["state"]))):
        print({name: data[name][i].item() for name in data.files})
```

Đây là dữ liệu chạy thử phục vụ đọc mã. Nó không thay thế dữ liệu hay kết quả của
đợt kiểm chứng đã khóa. Xem [hướng dẫn phục hồi runtime](../../docs/RESTORED_RUNTIME.md).
