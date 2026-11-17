# Mã nguồn được phục hồi ngày 04/10/2026

Mã nguồn phát triển và tài liệu đã bị dọn khỏi cây thư mục chính được phục hồi
từ commit `e715e3c3e8e0565ef4d747e74326c807a5cec0d5`. Các file đã có trong bản
`main` ngày 04/10 được giữ lại. Không thay đổi dự báo đã khóa, outcome, protocol,
bảng điểm hay verdict của các đợt thí nghiệm.

Các thư mục được phục hồi gồm `scripts/data/`, `environments/`, `config/`,
`evaluation/`, `pilot/`, các module Python trong `code/`, các script huấn luyện,
adapter thuật toán bên ngoài và tài liệu phát triển. Tài liệu lịch sử mô tả các
giai đoạn tại thời điểm viết; kết quả hiện hành vẫn nằm ở README gốc và
`artifacts/oeh_confirmatory_unseen_120_2026-10-04/`.

`config/restored_runtime_manifest.json` ghi nguồn và SHA-256 của 308 file được
phục hồi, cùng 204 đầu vào mã nguồn để dựng runtime đã đóng băng. Những con số
này không bao gồm các file hướng dẫn và công cụ phục hồi mới. Các file đã có
trên main không bị ghi đè khi phục hồi mã nguồn.

## Tìm mã sinh dữ liệu

- `scripts/data/generate_offline_datasets.py`: thu thập và lưu trải nghiệm.
- `environments/benchmark_environments.py`: tạo các môi trường.
- `environments/behavior_policies.py`: năm cách thu thập trải nghiệm.
- `protocol_config.json`: cấu hình thí nghiệm gốc.
- `scripts/training/run_empirical_requirement.py`: chạy các thuật toán tabular.

Các bước chạy thử và cách đọc `.npz` nằm trong [scripts/data/README.md](../scripts/data/README.md).

## Dựng lại runtime mà các runner mới tham chiếu

Các runner ngày 30/09–04/10 tham chiếu mã gốc qua
`tmp/oeh_development_2026-09-30/`. Clone mới trước đây thiếu thư mục này.
Lệnh sau dựng nó từ các file đã được đưa vào Git, kiểm tra checksum trước khi
chép và từ chối ghi đè bất kỳ bản sao nào có nội dung khác:

```shell
python -m scripts.restore_frozen_runtime
python -m scripts.restore_frozen_runtime --check
```

Runner sẽ tự dựng/kiểm tra runtime khi gọi `archive_imports()`. Chạy runner trong
một tiến trình Python mới để dùng đúng các module đã khóa. Bộ sinh dữ liệu trong
runtime dùng bản `requirements.txt` gốc được giữ riêng tại
`artifacts/provenance/runtime_inputs/requirements.txt`; requirements gốc hiện
hành của package OEH không bị thay đổi.

Ba file `generate_offline_datasets.py`, `e1_pipeline.py`, `e1_external.py` được
khôi phục đúng checksum mà protocol của cohort 04/10 đã ghi. Công cụ phục hồi
chỉ chép mã nguồn, không sinh lại outcome hay mở dữ liệu kiểm chứng.

## Dữ liệu chưa có trong Git

Thư mục `datasets/`, buffer `.npz` lớn, store kết quả từng job, các checkout
thuật toán của bên thứ ba và cache vẫn là dữ liệu cục bộ. Việc phục hồi mã nguồn
không chứng minh đã tái huấn luyện toàn bộ thí nghiệm. Các thuật toán neural
còn cần thư viện và checkout bên ngoài đúng phiên bản như adapter quy định.
Các nguồn và kết quả cũ khác tiếp tục có thể lấy từ lịch sử Git.
