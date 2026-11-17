# Tuần 5

- [`H1_SCIENTIFIC_VALIDATION_2026-09-15.md`](H1_SCIENTIFIC_VALIDATION_2026-09-15.md)
  — biên bản kiểm chứng khoa học OEH của Hưng: ma trận claim–assumption–evidence,
  đối chiếu LaTeX với code, phân biệt statistic saturation với learner
  learnability, và nội dung bàn giao để Linh tích hợp paper.
- [`../../artifacts/week5/s02_handover_2026-09-12/`](../../artifacts/week5/s02_handover_2026-09-12/)
  — danh sách nghiệm thu và kết quả kiểm tra S02 theo quy trình của Linh.
- [`../../audit_week5_training.py`](../../audit_week5_training.py) — chương trình
  kiểm tra tự động cho bộ dữ liệu phát triển và đầu ra bàn giao trước khi phân tích.
- [`../../scripts/analysis/analyze_learner_results.py`](../../scripts/analysis/analyze_learner_results.py)
  — kiểm toán D2 ở mức job, tái tính learning curves, phân phối `N_emp` và các
  bảng so sánh CFQI/IQL/BC từ bộ artifact training đầy đủ.
- [`../../artifacts/week5/d2_learner_analysis_2026-09-15/`](../../artifacts/week5/d2_learner_analysis_2026-09-15/)
  — báo cáo D2 hoàn chỉnh, tái tính từ dữ liệu mức job trong gói kết quả bàn giao.
- [`../../artifacts/week5/d2_training_audit_2026-09-15/`](../../artifacts/week5/d2_training_audit_2026-09-15/)
  — kiểm toán checksum, schema, lưới 43.200 job và 480 nhãn `N_emp`.
- [`../../scripts/validation/validate_full_pipeline_results.py`](../../scripts/validation/validate_full_pipeline_results.py)
  — kiểm định lại full pipeline, tách oracle không xác định theo hợp đồng khỏi
  lỗi oracle bất ngờ và kiểm tra mẫu số cùng checksum artifact.
- [`../../artifacts/week5/pipeline_validation_2026-09-15/`](../../artifacts/week5/pipeline_validation_2026-09-15/)
  — kết quả sửa lỗi trên gói ngày 14/09: toàn vẹn pass với 59 oracle exclusions,
  0 lỗi oracle bất ngờ; kết quả khoa học vẫn âm vì 266/266 đường cong bằng phẳng.
