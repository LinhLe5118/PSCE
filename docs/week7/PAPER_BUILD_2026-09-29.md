# Kiểm tra build paper — 29/09/2026

Đã chạy thành công ba pass pdflatex và một pass BibTeX trên nguồn paper hiện
hành. `paper/main.pdf` gồm 13 trang, bao gồm appendix; bibliography/citation
đã được xử lý, không có undefined citation/reference trong log cuối.
Còn cảnh báo font fallback và một overfull hbox 1,55 pt.

PowerShell chặn thực thi `paper/build.ps1` do execution policy hiện tại;
đã dùng trực tiếp các executable pdflatex/BibTeX đã cài, theo đúng bốn bước
trong script, không thay execution policy.

`paper/README.md` thuộc H1 freeze nên giữ nguyên snapshot lịch sử, kể cả
ghi chú PDF chưa build tại ngày handover H1. Tài liệu này cập nhật trạng thái
build sau handover mà không đổi checksum freeze. Kết quả OEH-v5 R34 đã qua
screen hậu nghiệm và được ghi trong báo cáo tuần 7 riêng; chưa tích hợp vào
paper trước review độc lập. Page budget của venue vẫn cần được chốt;
build thành công không tự xác nhận sẵn sàng nộp bài.
