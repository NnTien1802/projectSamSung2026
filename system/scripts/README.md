# Gói chuyển sang thiết kế hệ thống

Gồm:
- `01_model_release.py`: kiểm kê, tạo config, khóa SHA-256 bất biến, và kiểm tra thay đổi.
- `model_lock_config.template.json`: ví dụ cấu hình yêu cầu xác nhận đường dẫn và giá trị thật.
- `SYSTEM_DESIGN_V1.md`: bản thiết kế kiến trúc, API, database, workflow và nghiệm thu.

Dùng trong Windows, từ thư mục `C:\1_projectSamSung`:

```cmd
mkdir system\scripts
mkdir system\release
copy 01_model_release.py system\scripts\01_model_release.py
python system\scripts\01_model_release.py inventory
python system\scripts\01_model_release.py init
```

Bước kế tiếp: đọc `system\release\model_inventory_v1.json` và chỉnh các trường `null` trong `system\release\model_lock_config.json`. **Chưa gọi freeze khi chưa biết hai checkpoint M2 và threshold M3.** Kết quả SHA lock chỉ là khóa file; phải chạy regression test riêng trước khi công bố release.
