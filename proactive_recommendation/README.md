# Proactive recommendation

Thư mục `C:/Users/Ai/Documents/LLMVietSage/proactive_recommendation` tập trung toàn bộ phần nghiên cứu proactive C01-F1.0: cấu hình, mã Python, controller, đặc tả, dữ liệu demo, công cụ và kiểm thử. Phần này chạy offline và không phụ thuộc `app/`, `config/`, LLM hoặc API nghiệp vụ.

```text
proactive_recommendation/
  config.py     Phiên bản, feature schema, fixture parameters/priors và seed
  controller.py State machine, poll, receipt, reply, outbox và audit
  *.py          Engine, model xác suất, adapter, snapshot, transactions và manifest
  docs/         Năm tài liệu nghiên cứu
  demo_data/    Nguồn JSON giả lập, expected cases và ingestion manifest
  scripts/      Research CLI, kiểm tra dữ liệu và sinh PDF demo
  tests/        Kiểm thử thuật toán và thí nghiệm
```

Import bằng `proactive_recommendation`, ví dụ `from proactive_recommendation.engine import decide`. Thuật toán, CLI nghiên cứu và test chỉ cần thư viện chuẩn Python. Có thể sao chép nguyên thư mục này sang nơi khác và chạy các lệnh dưới đây từ thư mục cha của nó.

Chạy từ thư mục gốc của project:

```powershell
python -B proactive_recommendation/scripts/proactive_research.py validate-implementation
python -B proactive_recommendation/scripts/proactive_research.py smoke
python -B proactive_recommendation/scripts/proactive_research.py readiness
python -B proactive_recommendation/scripts/validate_demo_data.py
```

Script `generate_demo_source_pdf.py` đọc `demo_data/` trong thư mục này và tạo `output/pdf/VSDEMO 0001-26.pdf` cũng trong thư mục này. Việc sinh PDF cần `reportlab`; xem hướng dẫn trong [demo_data/README.md](demo_data/README.md).

Các fixture D0 và dữ liệu demo là dữ liệu giả lập. Trạng thái sẵn sàng cho dữ liệu thực vẫn do manifest và các yêu cầu hiệu chuẩn quyết định. Đặc tả nằm trong [docs/00_AI_RESEARCH_INDEX.md](docs/00_AI_RESEARCH_INDEX.md).
