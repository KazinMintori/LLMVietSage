# Domain generation

Domain nằm trong app/, ngang hàng với controllers/services/core.

| File | Trách nhiệm |
|---|---|
| schema/inference_schema.py | Pydantic khai báo request/response, field, role, length và extra fields. |
| models/generation.py | ProviderReply, ProviderChunk và StreamEvent immutable/slots, chỉ trong request. |
| models/provider.py | Protocol ModelProvider cho Ollama và fake provider. |
| models/response_preferences.py | Record phong cách cho helper offline evaluation. |
| errors.py | Lỗi có kiểu, không nhập thư viện HTTP/provider. |

Domain không có ORM/database, checkpoint, dữ liệu caller lưu lâu dài hoặc điều kiện nghiệp vụ. Không import service/controller/core/config/FastAPI/httpx.

Schema ràng buộc kiểu/field; validation cần if/else hoặc liên trường nằm trong input_validation_service. Không di chuyển cả logic vào model chỉ để đổi folder. Service không nhập ngược controller.

Checkpoint/tokenizer offline ở training/models, trọng số runtime do Ollama quản lý. Dataset schema của training riêng với schema API.

[Kiến trúc](../../README.md), [quy định và xác nhận từng file](../../AGENTS.md).
