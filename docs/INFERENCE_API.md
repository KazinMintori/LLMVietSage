# Sinh phản hồi

Caller gửi yêu cầu sinh phản hồi, model phục vụ qua Ollama.

## Chat JSON

POST /api/chat nhận message hoặc messages; chỉ gửi một trong hai.

    {
      "messages": [
        {"role":"user","content":"Tôi tên Linh."},
        {"role":"assistant","content":"Chào Linh."},
        {"role":"user","content":"Tôi tên gì?"}
      ]
    }

Role caller chỉ là user/assistant; lượt cuối user. Không chấp nhận system/developer/tool, session, conversation, model override hoặc extra fields. Không thay đổi/cắt/dedupe nội dung input.

Context đã được caller đóng gói trong content/message. Không có field question/context riêng; system prompt do config/prompts.py cung cấp.

Response có reply, model và usage. Thêm json_schema để yêu cầu JSON object; reply khi đó là object. Client kiểm tra JSON parse hợp lệ và object, chưa xác thực đầy đủ tất cả ràng buộc JSON Schema.

Schema Pydantic kiểm tra kiểu/field. InputValidationService kiểm tra liên trường/blank; invalid input trả HTTP 422 trước provider. Model không có/kết quả lỗi trả 503 với thông báo chung.

## SSE

POST /api/chat/stream nhận cùng body nhưng không hỗ trợ json_schema; gửi nó trả 422 trước khi tạo stream.

    event: token
    data: {"text":"Xin chào"}

    event: done
    data: {"model":"...","usage":{}}

Lỗi provider/output tạo sự kiện error; không phát done sau error. HTTP của stream có thể 200 dù có error; caller phải đọc sự kiện. Stream và kết nối provider được đóng khi hoàn tất/lỗi/hủy.

Các block thinking có tag đầy đủ hoặc chia chunk được lọc, tag mở chưa đóng không phát nội dung. Không coi cơ chế này là bảo đảm chống mọi dạng model tiết lộ thông tin.

## Metadata và ranh giới

- GET /api/info: provider/model, history=caller_supplied, rag_enabled=false (không có retrieval trong app).
- GET /api/health/live: liveness API, không chứng minh Ollama/model sẵn sàng.
- POST /api/embeddings và các route database/retrieval/UI không tồn tại.

Caller quản lý history, tra cứu và fallback workflow bên ngoài nếu có. App chỉ bảo vệ hợp đồng generation và vòng đời inference: nhận input, gọi Ollama và trả về kết quả.

Code: controllers khai báo route/HTTP formatter; services xử lý; domain giữ schema/record/protocol/lỗi; core serialize SSE dùng chung. Xem [README](../README.md) và [AGENTS](../AGENTS.md).
