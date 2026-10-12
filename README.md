# LLMVietSage — engine sinh phản hồi LLM

LLMVietSage là engine sinh phản hồi từ LLM:
- **Nhận vào**: Payload gồm câu hỏi/nội dung (`message` hoặc danh sách `messages` chứa lịch sử hội thoại, thông tin tham chiếu/context).
- **Gửi đi**: Gửi yêu cầu inference tới model LLM phục vụ trên Ollama.
- **Trả về**: Phản hồi được sinh từ LLM (dưới dạng văn bản hoặc JSON object ở endpoint chat, hoặc luồng SSE token tăng dần ở endpoint stream).

Dự án là tầng inference thuần túy: không sở hữu kho tri thức/lịch sử hội thoại dài hạn, không có database, ORM, ingestion, vector store, retrieval, persistence hội thoại, embedding endpoint hay Generative UI. System prompt hiện vẫn chuyên tư vấn du lịch vì config được giữ nguyên.

## Cấu trúc đã triển khai

    app/
      main.py
      controllers/
        inference_controller.py        Route/dependency, gọi service
        inference_errors_controller.py HTTP formatter cho lỗi đã phân loại
      services/
        inference_service.py           Use case, prompt, lifecycle, SSE events
        input_validation_service.py    Validation cần điều kiện/liên trường
        llm_service.py                 Chính sách sinh, usage, reasoning filter
        ollama_service.py              HTTP pool, payload, NDJSON, lỗi provider
        generation_budget_service.py   Context/output budget
        response_sanitization_service.py Text/JSON object và thinking tags
        response_style_service.py      Helper dùng test/offline eval
      domain/
        errors.py
        schema/inference_schema.py     Pydantic field/role/response declarations
        models/generation.py           ProviderReply, ProviderChunk, StreamEvent
        models/provider.py             ModelProvider protocol
        models/response_preferences.py Record phong cách
      core/
        streaming.py                   SSE encoder dùng chung

Luồng thực thi: controller → InferenceService → LLMService → OllamaService → Ollama. Main lắp ráp dependency; core không phụ thuộc nghiệp vụ. Record domain dùng immutable dataclass/slots khi phù hợp.

Mọi file service có hậu tố _service.py. Quyết định if/else/match/ternary chỉ nằm trong services. Schema khai báo kiểm tra kiểu/field; kiểm tra liên trường thực hiện trước provider trong validation service. Controller không có nhánh xử lý.

Các helper UI/retrieval/hội thoại cũ, legacy code/test và tài liệu monolith đã xóa. Test lịch sử/security kiểm tra API đang chạy thay vì module đã bỏ.

## Cấu hình, triển khai và dữ liệu

Model inference mặc định là `qwen3.5:4b`; đổi model bằng OLLAMA_CHAT_MODEL trong env; API không nhận model override trong request. Các field embedding cũ đã bỏ theo yêu cầu cấu hình riêng. `.env` và `.env.example` nằm ở root; chỉ sửa khóa được yêu cầu trong env đang có. `config/conversation.py` vẫn là cấu hình cũ không dùng trong runtime.

Compose ở root, chỉ triển khai Ollama. Startup bảo đảm model chat từ OLLAMA_CHAT_MODEL có sẵn; không tạo API Python. Compose và API dùng chung khóa OLLAMA_CHAT_MODEL. Model weights đang có được giữ; env thật chỉ cập nhật khóa model theo yêu cầu.

Training/data/golden và proactive độc lập được giữ. Dữ liệu tham khảo không tự thành SFT label. Xem [training](training/README.md) và [proactive research](proactive_recommendation/README.md).

## Chạy engine

Chạy từ project root trong môi trường Python inference:

    python -m pip install -r requirements.txt
    python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

Nếu chưa có `.env`, sao chép `.env.example` thành `.env` và kiểm tra URL phù hợp. Ollama phải đang chạy và đã có `qwen3.5:4b`.

Tùy chọn triển khai runtime bằng Docker; lệnh này có thể tải model thiếu:

    docker compose -f docker-compose.production.yml up -d --build

Nếu Docker đã hỗ trợ NVIDIA GPU:

    docker compose -f docker-compose.production.yml -f docker-compose.gpu.yml up -d --build

## Giao tiếp API và luồng dữ liệu

Caller gửi câu hỏi/context đã đóng gói trong message hoặc lịch sử messages. API không có field question/context riêng; system prompt do cấu hình quản lý. Caller giữ history và gửi lại các lượt liên quan; service không nhớ request trước.

Ví dụ một lượt, với dữ kiện minh họa hư cấu:

    {
      "message": "Thông tin tham chiếu: bảo tàng hư cấu A mở 09:00–17:00. Câu hỏi: A mở lúc nào?"
    }

| Endpoint | Chức năng |
|---|---|
| POST /api/chat | Reply text hoặc JSON object khi có json_schema. |
| POST /api/chat/stream | SSE token/done/error; không nhận json_schema. |
| GET /api/info | Model/provider, caller giữ history, không retrieval nội bộ. |
| GET /api/health/live | Tiến trình API sống; không xác nhận model sẵn sàng. |

Response JSON gồm reply, model, usage. Input sai trả 422; provider/kết quả model lỗi trả 503 ở endpoint không stream. Stream có thể HTTP 200 rồi phát error và không phát done. Caller xử lý và định dạng lỗi cho người dùng cuối nếu cần.

POST /api/embeddings đã bỏ vì không thuộc phạm vi sinh phản hồi. API không thực hiện retrieval hay embedding nội bộ.

Hợp đồng chi tiết: [INFERENCE_API.md](docs/INFERENCE_API.md).

## Tối ưu và giới hạn thực tế

- Một HTTP pool dùng lại kết nối; không tải model theo từng request.
- Chọn context/output budget chung; giữ đầy đủ context khi output budget là chế độ âm/unbounded.
- Stream tăng dần; đóng provider khi kết thúc, lỗi, hủy hoặc client dừng đọc. Usage/options cô lập theo request.
- Kiểm tra NDJSON, completion flag, JSON object và số liệu usage; không coi stream đứt giữa chừng là thành công.
- Bỏ dependency/logic ngoài phạm vi sinh phản hồi; không nạp embedding model hoặc thực hiện retrieval.
- Đo latency chỉ tính stream có token và done; lỗi/stream thiếu done không đi vào thống kê thành công.

Token planning vẫn là heuristic ký tự, không phải tokenizer chính xác. JSON object chưa được xác thực toàn bộ theo JSON Schema caller. Helper response_style dùng cho offline eval, không tự thêm vào system prompt runtime.

Các kiểm tra mock/offline không chứng minh chất lượng model, mức grounding hoặc tốc độ GPU thật. Chưa chạy benchmark model/training trong lần refactor này.

## Xác nhận mọi file và kiểm thử

Quy trình kiểm tra bắt buộc, không chỉ kiểm tra những file vừa sửa:

    python -B scripts/verify_architecture.py --report runtime/architecture-check.json
    python -B scripts/run_inference_tests.py
    python -B -m unittest discover -s tests -v
    python -B scripts/run_evals.py --mode offline --report runtime/generation-evals.json

Báo cáo architecture có danh sách source từng file, phân loại artifact, route inventory, config hash và vi phạm. Tool trả exit code khác 0 khi vi phạm; CI cũng chạy nó. Với config cần giữ nguyên, dùng baseline hash chụp trước khi sửa qua tùy chọn --config-baseline.

Tool kiểm tra syntax/name/layer/condition/dependency; người thực hiện vẫn phải kiểm tra trách nhiệm và tài liệu/data. Đừng coi artifact “classified” là dataset/model đã đạt chất lượng.

[Scripts benchmark](scripts/benchmark_local.py) và [đo latency](scripts/measure_model_latency.py) là công cụ chủ động, chỉ chạy khi cần đo model thật. Offline live-quality probe chưa có adapter thực thi; báo skip khi chưa chạy.


Luật chi tiết, gồm checklist xác nhận mọi file và điều kiện báo hoàn tất, ở [AGENTS.md](AGENTS.md).
