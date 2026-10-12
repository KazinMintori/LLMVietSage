# Kiến trúc hệ thống LLMVietSage

## 1. Nhiệm vụ của Engine suy luận LLM

Dự án đóng vai trò là engine sinh phản hồi từ mô hình ngôn ngữ lớn (LLM):
- **Nhận vào**: Câu hỏi, thông tin tham chiếu (context) và lịch sử hội thoại do caller cung cấp (qua trường `message` hoặc `messages`).
- **Gửi đi**: Gửi yêu cầu inference tới model LLM phục vụ cục bộ trên Ollama.
- **Trả về**: Phản hồi được sinh từ LLM (dưới dạng văn bản hoặc JSON object ở endpoint chat, hoặc Server-Sent Events (SSE) stream tăng dần ở endpoint chat/stream).

Runtime hoạt động hoàn toàn không trạng thái (stateless), không lưu trữ câu hỏi, context, lịch sử hội thoại hay kho tri thức. Không sử dụng ORM, cơ sở dữ liệu, vector store hay retrieval ngoài.

---

## 2. Cấu trúc cây thư mục dự án

```text
.
├── app/                                       # Ứng dụng chính: chạy API suy luận LLM sinh phản hồi, không lưu DB/lịch sử
│   ├── main.py                                # Điểm khởi động FastAPI: composition root, cấu hình nạp model khi mở app, dọn dẹp khi tắt app
│   ├── controllers/                           # Tầng đón HTTP: chỉ tiếp nhận và chuyển tiếp, tuyệt đối KHÔNG viết if/else
│   │   ├── inference_controller.py            # Khai báo các đường dẫn /api: /chat, /chat/stream, /info, /health/live
│   │   └── inference_errors_controller.py     # Định dạng lỗi trả về mã HTTP chuẩn (422 khi sai dữ liệu, 503 khi model lỗi)
│   ├── services/                              # Tầng nghiệp vụ: NƠI DUY NHẤT trong app được phép viết logic điều kiện if/else
│   │   ├── inference_service.py               # Nhạc trưởng điều phối: ghép prompt, chọn mode chat/stream, quản lý lifecycle
│   │   ├── input_validation_service.py        # Soát lỗi đầu vào: bắt buộc chọn message hoặc messages, lượt cuối phải là user
│   │   ├── llm_service.py                     # Quản lý gọi model: thống kê token theo từng request, nạp trước (warmup) model
│   │   ├── ollama_service.py                  # Kết nối mạng với Ollama: tái sử dụng HTTP pool, đọc luồng stream NDJSON
│   │   ├── generation_budget_service.py       # Cân đối VRAM: tự co context (2048) cho câu ngắn, tăng token cho câu hỏi lịch trình
│   │   ├── response_sanitization_service.py   # Lọc sạch thẻ suy luận <think>...</think>, kiểm tra định dạng JSON hợp lệ
│   │   └── response_style_service.py          # Nhận diện ngôn ngữ (Việt/Anh/Trung) và giọng văn (dùng cho test/eval offline)
│   ├── domain/                                # Tầng dữ liệu cốt lõi: độc lập 100%, không phụ thuộc FastAPI hay thư viện ngoài
│   │   ├── errors.py                          # Phân loại các kiểu lỗi: ModelUnavailableError, InvalidRequestError,...
│   │   ├── README.md                          # Tài liệu nội bộ hướng dẫn quy ước và ranh giới tầng domain
│   │   ├── models/                            # Cấu trúc dữ liệu thuần (dataclass bất biến, không sửa sau khi tạo):
│   │   │   ├── generation.py                  #   Gói kết quả trả về từ model: ProviderReply, ProviderChunk, StreamEvent
│   │   │   ├── provider.py                    #   Khung giao diện (Protocol) chuẩn mà dịch vụ LLM bắt buộc tuân theo
│   │   │   └── response_preferences.py        #   Lưu thông tin tùy chọn phong cách: ngôn ngữ, độ ngắn gọn, cách xưng hô
│   │   └── schema/                            # Ràng buộc dữ liệu vào/ra bằng Pydantic (tự động báo lỗi nếu sai kiểu):
│   │       └── inference_schema.py            #   Khai báo schema: ChatRequest, ChatResponse, InfoResponse, LivenessResponse
│   └── core/                                  # Tiện ích nền tảng dùng chung
│       └── streaming.py                       # Đóng gói dữ liệu văn bản thành chuẩn Server-Sent Events (SSE) trả về client
│
├── config/                                    # Cấu hình dự án (được khóa và giám sát toàn vẹn bằng mã hash SHA-256)
│   ├── settings.py                            # Đọc .env cho model/kết nối/ngân sách/timeout; mặc định Qwen3.5 4B
│   ├── prompts.py                             # Chứa system prompt nhập vai chuyên gia bản địa, cấm nhắc từ kỹ thuật
│   ├── generation.py                          # Từ khóa nhận biết câu hỏi dài và câu thông báo khi câu trả lời chạm trần token
│   ├── conversation.py                        # Ngưỡng bộ nhớ và giới hạn lượt trò chuyện (bảo toàn theo baseline hash)
│   └── validation.py                          # Cấu hình danh sách test inference được kích hoạt cho các bộ chạy test
│
├── ollama/                                    # Cấu hình và script chạy Ollama cục bộ trong container Docker
│   ├── Dockerfile                             # Khởi tạo image Ollama phục vụ mô hình inference
│   └── start-ollama.sh                        # Script tự động tải model chat và chạy Ollama daemon khi container khởi động
│
├── training/                                  # Huấn luyện mô hình offline (chạy bằng GPU riêng, tách biệt hoàn toàn với runtime)
│   ├── pipeline.py                            # Lớp điều phối toàn bộ chu trình huấn luyện offline
│   ├── configs/
│   │   └── qlora.json                         # Tham số huấn luyện: rank, alpha, learning rate, batch size, đường dẫn lưu
│   ├── data/                                  # Dữ liệu huấn luyện (phân loại nguồn để chống rò rỉ dữ liệu train/test):
│   │   ├── raw/                               #   Dữ liệu thô ban đầu (gitignore, giữ bằng .gitkeep)
│   │   ├── prepared/                          #   Dữ liệu sau khi làm sạch, chia train/val/test (gitignore, giữ bằng .gitkeep)
│   │   ├── examples/synthetic.jsonl           #   File ví dụ giả lập để kiểm tra thử pipeline chạy thông suốt
│   │   ├── schema.json                        #   JSON Schema chuẩn hóa cấu trúc mẫu dữ liệu huấn luyện
│   │   └── README.md                          #   Quy định chuẩn bị và kiểm soát chất lượng dữ liệu huấn luyện
│   ├── evaluation/                            # Dữ liệu và cấu hình đánh giá mô hình sau huấn luyện:
│   │   └── reference/                         #   Dữ liệu tham chiếu chuẩn
│   │       ├── golden_queries.json            #   Bộ câu hỏi vàng để kiểm thử chất lượng sinh của adapter
│   │       ├── legacy_evals/                  #   Bộ dữ liệu đánh giá kế thừa (context, routing, markdown, rag leakage)
│   │       └── README.md                      #   Hướng dẫn quy trình thẩm định chất lượng sau huấn luyện
│   ├── scripts/
│   │   ├── prepare_dataset.py                 #   Kiểm tra định dạng, lọc câu trả lời rỗng và chia tập train/val/test
│   │   ├── train_lora.py                      #   Chạy huấn luyện LoRA/QLoRA hoặc chạy thử giả lập (--dry-run)
│   │   └── merge_adapter.py                   #   Gộp trọng số adapter đã huấn luyện vào base model để xuất ra file GGUF
│   ├── tests/
│   │   └── test_pipeline.py                   #   Kiểm thử đơn vị cho quy trình pipeline huấn luyện
│   ├── export/Modelfile.example               # Mẫu file cấu hình nạp trọng số GGUF đã huấn luyện vào lại Ollama
│   ├── artifacts/                             # Nơi lưu trữ checkpoint và adapter đầu ra (giữ bằng .gitkeep)
│   ├── models/                                # Nơi lưu trữ base model tải về (giữ bằng .gitkeep)
│   ├── requirements.txt                       # Thư viện riêng cho việc train: PyTorch, TRL, PEFT, bitsandbytes
│   └── README.md                              # Hướng dẫn chi tiết cách chuẩn bị dữ liệu và lệnh chạy huấn luyện
│
├── proactive_recommendation/                  # Nhánh nghiên cứu độc lập C01-F1.0: thuật toán gợi ý chủ động (không dùng LLM)
│   ├── config.py                              # Thiết lập các tham số, trọng số đặc trưng (features) và phân phối xác suất
│   ├── controller.py                          # Máy trạng thái (state machine) điều hướng chu trình gợi ý
│   ├── engine.py                              # Lõi thuật toán tính toán và quyết định nội dung gợi ý
│   ├── model.py                               # Mô hình tính xác suất nhu cầu của khách hàng
│   ├── adapter.py                             # Bộ chuyển đổi dữ liệu đầu vào cho engine gợi ý
│   ├── snapshot.py                            # Lưu ảnh chụp trạng thái phục vụ kiểm toán quyết định gợi ý
│   ├── transactions.py                        # Mô hình hóa dữ liệu giao dịch hành vi
│   ├── types.py                               # Định nghĩa kiểu dữ liệu và cấu trúc bản ghi nội bộ
│   ├── experiments.py                         # Thực thi các kịch bản thực nghiệm đo lường hiệu quả thuật toán
│   ├── manifest.py                            # Kiểm tra và đối soát tính toàn vẹn của manifest dữ liệu
│   ├── demo_data/                             # Dữ liệu mẫu giả lập phục vụ chạy thử thuật toán:
│   │   ├── expected_proactive_cases.json      #   Các ca kiểm thử kết quả gợi ý kỳ vọng
│   │   ├── manifest.json                      #   Bản kê khai kiểm soát nguồn gốc dữ liệu demo
│   │   ├── vietsage_demo_source.json          #   Dữ liệu nguồn danh mục dịch vụ phục vụ chạy thử
│   │   └── README.md                          #   Mô tả ngữ nghĩa tập dữ liệu demo
│   ├── docs/                                  # Tài liệu báo cáo và đặc tả nghiên cứu độc lập:
│   │   ├── 00_AI_RESEARCH_INDEX.md            #   Mục lục tài liệu nghiên cứu thuật toán
│   │   ├── 01_ALGORITHM_SPEC.md               #   Đặc tả chi tiết thuật toán gợi ý chủ động
│   │   ├── 02_DATA_AND_INTEGRATION_CONTRACT.md#   Hợp đồng tích hợp và cấu trúc dữ liệu
│   │   ├── 03_EXPERIMENT_PROTOCOL.md          #   Giao thức thực nghiệm và tiêu chí đo lường
│   │   └── 04_IMPLEMENTATION_HANDOFF.md       #   Hướng dẫn bàn giao mã nguồn nghiên cứu
│   ├── scripts/                               # Công cụ dòng lệnh:
│   │   ├── proactive_research.py              #   Chạy kiểm thử và thực nghiệm thuật toán
│   │   ├── generate_demo_source_pdf.py        #   Tạo tài liệu PDF nguồn từ dữ liệu demo
│   │   └── validate_demo_data.py              #   Kiểm tra tính hợp lệ của dữ liệu demo
│   ├── tests/                                 # Bộ kiểm thử cho nhánh nghiên cứu:
│   │   ├── test_proactive_engine.py           #   Kiểm thử tính đúng đắn của lõi gợi ý
│   │   └── test_proactive_experiments.py      #   Kiểm thử các kịch bản thực nghiệm
│   └── README.md                              # Báo cáo phương pháp nghiên cứu và hướng dẫn thực thi
│
├── scripts/                                   # Bộ công cụ kiểm tra tự động và đo kiểm hiệu năng hệ thống
│   ├── verify_architecture.py                 # Dùng AST soi từng dòng code, cấm rẽ nhánh ngoài services, kiểm tra hash config
│   ├── run_inference_tests.py                 # Bộ chạy toàn bộ bài kiểm tra tự động của hệ thống suy luận
│   ├── run_evals.py                           # Chạy bài test đánh giá chất lượng câu trả lời và chuẩn stream SSE
│   ├── verify_requirements.py                 # Kiểm tra đối soát các thư viện đang cài trong máy xem có khớp 100% không
│   ├── measure_model_latency.py               # Bấm giờ thời gian phản hồi (TTFT) và đo lượng VRAM GPU bị chiếm dụng
│   ├── benchmark_local.py                     # Chạy benchmark hiệu năng với các bộ câu hỏi thực tế (ngắn, so sánh, lịch trình)
│   └── validate_catalogue.py                  # Kiểm tra tính toàn vẹn và bản quyền của danh mục địa điểm/dịch vụ
│
├── tests/                                     # Bộ kiểm thử tự động của runtime suy luận (chạy bằng unittest)
│   ├── test_inference_api.py                  # Kiểm tra các route HTTP và cơ chế trả mã lỗi 422, 503
│   ├── test_chat_security.py                  # Kiểm tra bảo mật: cấm người dùng mạo danh role hệ thống, lọc thẻ <think>
│   ├── test_conversation_context.py          # Kiểm tra cô lập ngữ cảnh giữa các request song song, không nhớ chéo thông tin
│   ├── test_llm_streaming.py                  # Kiểm tra độ mượt của stream SSE, ngắt kết nối an toàn, lọc thẻ suy luận
│   ├── test_llm_budget.py                     # Kiểm tra bộ tính toán token và cơ chế tự động co giãn context window
│   ├── test_llm_warmup.py                     # Kiểm tra cơ chế nạp ấm model ngay khi khởi động
│   ├── test_output_limit.py                   # Kiểm tra cảnh báo khi câu trả lời dài chạm giới hạn
│   ├── test_response_style.py                 # Kiểm tra khả năng nhận diện ngôn ngữ và văn phong
│   └── test_eval_runner.py                    # Kiểm tra bộ khung chạy đánh giá evals
│
├── evals/                                     # Các bộ test-case mẫu dạng JSON dùng để chấm điểm câu trả lời ngoại tuyến
│   ├── normal-chat.json                       # Bộ test chat tự nhiên và chuyển đổi ngôn ngữ
│   ├── context-and-controls.json              # Bộ test kiểm tra việc tuân thủ thông tin tham chiếu
│   ├── loading-state.json                     # Bộ test trạng thái nạp dữ liệu ban đầu
│   ├── markdown-rendering.json                # Bộ test định dạng Markdown
│   ├── rag-format-leakage.json                # Bộ test kiểm tra xem câu trả lời có bị lộ từ khóa hệ thống không
│   └── response-routing.json                  # Bộ test kiểm tra định tuyến phản hồi
│
├── datasets/                                  # Dữ liệu tham chiếu chuẩn
│   ├── golden/queries.json                    # Bộ câu hỏi vàng chuẩn có sẵn điều kiện lọc để thẩm định danh mục
│   ├── manifests/data_sources.json            # Quản lý nguồn gốc và giấy phép của dữ liệu
│   └── README.md                              # Hướng dẫn quản lý dữ liệu tham chiếu
│
├── docs/                                      # Tài liệu hợp đồng hệ thống
│   ├── INFERENCE_API.md                       # Bản hợp đồng giao tiếp API suy luận: format JSON, SSE stream và mã lỗi
│   └── ARCHITECTURE.md                        # Tệp tài liệu kiến trúc toàn diện của dự án
│
├── runtime/                                   # Trạng thái máy cục bộ (báo cáo kiểm thử, hash baseline, logs - không commit code)
│   ├── architecture-check.json                # Báo cáo kiểm tra cấu trúc AST từ verify_architecture.py
│   ├── config-baseline.json                   # Bảng mã băm SHA-256 bảo vệ tính toàn vẹn của thư mục config/
│   └── *.log / *.json                         # Logs và kết quả chạy kiểm thử thực tế
│
├── .env / .env.example                        # URL Ollama, timeout, context size; model chọn qua OLLAMA_CHAT_MODEL
├── .dockerignore                              # Loại trừ các file không cần thiết khi đóng gói Docker image
├── .gitignore / .gitattributes                # Quy tắc quản lý phiên bản Git và kết thúc dòng LF/CRLF
├── docker-compose.production.yml              # File compose chạy container Ollama bản production, lưu trữ model volume
├── docker-compose.gpu.yml                     # Cấu hình kích hoạt GPU NVIDIA cho docker compose
├── requirements.txt                           # Danh sách các thư viện Python cho runtime đã ghim phiên bản chính xác
├── AGENTS.md                                  # Văn bản quy định kiến trúc bắt buộc (Clean Architecture, cấm DB)
└── README.md                                  # Hướng dẫn tổng quan về dự án và các lệnh khởi chạy
```

---

## 3. Luồng xử lý và Chuỗi điều phối (Execution Flow)

Luồng xử lý một yêu cầu suy luận tuân thủ nghiêm ngặt mô hình phân tầng hướng tâm (Onion / Clean Architecture):

```text
Caller (HTTP Client)
       │
       ▼
[app/controllers/inference_controller.py]
       │  (Tiếp nhận HTTP request, ánh xạ Schema, không có if/else)
       ▼
[app/services/inference_service.py]
       │  (Nhạc trưởng: gọi validation, nạp prompt hệ thống, điều phối chat/stream)
       ├──► [app/services/input_validation_service.py] (Kiểm tra liên trường: message XOR messages, lượt cuối là user)
       │
       ▼
[app/services/llm_service.py]
       │  (Chính sách sinh: đo đạc context/output budget, cô lập usage theo request)
       ├──► [app/services/generation_budget_service.py] (Co context 2048 cho câu ngắn, mở rộng cho câu dài)
       │
       ▼
[app/domain/models/provider.py (ModelProvider Protocol)]
       │  (Giao diện trừu tượng, Dependency Inversion)
       ▼
[app/services/ollama_service.py]
       │  (Tái sử dụng HTTP connection pool, serialize payload Ollama, parse NDJSON stream)
       ▼
Ollama Server (Local LLM Daemon)
       │
       ▼
[app/services/response_sanitization_service.py]
       │  (Lọc bỏ block suy luận <think>...</think>, đảm bảo JSON hợp lệ khi có json_schema)
       ▼
[app/core/streaming.py] (nếu gọi /chat/stream)
       │  (Đóng gói SSE: event token/done/error)
       ▼
Caller (HTTP Response: JSON Object hoặc SSE Stream)
```

---

## 4. Các quy tắc kiến trúc bất biến (Architectural Invariants)

1. **Ranh giới logic điều kiện (Conditional Boundary)**:
   - Trong toàn bộ mã nguồn `app/`, các biểu thức điều kiện (`if/else`, toán tử 3 ngôi, `match`, list comprehension có `if`) **chỉ được phép xuất hiện bên trong `app/services/`**.
   - `controllers/`, `domain/`, và `core/` tuyệt đối không chứa nhánh rẽ logic. Điều này được kiểm soát tự động thông qua công cụ phân tích AST [scripts/verify_architecture.py](file:///c:/Users/Ai/Documents/LLMVietSage/scripts/verify_architecture.py).
2. **Quy tắc đặt tên file**:
   - Mọi module xử lý nghiệp vụ trong tầng service phải có hậu tố `_service.py` (ví dụ `inference_service.py`).
   - Tầng controller có hậu tố `_controller.py`, schema có hậu tố `_schema.py`.
3. **Độc lập tầng Domain (Domain Isolation)**:
   - `app/domain/` không được phép import bất kỳ thành phần nào từ `services/`, `controllers/`, `core/`, `config/`, cũng như không phụ thuộc vào `fastapi` hay `httpx`.
   - Các bản ghi dữ liệu thuần (immutable dataclass) đặt tại `app/domain/models/`.
   - Các schema xác thực đầu vào/đầu ra đặt tại `app/domain/schema/`.
4. **Hợp đồng xử lý lỗi phân tầng (Error Handling)**:
   - Service bắt lỗi hạ tầng và ném các lỗi nghiệp vụ có kiểu rõ ràng trong `app/domain/errors.py` (`InvalidRequestError`, `ModelUnavailableError`, `UnsupportedGenerationModeError`).
   - `inference_errors_controller.py` đóng vai trò là HTTP Exception Handler được đăng ký tại `main.py`, chuyển đổi lỗi domain thành mã HTTP chuẩn (`422 Unprocessable Entity` khi dữ liệu sai, `503 Service Unavailable` khi Ollama gặp sự cố).
   - Khi stream SSE đã bắt đầu truyền dữ liệu, nếu xảy ra lỗi giữa chừng, service sẽ phát sự kiện `event: error` và lập tức kết thúc stream (tuyệt đối không phát `event: done` sau `error`).
5. **Không lưu trữ trạng thái (Zero State Persistence)**:
   - Runtime không duy trì session, không lưu lại lịch sử hội thoại giữa các request.
   - Không chứa ORM, SQLite, PostgreSQL, Redis hay Vector Database trong runtime suy luận.
   - Toàn bộ ngữ cảnh (context) và lịch sử phải do caller đóng gói và chịu trách nhiệm cung cấp.

---

## 5. Ranh giới giữa các phân hệ độc lập

- **Engine Suy luận (`app/`)**: Phục vụ API suy luận thời gian thực, triển khai qua Docker hoặc chạy trực tiếp bằng Uvicorn.
- **Pipeline Huấn luyện Offline (`training/`)**: Tách biệt hoàn toàn về môi trường và thư viện (`training/requirements.txt`), chỉ thực thi khi tinh chỉnh mô hình (fine-tuning QLoRA) trên máy chủ GPU riêng.
- **Nghiên cứu Gợi ý Chủ động (`proactive_recommendation/`)**: Đề tài nghiên cứu độc lập (mã C01-F1.0) phát triển thuật toán gợi ý không dùng LLM, được giữ nguyên vẹn tài liệu và bộ thực nghiệm.
- **Bảo toàn Cấu hình (`config/`)**: Toàn bộ nội dung thư mục cấu hình được khóa và xác minh tính toàn vẹn bằng SHA-256 (`runtime/config-baseline.json`), ngăn ngừa các thay đổi ngoài ý muốn trong quá trình tái cấu trúc mã nguồn.
