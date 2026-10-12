# Quy ước bắt buộc của LLMVietSage

## Nhiệm vụ của engine sinh phản hồi

App thực hiện nhiệm vụ sinh phản hồi từ LLM:
- **Nhận vào**: Câu hỏi, thông tin tham chiếu (context) và lịch sử hội thoại do caller cung cấp (qua trường `message` hoặc `messages`).
- **Gửi đi**: Gửi yêu cầu inference tới model LLM phục vụ trên Ollama.
- **Trả về**: Phản hồi được sinh từ LLM (dưới dạng văn bản hoặc JSON object ở endpoint chat, hoặc SSE stream tăng dần ở endpoint chat/stream).

App không sở hữu hay quản lý kho tri thức, việc chuẩn hóa vị trí, tra cứu, retrieval, fallback workflow hoặc định dạng phản hồi phía ngoài. Không đưa các chức năng đó vào app.

Runtime không lưu câu hỏi, context, hội thoại hoặc kho tri thức. Không có ORM, migration, vector store, ingestion, retrieval, CRUD, session, tài khoản/đối tác, Generative UI hoặc embedding endpoint. HTTP pool và model weights là tài nguyên inference; dữ liệu tạm của request được giải phóng theo vòng đời request/stream.

Training là luồng offline riêng ở training/. Proactive recommendation là nghiên cứu độc lập đã yêu cầu giữ. Không thay đổi hai phạm vi này chỉ để tổ chức app; không xóa dữ liệu training/golden vì inference không lưu data.

## Vị trí và trách nhiệm trong app

| Vị trí | Quy định |
|---|---|
| app/main.py | Composition root: tạo app, đăng ký route, handler, lifecycle và service. |
| app/controllers/ | Khai báo API/dependency và định dạng HTTP. Chỉ gọi service, nhận/trả dữ liệu. |
| app/services/ | Toàn bộ quyết định có điều kiện, validation liên trường, chính sách sinh, lifecycle, lỗi và giao thức Ollama. |
| app/domain/schema/ | Pydantic khai báo kiểu, field, role, min length, extra fields và response. Không có nhánh xử lý. |
| app/domain/models/ | Record dữ liệu thuần, immutable khi phù hợp, và protocol provider. Không có ORM/trọng số LLM. |
| app/domain/errors.py | Lỗi có kiểu, độc lập thư viện HTTP/provider. |
| app/core/ | Tiện ích dùng chung không phụ thuộc nghiệp vụ, ví dụ SSE encoder tổng quát. Không dựng riêng inference service ở core. |

Domain nằm trực tiếp trong app/, ngang hàng với các tầng còn lại. Không tạo thêm tầng chỉ để tăng số lượng folder; không tạo lại app/adapters hoặc lớp wrapper trùng vai trò.

### Tên file và quyết định xử lý

- Mọi file Python có logic trong services phải mang tên chức_năng_service.py. Chỉ package marker __init__.py là ngoại lệ.
- Controller mang tên chức_năng_controller.py; schema mang tên chức_năng_schema.py.
- Trong source app, if/else, biểu thức điều kiện, match và comprehension có điều kiện chỉ nằm trong services. Không tự nới quy tắc này trong tài liệu.
- Controller không có nhánh, vòng lặp hoặc try/except xử lý tình huống. HTTP error handler định dạng lỗi đã được phân loại; framework chọn handler theo kiểu lỗi.
- Schema giữ ràng buộc khai báo. Các kiểm tra cần điều kiện như message XOR messages, lượt cuối user và whitespace-only nằm trong input_validation_service, chạy trước khi gọi provider.
- Không lách quy tắc bằng logic điều kiện giấu trong BoolOp, lookup phức tạp hoặc helper đặt ở controller/domain/core.
- Service trả dữ liệu hoặc lỗi có kiểu; không nhập FastAPI/Starlette, không trả HTTPException/JSONResponse/StreamingResponse.
- Domain không nhập services/controllers/core/config, FastAPI hoặc httpx. Core không nhập các tầng nghiệp vụ.
- LLMService phụ thuộc ModelProvider protocol; OllamaService hiện thực HTTP provider. Main nối các thành phần; không có import vòng.
- Dataclass dữ liệu nằm trong domain/models; Pydantic BaseModel nằm trong domain/schema. Không giữ khai báo record lẫn trong service.
- Response style helper hiện chỉ dùng test/offline eval, chưa nối vào prompt runtime. Không tuyên bố nó đang điều khiển model và không tự nối khi chỉ refactor.

### Hợp đồng và vòng đời

- Giữ nội dung câu hỏi/context/history nguyên vẹn. Không tự trích vị trí, dedupe, summarize hoặc cắt dữ liệu dưới tên chuẩn hóa.
- Chat nhận message hoặc messages, không nhận đồng thời cả hai. Role caller chỉ là user/assistant, lượt cuối user; system prompt do project quản lý.
- Service không nhớ hội thoại giữa các request. Usage và options phải cô lập; không dùng shared mutable state chứa caller data.
- Stream tăng dần, đóng generator/provider khi hoàn tất, lỗi, hủy hoặc client dừng đọc. Không phát done sau error.
- Lỗi sau khi stream bắt đầu đi qua SSE error. Không giả định đổi HTTP status sau khi đã gửi nội dung.
- Không log nguyên prompt/context, credential hoặc URL lỗi nội bộ. Không coi regex/system prompt là bảo đảm tuyệt đối chống prompt injection.
- Structured output hiện bảo đảm JSON object hợp lệ; chưa xác thực toàn bộ JSON Schema caller. Không tăng lời cam kết vượt quá thực thi/test.

## Cấu hình và dữ liệu

- Giữ nguyên source config/ trong công việc này, xác minh SHA256 trước/sau. Không chuyển settings/prompt vào core, không thay model tag để refactor.
- .env và env template ở project root. Không ghi đè, in bí mật hoặc commit env thật.
- Compose ở root; build/startup Ollama ở ollama/. Default triển khai chỉ tải model chat. Không tự chạy Docker, tải model hoặc training khi làm kiến trúc.
- Config còn một số field embedding/conversation cũ do yêu cầu giữ nguyên folder; runtime không dùng chúng. Chỉ sửa khi có yêu cầu cấu hình riêng.
- Dependency inference và training tách riêng. Dữ liệu/checkpoint/artifact training không nhập vào runtime.
- Evals/golden là dữ liệu đánh giá, không phải cấu hình và không tự trở thành SFT label.
- Runtime outputs/cache/model weights/env thật là trạng thái máy, không phải source để đổi tên theo quy ước app.

## Quy trình xác nhận bắt buộc cho mọi file

Không báo hoàn tất chỉ vì code chạy hoặc một bộ test pass. Sau mỗi lần tạo, xóa, chuyển, đổi tên hoặc refactor file, phải thực hiện đủ các yêu cầu sau:

1. Lập inventory mọi source và artifact thuộc Git hoặc chưa bị ignore; kiểm tra cả file mới/untracked. Phân loại runtime, config, test/tool, docs/deployment, training/data và nghiên cứu độc lập. Không đọc env thật hoặc corpus riêng tư để tạo inventory.
2. Duyệt từng file source: vị trí, tên, trách nhiệm, nhánh xử lý, class dữ liệu, dependency/import và phạm vi sinh phản hồi LLM. Kiểm tra mọi nơi gọi sau di chuyển; không để shim/source cũ hoặc cache orphan.
3. Chạy kiểm tra tự động, tạo báo cáo theo từng file:

       python -B scripts/verify_architecture.py --report runtime/architecture-check.json

   Khi config phải giữ nguyên, lưu baseline hash trước khi sửa rồi đối chiếu:

       python -B scripts/verify_architecture.py --config-baseline runtime/config-baseline.json --report runtime/architecture-check.json

   Baseline là JSON mapping đường dẫn config tương đối tới SHA256, loại cache. Không tạo baseline sau sửa rồi coi đó là bằng chứng trước/sau.
4. Kiểm tra thủ công phần công cụ không chứng minh được: module có đúng nhiệm vụ, tài liệu có phản ánh code, artifact/data có đúng phạm vi; đọc link/command của docs liên quan. Inventory “classified” không có nghĩa chất lượng dataset/model đã được chứng minh.
5. Chạy test/eval phù hợp. Với app: run_inference_tests và offline eval; khi đổi boundary/source inventory, chạy cả unittest discovery để không che test hỏng. Với training: test dữ liệu/dry-run; không training thật nếu chưa được yêu cầu.
6. Kiểm tra diff, link, source syntax, orphan bytecode, legacy, import vòng, route inventory và config baseline. CI phải chạy verify_architecture; vi phạm làm job thất bại.
7. Đọc lại báo cáo từng file và sửa các vi phạm trước khi kết thúc. Báo số source/runtime file đã kiểm tra, số vi phạm, kết quả test/eval, config có thay đổi không, và các giới hạn chưa kiểm chứng. Không dùng từ “hoàn hảo” hoặc “đã tối ưu nhanh nhất” thay cho bằng chứng.

Tool tự động kiểm tra cấu trúc source; review thủ công xác nhận ý nghĩa. Cả hai là bắt buộc. Không sửa training/proactive chỉ để làm cùng một quy ước naming của app.

## Làm việc với yêu cầu

Yêu cầu liệt kê/soát/nhận xét chỉ cho phép đọc. Yêu cầu refactor cho phép sửa phần đã giao và cập nhật nơi gọi/test/docs. Bảo toàn thay đổi của người dùng; không revert hoặc xóa dữ liệu ngoài phạm vi.

Nếu cần thay đổi hợp đồng vì nhiệm vụ đổi, cập nhật controller/service/schema/test/docs đồng bộ và nêu rõ. Runtime hiện chỉ có chat, stream, info và liveness; embedding đã bỏ theo phạm vi đã chốt.
