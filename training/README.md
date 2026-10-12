# Training LLM offline

Thư mục này dùng dữ liệu đã chuẩn bị để fine-tune model bằng SFT và LoRA/QLoRA. Nó độc lập với API inference: dữ liệu, checkpoint và kết quả training không được API tự đọc hoặc lưu.

```text
training/
  configs/qlora.json              Hyperparameter và đường dẫn training
  data/raw/                      Dữ liệu thật do bạn cung cấp
  data/examples/                 Ví dụ hư cấu để kiểm tra pipeline
  data/prepared/                 Train, validation, test đã chuẩn hóa
  evaluation/reference/          Dữ liệu đánh giá cũ được giữ làm tham khảo
  models/                        Checkpoint Hugging Face cục bộ
  artifacts/                     Adapter, checkpoint và metrics
  scripts/prepare_dataset.py      Kiểm tra và chia dữ liệu
  scripts/train_lora.py           Chạy fine-tuning hoặc dry-run
  scripts/merge_adapter.py        Merge adapter để xuất model
  export/Modelfile.example        Mẫu nạp GGUF vào Ollama
  tests/                         Test dữ liệu, không tải model
```

## Inference và training có trách nhiệm khác nhau

Inference nhận câu hỏi/context từ caller rồi trả lời. Training đọc một corpus offline có câu trả lời mẫu để cập nhật tham số/adapter. Không cần đưa vector store, ingestion hay database nghiệp vụ trở lại API để training.

## Dữ liệu cần cung cấp

Mỗi dòng JSONL cần `id`, `source_id` và câu trả lời đã được kiểm tra. Có hai dạng:

```json
{"id":"sample-001","source_id":"document-001","question":"Câu hỏi","context":"Thông tin tham chiếu","answer":"Câu trả lời mẫu đúng"}
```

Hoặc `messages` theo thứ tự system/user/assistant, kết thúc bằng câu trả lời assistant. Pipeline chỉ hỗ trợ text. `source_id` đại diện cùng nguồn tài liệu hoặc nhóm hội thoại liên quan; các câu hỏi chung nguồn luôn ở cùng một tập để tránh rò rỉ train/test.

`question/context` được ghép bằng prompt hiện tại của project. Nếu caller dùng một cách đóng gói khác, cung cấp `messages` đúng định dạng caller thực sự gửi để training và inference khớp nhau.

Chỉ đưa vào corpus dữ liệu có quyền sử dụng và đã xử lý thông tin cá nhân phù hợp. `data/raw`, `data/prepared`, `models` và `artifacts` được Git-ignore. File ví dụ có `synthetic=true`; không được tự coi là dữ liệu thật.

## Chuẩn bị corpus

Chạy từ project root:

```powershell
python -B training/scripts/prepare_dataset.py --input training/data/raw/corpus.jsonl
```

Công cụ kiểm tra role, label trống, input trùng/mâu thuẫn, reasoning tag và chia theo nguồn. Nó tạo `train.jsonl`, `validation.jsonl`, `test.jsonl` và manifest/hash. Cần ít nhất ba nhóm nguồn độc lập.

Để thử công cụ bằng dữ liệu hư cấu, dùng thư mục output tạm:

```powershell
python -B training/scripts/prepare_dataset.py --input training/data/examples/synthetic.jsonl --output-dir runtime/training-smoke --allow-synthetic
```

## Môi trường và checkpoint

Tạo môi trường Python riêng cho training; không thêm PyTorch/TRL vào dependency của API. Cài PyTorch phù hợp CUDA bằng [hướng dẫn chính thức](https://pytorch.org/get-started/locally/), rồi cài `training/requirements.txt` trong môi trường đó. Đây là các khoảng phiên bản; script ghi phiên bản thực tế trong `training_run.json` để có thể khóa môi trường sau khi đã thử nghiệm.

Đặt checkpoint Hugging Face Qwen3.5 4B tương thích, tokenizer và chat template tại `training/models/base/`, rồi đặt `base_model_dir` trong `configs/qlora.json`. Giá trị ban đầu là `null` để tránh tự chọn hoặc tải nhầm model. Tag Ollama `qwen3.5:4b` không phải đường dẫn checkpoint Hugging Face.

GPU đã đọc trên máy hiện tại: RTX 4050 Laptop, 6141 MiB VRAM. Cấu hình mẫu dùng batch 1, gradient accumulation, LoRA và NF4 để giảm bộ nhớ, nhưng **chưa xác nhận checkpoint Qwen3.5 4B train được bằng trainer này trên GPU này**. Cần chọn đúng checkpoint, kiểm tra hỗ trợ kiến trúc và đo VRAM thực tế; có thể cần GPU/RAM lớn hơn. Script không tải model, không bật `trust_remote_code`, không gửi dataset lên Hub.

## Fine-tuning

Sau khi đã có corpus thật và checkpoint:

```powershell
python -B training/scripts/train_lora.py --dry-run
python -B training/scripts/train_lora.py
```

Dry-run chỉ kiểm tra dữ liệu/config, không nạp GPU hoặc cài package. Lệnh thứ hai mới bắt đầu training. Nó dùng conversational prompt/completion và tính loss trên câu trả lời cuối; validation/test riêng. Input quá dài bị báo lỗi thay vì âm thầm cắt mất label. Trainer/API của kiến trúc model, đặc biệt model vision-language, vẫn phải được kiểm chứng với checkpoint đã chọn.

File mẫu `allow_synthetic=false`. Bật nó chỉ để kiểm tra trên fixture; ví dụ đi kèm không đủ để tạo một model dùng thật. Trainer mẫu chạy checkpoint text-only; nếu checkpoint có `vision_config` thì dừng và yêu cầu processor/collator đúng kiến trúc. Điều này cần kiểm tra khi chọn checkpoint tương ứng với tag Qwen3.5 đang serve, không được coi mọi checkpoint đều tương đương.

## Đưa kết quả về inference

Kết quả LoRA nằm trong `artifacts/lora/`. Đây là adapter, chưa mặc định là model Ollama có thể chạy. Một đường xuất phổ biến:

```powershell
python -B training/scripts/merge_adapter.py --base-model training/models/base --adapter training/artifacts/lora
```

Merge cần đủ RAM cho base model. Sau đó dùng phiên bản converter GGUF hỗ trợ đúng kiến trúc checkpoint, quantize nếu cần và tạo Modelfile từ `export/Modelfile.example`. Đây là template xuất offline, không phải model thứ hai được runtime nạp. Không tự động chuyển adapter sang Ollama vì khả năng import phụ thuộc kiến trúc/phiên bản. Runtime hiện cấu hình Qwen3.5 4B; sau khi đánh giá và xuất model đã train tương thích, đổi `OLLAMA_CHAT_MODEL` trong env và khởi động lại API.

## Đánh giá và kiểm tra

Loss không đủ để kết luận model trả lời tốt. So sánh model gốc với model đã train trên các câu hỏi held-out: bám context, không bịa dữ kiện/giá, phản hồi khi thiếu căn cứ, ngôn ngữ, định dạng và độ trễ.

```powershell
python -B -m unittest discover -s training/tests -v
```

`evaluation/reference/golden_queries.json` được sao chép từ bộ câu hỏi catalogue cũ để giữ dữ liệu. Nó có predicate/điều kiện lookup, **không có assistant label**, nên không được nạp trực tiếp làm SFT. Cần corpus hỏi–đáp được gán nhãn thật.

Nguồn kỹ thuật: [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer), [PEFT quantization](https://huggingface.co/docs/peft/developer_guides/quantization), [bitsandbytes](https://huggingface.co/docs/transformers/quantization/bitsandbytes), [Ollama import](https://docs.ollama.com/import).
