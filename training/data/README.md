# Dữ liệu offline cho training

- `raw/`: corpus thật do bạn cung cấp, được Git-ignore.
- `examples/synthetic.jsonl`: sáu mẫu hư cấu, chỉ để kiểm tra công cụ.
- `prepared/`: dữ liệu chuẩn hóa và source-separated splits, được Git-ignore.

Dạng hàng JSONL: `id`, `source_id`, `question`, `context`, `answer`; hoặc `id`, `source_id`, `messages`. Không có assistant answer thì chưa phải dữ liệu SFT. Không tự chuyển điều kiện lookup trong golden thành đáp án bằng cách đoán thông tin.

Corpus cần phản ánh cách caller đóng gói câu hỏi/context trong lúc inference. Giữ cùng nguồn tài liệu/nhóm hội thoại trong một split. Đánh giá held-out cần bám context, xử lý thiếu căn cứ và chất lượng câu trả lời, không chỉ dựa vào training loss.
