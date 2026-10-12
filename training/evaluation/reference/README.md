# Dữ liệu tham khảo được giữ lại

`golden_queries.json` là bản sao nguyên byte của bộ câu hỏi catalogue cũ. Nó có predicate và số kết quả kỳ vọng, chưa có response label nên không được nạp trực tiếp làm SFT.

`legacy_evals/` giữ bản sao đầy đủ các suite trước khi tách generator khỏi backend/UI. Đây là tài liệu tham khảo để thiết kế bài đánh giá model mới, không phải bộ eval đang chạy và không phải corpus đã được gán nhãn.

Dữ liệu gốc trong `datasets/` chưa bị xóa. Việc có bộ tham khảo không chứng minh đã có đủ corpus thật để fine-tune.
