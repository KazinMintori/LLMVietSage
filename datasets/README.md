# Dữ liệu tham khảo offline

Thư mục này giữ golden queries và manifest nguồn đã có để phục vụ đánh giá/chuyển đổi dữ liệu, không được API generation đọc trong request.

- golden/queries.json chứa predicate/điều kiện tra cứu, chưa có câu trả lời assistant để nạp trực tiếp thành SFT.
- manifests/data_sources.json là metadata tham khảo, không khởi động ingestion/retrieval.
- Corpus có nhãn, prepared splits và model artifacts mới thuộc training/. Không xóa reference data chỉ vì app không lưu data.
- Bản copy tham khảo trong training giữ nguyên dữ liệu gốc; dữ liệu demo/reference không chứng minh model sẵn sàng.

App chỉ đóng vai trò engine inference sinh phản hồi; kho tri thức và retrieval thuộc các dịch vụ bên ngoài. Các mô tả catalogue/UI/ingestion cũ không còn là chức năng runtime của repository.

Xem [training](../training/README.md) và [kiến trúc hiện tại](../README.md).
