"""Inference response limits and long-answer classification rules."""
import re

OUTPUT_LIMIT_NOTICE = (
    "\n\n---\nPhản hồi đã chạm giới hạn độ dài và chưa hoàn tất. "
    "Bạn có thể yêu cầu tiếp tục phần còn lại."
)

_LONG_RESPONSE_MARKERS = (
    "lịch trình", "lich trinh", "kế hoạch", "ke hoach", "từng ngày",
    "tung ngay", "chi tiết", "chi tiet", "đầy đủ", "day du", "so sánh",
    "so sanh", "itinerary", "day-by-day", "detailed", "compare", "行程",
    "详细", "逐日", "比较",
)

_MULTI_DAY_RESPONSE_PATTERN = re.compile(
    r"(?:\b(?:[2-9]|[1-9]\d)\s*(?:ngày|ngay|days?)\b|\b\d+\s*天)",
    re.IGNORECASE,
)
