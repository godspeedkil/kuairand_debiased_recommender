"""Column contracts for the raw KuaiRand files and the processed tables.

Source of truth: https://github.com/chongminggao/KuaiRand (README, "Data Descriptions").
"""

from __future__ import annotations

LOG_COLUMNS = [
    "user_id", "video_id", "date", "hourmin", "time_ms",
    "is_click", "is_like", "is_follow", "is_comment", "is_forward", "is_hate",
    "long_view", "play_time_ms", "duration_ms", "profile_stay_time",
    "comment_stay_time", "is_profile_enter", "is_rand", "tab",
]

BINARY_LOG_COLUMNS = [
    "is_click", "is_like", "is_follow", "is_comment", "is_forward", "is_hate",
    "long_view", "is_profile_enter", "is_rand",
]

USER_COLUMNS = [
    "user_id", "user_active_degree", "is_lowactive_period", "is_live_streamer",
    "is_video_author", "follow_user_num", "follow_user_num_range", "fans_user_num",
    "fans_user_num_range", "friend_user_num", "friend_user_num_range",
    "register_days", "register_days_range",
    *[f"onehot_feat{i}" for i in range(18)],
]

ITEM_COLUMNS = [
    "video_id", "author_id", "video_type", "upload_dt", "upload_type",
    "visible_status", "video_duration", "server_width", "server_height",
    "music_id", "music_type", "tag",
]

# Processed interaction table, one row per logged impression.
INTERACTION_COLUMNS = [
    "user_id", "item_id", "date", "time_ms", "tab", "source",
    "is_click", "long_view", "is_like", "is_follow", "is_comment", "is_forward",
    "is_hate", "is_profile_enter", "play_time_ms", "duration_ms",
]

SOURCES = ("standard", "random")
SPLITS = ("train", "val", "test")
