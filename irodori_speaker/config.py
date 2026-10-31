from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CHAPTER_REGEX = (
    r"^\s*(第[0-9０-９一二三四五六七八九十百]+[章話節]|プロローグ|エピローグ).*$"
)

EMOTIONS = [
    "neutral",
    "happy",
    "sad",
    "angry",
    "surprised",
    "fearful",
    "tender",
    "whisper",
    "excited",
    "serious",
]

# 感情 × 強度(1/2/3) → 挿入する絵文字。空文字は挿入なし
DEFAULT_EMOTION_EMOJI: dict[str, list[str]] = {
    "neutral": ["", "", ""],
    "happy": ["", "😊", "😆"],
    "sad": ["", "😟", "😭"],
    "angry": ["", "😠", "😠😠"],
    "surprised": ["", "😲", "😲😲"],
    "fearful": ["", "😰", "😰😰"],
    "tender": ["", "🫶", "🫶🫶"],
    "whisper": ["", "👂", "👂👂"],
    "excited": ["", "💥", "💥💥"],
    "serious": ["", "", ""],
}


@dataclass
class SamplingConfig:
    num_steps: int | None = None
    t_schedule_mode: str = "linear"
    sway_coeff: float = -1.0
    cfg_scale_text: float = 3.0
    cfg_scale_speaker: float = 5.0
    duration_scale: float = 0.8
    max_seconds: float = 30.0
    seed: int | None = 42


@dataclass
class LLMConfig:
    api: str = "ollama"
    base_url: str = "http://localhost:11434"
    model: str = "qwen3.5:4b-q8_0"
    think: bool = False
    api_key: str | None = None
    timeout: float = 120.0
    concurrency: int = 1
    unload: bool = True
    server_command: list[str] | None = None
    server_start_timeout: float = 300.0


@dataclass
class SplitConfig:
    chapter_regex: str = DEFAULT_CHAPTER_REGEX
    max_chars: int = 40


@dataclass
class SilenceConfig:
    sentence: float = 0.25
    paragraph: float = 0.6
    chapter: float = 1.0
    title: float = 1.0  # 章題の後
    scene: float = 2.0  # 区切り行の位置
    turn: float = 0.4  # 同じ段落で台詞と地の文が切り替わる所
    ellipsis: float = 0.6  # 前の文が …‥―— で終わる所


@dataclass
class BgmConfig:
    secrets_file: str = "~/.config/irodori-speaker/secrets.env"
    clap_model: str = "laion/larger_clap_general"
    device: str = "cuda"
    candidates: int = 12
    ambient_duration: str = "[30 TO 900]"
    music_duration: str = "[30 TO 300]"
    scene_chunk_chars: int = 3000
    min_scene_chars: int = 400
    download_workers: int = 6
    ambient_db: float = -20.0
    music_db: float = -3.0
    intro_seconds: float = 20.0
    bridge_seconds: float = 8.0
    music_fade_in: float = 2.0
    music_fade_out: float = 3.0
    ambient_crossfade: float = 4.0
    mp3_bitrate: str = "192k"


@dataclass
class Config:
    checkpoint: str = "Aratako/Irodori-TTS-v4.1-Small"
    codec_repo: str = "Aratako/Semantic-DACVAE-Japanese-32dim"
    device: str = "cuda"
    precision: str = "bf16"
    min_free_vram_gb: float = 3.0
    empty_cache_interval: int = 10
    voices_dir: str = "voices"
    voice: str | None = None
    ref_normalize_db: float | None = -16.0
    max_ref_seconds: float | None = None
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    # emoji | none
    emotion_mode: str = "emoji"
    narration_intensity_shift: int = 1
    emotion_emoji: dict[str, list[str]] = field(
        default_factory=lambda: deepcopy(DEFAULT_EMOTION_EMOJI)
    )
    llm: LLMConfig = field(default_factory=LLMConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    silence: SilenceConfig = field(default_factory=SilenceConfig)
    read_chapter_title: bool = True
    bgm: BgmConfig = field(default_factory=BgmConfig)
    output_dir: str = "out"


_SECTIONS = {
    "sampling": SamplingConfig,
    "llm": LLMConfig,
    "split": SplitConfig,
    "silence": SilenceConfig,
    "bgm": BgmConfig,
}


def _check_emotion_emoji(table: Any) -> None:
    if not isinstance(table, dict):
        raise ValueError("Config: emotion_emoji は感情名をキーとする辞書にしてください")
    for name, items in table.items():
        if not isinstance(items, list) or len(items) != 3:
            raise ValueError(
                f"Config: emotion_emoji[{name!r}] は強度 1〜3 に対応する 3 要素のリストにしてください"
            )


def _build(cls, data: dict[str, Any]):
    unknown = set(data) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"{cls.__name__}: 不明な設定キー {sorted(unknown)}")
    kwargs = dict(data)
    if cls is SplitConfig and "max_chars" in kwargs:
        n = kwargs["max_chars"]
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError("Config: split.max_chars は 1 以上の整数にしてください")
    if cls is BgmConfig:
        fades = kwargs.get("music_fade_in", 2.0) + kwargs.get("music_fade_out", 3.0)
        for name, default in (("bridge_seconds", 8.0), ("intro_seconds", 20.0)):
            if kwargs.get(name, default) < fades:
                raise ValueError(
                    f"Config: bgm.{name} は music_fade_in と music_fade_out の合計（{fades}）以上にしてください"
                )
    if cls is Config:
        for key, sub in _SECTIONS.items():
            if key in kwargs:
                kwargs[key] = _build(sub, kwargs[key] or {})
        if "emotion_emoji" in kwargs:
            _check_emotion_emoji(kwargs["emotion_emoji"])
    return cls(**kwargs)


def load_config(path: str | Path | None) -> Config:
    if path is None:
        return Config()
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return _build(Config, data)
