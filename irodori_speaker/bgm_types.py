from __future__ import annotations

import re
from dataclasses import asdict, dataclass


@dataclass
class Scene:
    chapter: int  # 0 始まり
    index: int  # 章内の場面番号（0 始まり）
    start_para: int
    end_para: int  # この場面の最後の段落（含む）
    description: str  # 英語の場面説明
    ambient_query: str | None  # None は意図した無音
    mood: str
    music_query: str  # 空文字は音楽を探さない
    status: str = "ok"  # "ok" | "llm_error"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Scene:
        return cls(**d)


@dataclass
class Sound:
    id: int
    name: str
    tags: list[str]
    description: str
    duration: float
    license: str  # Freesound の license URL
    username: str
    url: str  # Freesound のページ
    preview: str  # preview-hq-mp3 の URL

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Sound:
        return cls(**d)


@dataclass
class Choice:
    chapter: int
    scene: int
    kind: str  # "ambient" | "intro" | "bridge"
    query: str
    sound: Sound | None
    score: float | None
    reason: str | None  # None は選べた。"no_result" | "all_rejected" | "llm_error" | "error"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Choice:
        d = dict(d)
        if d.get("sound") is not None:
            d["sound"] = Sound.from_dict(d["sound"])
        return cls(**d)


_ZERO_RE = re.compile(r"/publicdomain/zero/([^/]+)")
_CC_RE = re.compile(r"/licenses/([^/]+)/([^/]+)")


def license_label(url: str) -> str:
    """Freesound の license URL を "CC BY 4.0" のような表記にする。"""
    m = _ZERO_RE.search(url)
    if m:
        return f"CC0 {m.group(1)}"
    m = _CC_RE.search(url)
    if m:
        return f"CC {m.group(1).upper()} {m.group(2)}"
    return url
