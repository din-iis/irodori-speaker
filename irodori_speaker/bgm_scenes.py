"""LLM で章を場面に区切る（bgm ステップ 1）。"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections import defaultdict
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from .bgm_types import Scene
from .config import LLMConfig

SCENE_PROMPT_VERSION = 1
SCENE_SYSTEM_PROMPT = (
    "You split a chapter of a Japanese novel into scenes for an audiobook soundtrack. "
    "A new scene starts only when the location or the time clearly changes (a major scene shift); "
    "otherwise the paragraphs continue the current scene. For each scene give: "
    "start_para (the paragraph number where it starts); "
    "description (one English sentence: place, time of day, weather, the sounds one would hear there, mood); "
    "ambient_query (2-4 English words to search a sound library for the background ambience, "
    "or null when the scene should be silent, e.g. a quiet room with nothing to hear); "
    "mood (1-3 English words); "
    "music_query (2-4 English words to search for a short instrumental piece matching the mood). "
    "If the first paragraph shown continues the previous scene, do not start a scene there. "
    "Use only the paragraph numbers shown."
)
SCENE_SCHEMA = {
    "type": "object",
    "properties": {"scenes": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "start_para": {"type": "integer"},
            "description": {"type": "string"},
            "ambient_query": {"type": ["string", "null"]},
            "mood": {"type": "string"},
            "music_query": {"type": "string"},
        },
        "required": ["start_para", "description", "ambient_query", "mood", "music_query"],
    }}},
    "required": ["scenes"],
}

_NO_PREVIOUS = "none (start of chapter)"

Paragraphs = list[tuple[int, str]]


def chapter_paragraphs(rows: list[dict]) -> dict[int, Paragraphs]:
    """script.tsv の行から、章 → [(para, 段落の原文)]。原文は source 列を seg 順に連結する。"""
    segs: dict[tuple[int, int], list[tuple[int, str]]] = defaultdict(list)
    for r in rows:
        segs[(int(r["chapter"]), int(r["para"]))].append((int(r["seg"]), r.get("source", "")))
    out: dict[int, Paragraphs] = defaultdict(list)
    for (chapter, para), items in sorted(segs.items()):
        out[chapter].append((para, "".join(text for _, text in sorted(items, key=lambda x: x[0]))))
    return dict(out)


def chunk_paragraphs(paras: Paragraphs, chunk_chars: int) -> list[Paragraphs]:
    """段落を途中で切らずに、chunk_chars 文字ごとのまとまりに分ける。"""
    chunks: list[Paragraphs] = []
    cur: Paragraphs = []
    cur_chars = 0
    for para, text in paras:
        if cur and cur_chars + len(text) > chunk_chars:
            chunks.append(cur)
            cur, cur_chars = [], 0
        cur.append((para, text))
        cur_chars += len(text)
    if cur:
        chunks.append(cur)
    return chunks


def _is_text(v: object) -> bool:
    return isinstance(v, str) and bool(v.strip())


def validate_scenes(reply: object, shown: list[int]) -> list[dict] | None:
    """応答が正しければ場面の dict の並びを、壊れていれば None を返す。"""
    if not isinstance(reply, dict) or not isinstance(reply.get("scenes"), list):
        return None
    out: list[dict] = []
    last = -1
    for item in reply["scenes"]:
        if not isinstance(item, dict):
            return None
        start = item.get("start_para")
        if isinstance(start, bool) or not isinstance(start, int) or start not in shown or start <= last:
            return None
        amb = item.get("ambient_query")
        if not (amb is None or _is_text(amb)):
            return None
        if not _is_text(item.get("description")) or not _is_text(item.get("mood")):
            return None
        if not isinstance(item.get("music_query"), str):
            return None
        last = start
        out.append({
            "start_para": start,
            "description": item["description"].strip(),
            "ambient_query": amb.strip() if amb is not None else None,
            "mood": item["mood"].strip(),
            "music_query": item["music_query"].strip(),
        })
    return out


def _scene_key(cfg: LLMConfig, chunk_text: str, prev_desc: str) -> str:
    raw = json.dumps(
        [cfg.model, chunk_text, prev_desc, cfg.think, SCENE_PROMPT_VERSION], ensure_ascii=False
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


class ScenePlanner:
    """段落ごとのまとまりを LLM に渡して場面を得る。成功した応答だけ jsonl にキャッシュする。"""

    def __init__(
        self,
        cfg: LLMConfig,
        cache_path: Path,
        chat: Callable[[str, str, dict], object] | None = None,
    ) -> None:
        self.cfg = cfg
        self.cache_path = cache_path
        self._chat = chat
        self.cache: dict[str, list[dict]] = {}
        self._lock = threading.Lock()
        if cache_path.exists():
            broken = 0
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row["scenes"], list):
                        raise TypeError
                    self.cache[row["hash"]] = row["scenes"]
                except (json.JSONDecodeError, KeyError, TypeError):
                    broken += 1
            if broken:
                print(f"[bgm] 場面のキャッシュの壊れた行を {broken} 件読み飛ばしました: {cache_path}")

    def _call(self, user: str) -> object:
        if self._chat is not None:
            return self._chat(SCENE_SYSTEM_PROMPT, user, SCENE_SCHEMA)
        from .emotion import chat_json

        return chat_json(self.cfg, SCENE_SYSTEM_PROMPT, user, SCENE_SCHEMA, "bgm_scenes")

    def _needs_newline(self) -> bool:
        try:
            with self.cache_path.open("rb") as f:
                f.seek(-1, 2)
                return f.read(1) != b"\n"
        except OSError:  # 無い・空のファイル
            return False

    def _store(self, key: str, scenes: list[dict]) -> None:
        with self._lock:
            self.cache[key] = scenes
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            row = json.dumps({"hash": key, "scenes": scenes}, ensure_ascii=False) + "\n"
            # 末尾が途中で切れていると次の行まで壊れるので、改行で終わらせてから追記する
            if self._needs_newline():
                row = "\n" + row
            with self.cache_path.open("a", encoding="utf-8") as f:
                f.write(row)

    def _ask(self, chunk: Paragraphs, prev_desc: str) -> list[dict] | None:
        """1 つのまとまりを問い合わせる。壊れていれば 1 回だけやり直し、だめなら None（キャッシュしない）。"""
        body = "\n".join(f"[{p}] {t}" for p, t in chunk)
        key = _scene_key(self.cfg, body, prev_desc)
        with self._lock:
            cached = self.cache.get(key)
        if cached is not None:
            return cached
        user = f"Previous scene: {prev_desc}\n\nParagraphs:\n{body}"
        shown = [p for p, _ in chunk]
        for _ in range(2):
            try:
                scenes = validate_scenes(self._call(user), shown)
            except Exception as e:  # 通信の失敗・応答の形の崩れ
                print(f"[bgm] 場面の問い合わせに失敗: {type(e).__name__}")
                scenes = None
            if scenes is not None:
                self._store(key, scenes)
                return scenes
        return None

    def plan_chapter(self, chapter: int, paras: Paragraphs, chunk_chars: int) -> list[Scene]:
        scenes: list[Scene] = []
        for ci, chunk in enumerate(chunk_paragraphs(paras, chunk_chars)):
            prev_desc = (scenes[-1].description if scenes else "") or _NO_PREVIOUS
            got = self._ask(chunk, prev_desc if scenes else _NO_PREVIOUS)
            first_para = chunk[0][0]
            if got is None or (ci == 0 and not got):
                if not scenes:  # 章頭の失敗: 環境音も音楽も無い場面
                    scenes.append(Scene(chapter, 0, first_para, first_para, "", None, "", "", status="llm_error"))
                continue  # それ以外は直前の場面が続く
            for i, item in enumerate(got):
                start = first_para if (ci == 0 and i == 0) else item["start_para"]
                scenes.append(Scene(
                    chapter, len(scenes), start, start, item["description"],
                    item["ambient_query"], item["mood"], item["music_query"],
                ))
        last_para = paras[-1][0] if paras else 0
        for i, s in enumerate(scenes):
            s.end_para = scenes[i + 1].start_para - 1 if i + 1 < len(scenes) else last_para
        return scenes


def merge_short(scenes: list[Scene], para_chars: dict[int, int], min_chars: int) -> list[Scene]:
    """min_chars 未満の場面を前の場面にまとめる。章頭の短い場面は次の場面にまとめる。"""

    def size(s: Scene) -> int:
        return sum(para_chars.get(p, 0) for p in range(s.start_para, s.end_para + 1))

    out: list[Scene] = []
    for s in scenes:
        if out and size(s) < min_chars:
            out[-1] = replace(out[-1], end_para=s.end_para)
        else:
            out.append(replace(s))
    if len(out) > 1 and size(out[0]) < min_chars:
        out[1] = replace(out[1], start_para=out[0].start_para)
        out.pop(0)
    return [replace(s, index=i) for i, s in enumerate(out)]


def load_scenes(path: Path) -> dict[int, list[Scene]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(ch): [Scene.from_dict(d) for d in items] for ch, items in data["chapters"].items()}


def save_scenes(path: Path, scenes: dict[int, list[Scene]], keep_chapters: set[int]) -> None:
    """今回の章を書き込み、他の章は keep_chapters に含まれるものだけ残す。一時ファイル経由で置き換える。"""
    merged = {ch: v for ch, v in load_scenes(path).items() if ch in keep_chapters}
    merged.update(scenes)
    data = {"chapters": {str(ch): [s.to_dict() for s in merged[ch]] for ch in sorted(merged)}}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
