from __future__ import annotations

import concurrent.futures
import hashlib
import json
import re
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import requests

from .config import DEFAULT_EMOTION_EMOJI, EMOTIONS, LLMConfig
from .text_splitter import Segment

CLAUSE_END = "、。！？!?…‥，,"
OPEN_BRACKETS = "「『（(【"
CLOSE_BRACKETS = "」』）)】"


def split_clauses(sentence: str) -> list[str]:
    """文を句に分ける。`"".join(結果) == sentence` を常に満たす。"""
    pieces: list[str] = []
    start = 0
    i = 0
    n = len(sentence)
    while i < n:
        if sentence[i] in CLAUSE_END:
            while i < n and sentence[i] in CLAUSE_END:
                i += 1
            while i < n and sentence[i] in CLOSE_BRACKETS:
                i += 1
            pieces.append(sentence[start:i])
            start = i
        else:
            i += 1
    if start < n:
        pieces.append(sentence[start:])

    # 括弧や空白だけの断片は前の句に併合する
    skip = OPEN_BRACKETS + CLOSE_BRACKETS
    merged: list[str] = []
    for p in pieces:
        if merged and all(ch.isspace() or ch in skip for ch in p):
            merged[-1] += p
        else:
            merged.append(p)
    return merged


@dataclass
class Annotation:
    emotion: str = "neutral"
    intensity: int = 1


SYSTEM_PROMPT = f"""あなたは小説の朗読演出家です。与えられた段落の各句について、朗読するときの感情と強さを決めます。
感情は次の語彙から必ず1つ選びます: {", ".join(EMOTIONS)}
強さ(intensity)は 1(わずか) / 2(普通) / 3(強い) の整数です。
台詞は話者の気持ちを、地の文は場面の雰囲気を反映させます。迷ったら neutral にします。
次の JSON だけを出力してください（説明文は不要）:
{{"segments":[{{"s":0,"c":0,"emotion":"neutral","intensity":1}}]}}
s は文番号、c は句番号で、全ての句について 1 件ずつ出力します。"""

JSON_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "s": {"type": "integer"},
                    "c": {"type": "integer"},
                    "emotion": {"type": "string", "enum": EMOTIONS},
                    "intensity": {"type": "integer", "minimum": 1, "maximum": 3},
                },
                "required": ["s", "c", "emotion", "intensity"],
            },
        }
    },
    "required": ["segments"],
}


def build_user_prompt(prev: str | None, target: list[list[str]], nxt: str | None) -> str:
    lines = []
    if prev:
        lines.append("【直前の段落（文脈）】\n" + prev)
    lines.append("【対象の句】")
    for s, clauses in enumerate(target):
        lines.extend(f"{s}-{c}: {text}" for c, text in enumerate(clauses))
    if nxt:
        lines.append("【直後の段落（文脈）】\n" + nxt)
    return "\n".join(lines)


def _is_int(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def validate(data: object, shape: list[int]) -> list[list[Annotation]]:
    """LLM の出力を検証し、shape（文ごとの句数）どおりの Annotation にする。不正な値は neutral。"""
    result = [[Annotation() for _ in range(n)] for n in shape]
    items = data.get("segments") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return result
    for item in items:
        if not isinstance(item, dict):
            continue
        s, c = item.get("s"), item.get("c")
        if not _is_int(s) or not 0 <= s < len(shape):
            continue
        if not _is_int(c) or not 0 <= c < shape[s]:
            continue
        emotion = item.get("emotion")
        if emotion not in EMOTIONS:
            continue
        intensity = item.get("intensity")
        if not _is_int(intensity):
            intensity = 2
        result[s][c] = Annotation(emotion, min(3, max(1, intensity)))
    return result


def paragraph_hash(
    model: str,
    clauses: list[list[str]],
    prev: str | None = None,
    nxt: str | None = None,
) -> str:
    """段落と前後の文脈（None も区別）からキャッシュキーを作る。"""
    payload = json.dumps([model, clauses, prev, nxt], ensure_ascii=False)
    return hashlib.sha1(payload.encode()).hexdigest()


def _auth_headers(cfg: LLMConfig) -> dict[str, str]:
    return {"Authorization": f"Bearer {cfg.api_key}"} if cfg.api_key else {}


def apply_shift(ann: Annotation, kind: str, narration_shift: int) -> Annotation:
    """地の文は強度を下げる。0 になったら neutral 扱い。"""
    if kind != "narration" or ann.emotion == "neutral":
        return ann
    intensity = ann.intensity - narration_shift
    if intensity < 1:
        return Annotation("neutral", 1)
    return Annotation(ann.emotion, intensity)


def to_emoji(ann: Annotation, table: dict[str, list[str]] | None = None) -> str:
    merged = {**DEFAULT_EMOTION_EMOJI, **(table or {})}
    levels = merged.get(ann.emotion) or [""]
    return levels[min(ann.intensity, len(levels)) - 1]


def build_spoken_text(
    clauses: list[str],
    anns: list[Annotation],
    table: dict[str, list[str]] | None = None,
) -> str:
    out = []
    prev = None
    for k, clause in enumerate(clauses):
        e = to_emoji(anns[k] if k < len(anns) else Annotation(), table)
        if (k == 0 or e != prev) and e:
            pos = 1 if clause and clause[0] in OPEN_BRACKETS else 0
            clause = clause[:pos] + e + clause[pos:]
        out.append(clause)
        prev = e
    return "".join(out)


def labels_to_str(anns: list[Annotation]) -> str:
    return "|".join(f"{a.emotion}:{a.intensity}" for a in anns)


def labels_from_str(s: str) -> list[Annotation]:
    if not s:
        return []
    result = []
    for part in s.split("|"):
        emotion, _, intensity = part.partition(":")
        result.append(Annotation(emotion, int(intensity)))
    return result


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class LLMAnnotator:
    """段落ごとに LLM へ問い合わせ、句ごとの感情を jsonl にキャッシュする。"""

    def __init__(self, cfg: LLMConfig, cache_path: Path) -> None:
        self.cfg = cfg
        self.cache_path = cache_path
        self.cache: dict[str, list[list[dict]]] = {}
        self._lock = threading.Lock()
        if cache_path.exists():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.cache[row["hash"]] = row["annotations"]

    def _chat(self, user_prompt: str) -> object:
        if self.cfg.api == "ollama":
            return self._chat_ollama(user_prompt)
        if self.cfg.api == "openai":
            return self._chat_openai(user_prompt)
        raise ValueError(f"llm.api は ollama か openai にしてください: {self.cfg.api!r}")

    def _messages(self, user_prompt: str) -> list[dict]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    def _chat_ollama(self, user_prompt: str) -> object:
        resp = requests.post(
            f"{self.cfg.base_url.rstrip('/')}/api/chat",
            json={
                "model": self.cfg.model,
                "messages": self._messages(user_prompt),
                "think": self.cfg.think,
                "format": JSON_SCHEMA,
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=self.cfg.timeout,
        )
        resp.raise_for_status()
        return json.loads(resp.json()["message"]["content"])

    def _chat_openai(self, user_prompt: str) -> object:
        headers = _auth_headers(self.cfg)
        body: dict = {
            "model": self.cfg.model,
            "messages": self._messages(user_prompt),
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "emotion_segments", "schema": JSON_SCHEMA, "strict": True},
            },
            "chat_template_kwargs": {"enable_thinking": self.cfg.think},
        }
        if not self.cfg.think:
            body["reasoning_effort"] = "none"
        url = f"{self.cfg.base_url.rstrip('/')}/chat/completions"
        for attempt in range(3):
            resp = requests.post(url, headers=headers, json=body, timeout=self.cfg.timeout)
            if resp.status_code != 400 or attempt == 2:
                break
            if attempt == 0:
                body["response_format"] = {"type": "json_object"}
            else:
                for key in ("response_format", "chat_template_kwargs", "reasoning_effort"):
                    body.pop(key, None)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return json.loads(_THINK_RE.sub("", content).strip())

    def annotate(
        self,
        target: list[Segment],
        prev: list[Segment] | None = None,
        nxt: list[Segment] | None = None,
    ) -> list[list[Annotation]]:
        clauses = [split_clauses(s.text) for s in target]
        shape = [len(c) for c in clauses]
        prev_text = "".join(s.text for s in prev) if prev else None
        nxt_text = "".join(s.text for s in nxt) if nxt else None
        key = paragraph_hash(self.cfg.model, clauses, prev_text, nxt_text)
        with self._lock:
            cached = self.cache.get(key)
        if cached is not None:
            return [[Annotation(**a) for a in row] for row in cached]
        prompt = build_user_prompt(prev_text, clauses, nxt_text)
        try:
            anns = validate(self._chat(prompt), shape)
        except (requests.RequestException, json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"[emotion] LLM の推定に失敗したため neutral にします: {e}")
            return validate(None, shape)  # 失敗はキャッシュしない
        rows = [[a.__dict__.copy() for a in row] for row in anns]
        with self._lock:
            self.cache[key] = rows
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"hash": key, "annotations": rows}, ensure_ascii=False) + "\n")
        return anns

    def annotate_many(
        self,
        items: list[tuple[list[Segment], list[Segment] | None, list[Segment] | None]],
        progress: Callable[[int, int], None] | None = None,
    ) -> list[list[list[Annotation]]]:
        workers = max(1, self.cfg.concurrency)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(self.annotate, t, p, n) for t, p, n in items]
            if progress is not None:
                for done, _ in enumerate(concurrent.futures.as_completed(futures), 1):
                    progress(done, len(futures))
            return [f.result() for f in futures]

    def unload(self) -> None:
        """Ollama のモデルを即時解放する。openai のときは LLMServer が担当。"""
        if not self.cfg.unload or self.cfg.api != "ollama":
            return
        try:
            requests.post(
                f"{self.cfg.base_url.rstrip('/')}/api/generate",
                json={"model": self.cfg.model, "keep_alive": 0},
                timeout=self.cfg.timeout,
            )
        except requests.RequestException as e:
            print(f"[emotion] モデルの解放に失敗しました: {e}")


def _health_url(cfg: LLMConfig) -> str:
    base = cfg.base_url.rstrip("/")
    return f"{base}/api/tags" if cfg.api == "ollama" else f"{base}/models"


class LLMServer:
    """`with LLMServer(cfg): ...` で、必要なら LLM サーバーを起動し、抜けるときに止める。"""

    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self._proc: subprocess.Popen | None = None

    def _healthy(self, timeout: float = 5.0) -> bool:
        try:
            resp = requests.get(_health_url(self.cfg), headers=_auth_headers(self.cfg), timeout=timeout)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def _stop(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    def __enter__(self) -> "LLMServer":
        if not self.cfg.server_command or self._healthy():
            return self
        print(f"[emotion] LLM サーバーを起動します: {' '.join(self.cfg.server_command)}")
        self._proc = subprocess.Popen(
            self.cfg.server_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        deadline = time.monotonic() + self.cfg.server_start_timeout
        while (remaining := deadline - time.monotonic()) > 0:
            time.sleep(min(1.0, remaining))
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            if self._healthy(timeout=min(5.0, remaining)):
                return self
        self._stop()
        raise RuntimeError(
            f"LLM サーバーが {self.cfg.server_start_timeout} 秒以内に起動しませんでした"
        )

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._stop()
        return False


def list_models(cfg: LLMConfig) -> list[str]:
    resp = requests.get(_health_url(cfg), headers=_auth_headers(cfg), timeout=cfg.timeout)
    resp.raise_for_status()
    data = resp.json()
    if cfg.api == "ollama":
        return [m["name"] for m in data["models"]]
    return [m["id"] for m in data["data"]]
