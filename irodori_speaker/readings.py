from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import yaml

from .config import Config

BOOK_READINGS = "readings.yaml"


def _load_file(path: Path) -> dict[str, str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"読み辞書は「漢字表記: よみ」の mapping にしてください: {path}")
    for k, v in data.items():
        if not isinstance(k, str) or not k or not isinstance(v, str):
            raise ValueError(f"読み辞書のキーと値は文字列にしてください（{k!r}: {v!r}）: {path}")
    return dict(data)


def load_readings(cfg: Config, out_dir: Path | None) -> dict[str, str]:
    """config の `readings` を順に、続けて本ごとの `readings.yaml` を読む（後勝ち）。"""
    table: dict[str, str] = {}
    for name in cfg.readings:
        path = Path(name).expanduser()
        if not path.is_file():
            raise ValueError(f"読み辞書が見つかりません: {name}")
        table.update(_load_file(path))
    if out_dir is not None:
        book = Path(out_dir) / BOOK_READINGS
        if book.is_file():
            table.update(_load_file(book))
    return table


def apply_readings(text: str, table: dict[str, str] | None) -> str:
    """最長一致の 1 パスで置換する（置換結果をさらに置換しない）。"""
    if not table:
        return text
    pattern = re.compile("|".join(re.escape(k) for k in sorted(table, key=lambda k: (-len(k), k))))
    return pattern.sub(lambda m: table[m.group(0)], text)


def readings_key(table: dict[str, str] | None) -> str | None:
    if not table:
        return None
    payload = json.dumps(sorted(table.items()), ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()
