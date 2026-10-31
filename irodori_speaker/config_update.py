"""config.example.yaml にあって config.yaml に無い設定を、例のコメントごと書き足す。

YAML を読み直して書き出すとコメントが消えるので、テキストの行として差し込む。
見るのは最上位のキーと `config._SECTIONS` の節の 2 段目のキーだけで、
`emotion_emoji` のような辞書の値は中身を見ない（1 つの値として扱う）。
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

import yaml

from .config import Config, _build, _SECTIONS

_KEY = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):(?:\s|$)")
_CHILD_KEY = re.compile(r"^([ \t]+)([A-Za-z_][A-Za-z0-9_]*):(?:\s|$)")
# 「# chapter_regex: …」のようにキーをコメントアウトした行。説明のコメントは対象にしない
_COMMENTED_KEY = re.compile(r"#\s*[A-Za-z_][A-Za-z0-9_]*:(?:\s|$)")
_DEFAULT_INDENT = "  "


@dataclass
class _Child:
    key: str
    lines: list[str]


@dataclass
class _Block:
    key: str
    lines: list[str]  # 直前のコメント + キーの行
    body: list[str] = field(default_factory=list)  # キーより深くインデントされた行


def _drop_commented_out(comments: list[str]) -> list[str]:
    """コメントアウトされたキーの行（と、その続きの深いインデントのコメント）を除く。"""
    kept, off = [], False
    for line in comments:
        s = line.strip()
        if _COMMENTED_KEY.match(s):
            off = True
            continue
        if off and re.match(r"#\s{2,}\S", s):
            continue
        off = False
        kept.append(line)
    return kept


def _split_top(lines: list[str]) -> list[_Block]:
    blocks: list[_Block] = []
    pending: list[str] = []
    gap: list[str] = []
    cur: _Block | None = None
    for line in lines:
        s = line.strip()
        if not s:
            gap.append(line)
            pending = []
            continue
        if line[0] in " \t":
            if cur is not None:
                cur.body.extend(gap)
                cur.body.append(line)
            gap = []
            continue
        gap = []
        m = _KEY.match(line)
        if m:
            cur = _Block(m.group(1), _drop_commented_out(pending) + [line])
            blocks.append(cur)
            pending = []
        elif s.startswith("#"):
            cur = None
            pending.append(line)
        else:
            cur = None
            pending = []
    return blocks


def _split_children(body: list[str]) -> tuple[str, list[_Child]]:
    indent: str | None = None
    children: list[_Child] = []
    pending: list[str] = []
    cur: _Child | None = None
    for line in body:
        s = line.strip()
        if not s:
            pending = []
        elif s.startswith("#"):
            pending.append(line)
        else:
            m = _CHILD_KEY.match(line)
            if m and indent in (None, m.group(1)):
                indent = m.group(1)
                cur = _Child(m.group(2), _drop_commented_out(pending) + [line])
                children.append(cur)
            elif cur is not None:
                cur.lines.append(line)
            pending = []
    return indent or _DEFAULT_INDENT, children


def _reindent(lines: list[str], src: str, dst: str) -> list[str]:
    return [dst + ln[len(src):] if ln.startswith(src) else ln for ln in lines]


def _load(text: str, what: str) -> tuple[dict[str, Any], Config]:
    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"{what}を YAML として読めません: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"{what}の最上位はキーと値の対応にしてください")
    return data, _build(Config, data)


def _same_value(a: Any, b: Any) -> bool:
    """値も型も同じか。整数の設定に 1.0 のような浮動小数を足すと動きが変わるので別とみなす。

    浮動小数の設定（既定 300.0）に整数（300）を書くのは無害なので、そこだけ同じとみなす。
    """
    if type(a) is int and type(b) is float:
        return a == b
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same_value(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same_value(x, y) for x, y in zip(a, b))
    return a == b


def _default_of(section: str | None, key: str) -> Any:
    base = Config()
    if section is not None:
        base = getattr(base, section)
    return getattr(base, key)


def _value_of(cfg: Config, section: str | None, key: str) -> Any:
    return getattr(getattr(cfg, section) if section else cfg, key)


def _find_section(lines: list[str], key: str) -> tuple[int, int]:
    """config.yaml の節の行番号と、節に属する最後の行の行番号を返す。"""
    head = re.compile(rf"^{re.escape(key)}:(?P<rest>.*)$")
    for i, line in enumerate(lines):
        m = head.match(line)
        if not m:
            continue
        rest = m.group("rest").strip()
        if rest and not rest.startswith("#"):
            raise ValueError(
                f"config の {key} がブロック形式ではないため、キーを書き足せません。手で直してください"
            )
        last = i
        for j in range(i + 1, len(lines)):
            if not lines[j].strip():
                continue
            if lines[j][0] in " \t":
                last = j
            elif lines[j].startswith("#"):
                continue  # インデントの無いコメントは節を終わらせない
            else:
                break
        return i, last
    raise ValueError(f"config の {key} の行が見つかりません（ブロック形式で書いてください）")


def _child_indent(lines: list[str], head: int, last: int, default: str) -> str:
    for line in lines[head + 1 : last + 1]:
        if line.strip() and not line.strip().startswith("#"):
            return line[: len(line) - len(line.lstrip())]
    return default


def update_config_text(
    text: str, example_text: str
) -> tuple[str, list[str], list[tuple[str, Any]]]:
    """example にあって text に無いキーを書き足す。

    返り値は (新しいテキスト, 足したキー, 手で確かめるキーと例の値)。
    例の値が組み込みの既定値と違うキーは足さず、手で確かめる側に入れる。
    """
    data, before = _load(text, "config")
    ex_data, ex_cfg = _load(example_text, "例のファイル")

    lines = text.splitlines()
    if [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")] == ["{}"]:
        lines = [ln for ln in lines if ln.strip() != "{}"]
    nl = "\r\n" if "\r\n" in text else "\n"

    added: list[str] = []
    manual: list[tuple[str, Any]] = []
    inserts: dict[int, list[str]] = {}
    tail: list[list[str]] = []

    for block in _split_top(example_text.splitlines()):
        key = block.key
        if key not in ex_data:
            continue
        is_section = key in _SECTIONS and isinstance(ex_data[key], dict)
        if not is_section:
            if key in data:
                continue
            ex_value = _value_of(ex_cfg, None, key)
            if not _same_value(ex_value, _default_of(None, key)):
                manual.append((key, ex_value))
            else:
                added.append(key)
                tail.append(block.lines + block.body)
            continue

        ex_indent, children = _split_children(block.body)
        present = data[key] if key in data else None
        have = set(present) if isinstance(present, dict) else set()
        wanted: list[_Child] = []
        for child in children:
            if child.key in have:
                continue
            ex_value = _value_of(ex_cfg, key, child.key)
            if not _same_value(ex_value, _default_of(key, child.key)):
                manual.append((f"{key}.{child.key}", ex_value))
            else:
                wanted.append(child)
        if not wanted:
            continue
        if key not in data:
            added.append(key)
            tail.append(block.lines + [ln for c in wanted for ln in c.lines])
            continue
        added.extend(f"{key}.{c.key}" for c in wanted)
        head, last = _find_section(lines, key)
        indent = _child_indent(lines, head, last, ex_indent)
        new = [ln for c in wanted for ln in _reindent(c.lines, ex_indent, indent)]
        inserts.setdefault(last, []).extend(new)

    if not added:
        return text, [], manual

    out: list[str] = []
    for i, line in enumerate(lines):
        out.append(line)
        out.extend(inserts.get(i, []))
    if tail:
        while out and not out[-1].strip():
            out.pop()
        for blk in tail:
            if out:
                out.append("")
            out.extend(blk)
    new_text = nl.join(out) + nl

    _, after = _load(new_text, "書き足した結果")
    if asdict(after) != asdict(before):
        raise ValueError("書き足すと設定の値が変わってしまうため、書き込まずに中止しました")
    return new_text, added, manual
