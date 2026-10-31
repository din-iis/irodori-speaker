# irodori-speaker: 聴くための 5 機能（書き出し・品質チェック・場面区切り・読み辞書・LoRA）

> **実装後の差分（2026-10-31）**: 自動の題は `AUTO_TITLES` の文字列ではなく `text_splitter._split_chapters_marked` の旗で判定する（実際の見出し「序」も読むため）。章署名には `kind` の並びも入る。`is_separator` は U+1F000 以上か U+FE0F を含む行（パレット外の絵文字）を区切りにしない。`check` には `silent` の印が加わり、キー計算は `check.segment_key(synth, row)` = `synth.segment_key(text, seed=...)`。`export` は入力の stat も署名に入れ、失敗時・書き出し中の入力更新時は章 MP3 も含めて出力を消す。詳細は CLAUDE.md。

## Context

irodori-speaker は小説を Irodori-TTS で読み上げ、`bgm` で環境音と音楽を重ねるツール。2026-10-31 の提案で、「読んでいない小説を iPhone で聴く人が、通しで聴き直さずに悪い文を見つけて直せること」「スマホで聴ける形になっていること」を物差しに 6 件を挙げ、ユーザーが 1〜5 の実装を決めた（6 の登場人物ごとの声は見送り）。

今の出力は 48 kHz PCM16 の WAV（約 350 MB/時）だけで MP3 は `bgm` 経由のみ。`warnings.log` は 30 秒の切り詰めしか記録しない。`＊` や `◇　◇　◇` の行は地の文として TTS に渡り、章の見出しは読まれない。読みの修正は本ごとの `text` 手直しかルビだけ。LoRA は統合 `.safetensors` でしか使えない（上流の固定コミット 89f9d8f は `SamplingRequest.lora_adapter` で未統合アダプタを動的に読める。確認済み）。

### ユーザー決定（2026-10-31）

| 項目 | 決定 |
|---|---|
| 書き出し形式 | **選べるようにする**。既定は M4B（AAC、章マーカー付き、Apple Books 向け）、MP3 も選べる。聴く端末は iPhone、サイズ重視 |
| 品質チェックの ASR | **任意の `--asr` で入れる**。既定は信号だけの判定、`--asr` のときだけ kotoba-whisper で文字誤り率 |
| 章題の読み上げ | **既定で読む**（`read_chapter_title: true`）。既存の本は章 WAV の結合し直しになるが文のキャッシュは残る |
| 進め方 | オーケストレーションで並列（ワーカーは Sonnet、1 タスク = 1 ワークツリー） |

### コーディネーターの判断

- 5 機能を **4 つのワークツリー** に分ける。読み辞書と LoRA はどちらも `synth.py` の `_request` / `cache_key` 周りの小さな変更なので 1 本にまとめる。
- ワーカーは README と `config.example.yaml` の **割り当てた節だけ** 編集し、**CLAUDE.md はコーディネーターが全マージ後に一度だけ更新**する。
- `export` の本の題名は小説ファイル名（stem）を既定にし、`export.title` で上書きできる。
- README では未統合の LoRA アダプタも `voice: none` との組み合わせを推奨する（統合 checkpoint と同じ扱い。参照音声ありでも動く）。
- `check` は印があっても終了コード 0（`synth` が警告件数を表示するだけなのと揃える）。`--voice` / `--checkpoint` を受け取る（`synth --voice X` で作った本の鍵を合わせるため）。
- 新しいハッシュ項目は「値があるときだけ」キーと署名に入れ、既存のキャッシュと `.sig` を無効にしない。

## 工程の分け方と衝突の管理

| タスク | ブランチ | 主に触るファイル |
|---|---|---|
| A 書き出し `export` | `feature/export` | 新規 `export.py`、`audio_io.py`（追記）、`pipeline.py`（`build_export` 追加）、`cli.py`、`config.py`、`webui.py`（書き出しボタン）、README 新節 |
| B 品質チェック `check` と取り直し | `feature/check-retake` | 新規 `check.py`、`pipeline.py`（`TSV_FIELDS`・`_row`・`synthesize_book` の行ループ・`chapter_signature`・`check_book`）、`synth.py`（`cache_key` / `synthesize` の `seed`）、`cli.py`、`webui.py`（取り直し・チェック）、README 新節 |
| C 場面区切り・章題・間 | `feature/scene-breaks` | `text_splitter.py`、`assemble.py`（全体）、`pipeline.py`（`annotate_chapters` の照合・`synthesize_book` の break 飛ばし）、`config.py`、`bgm_scenes.py`（1 行）、README |
| D 読み辞書と LoRA | `feature/readings-lora` | 新規 `readings.py`・`lora.py`、`synth.py`（`_request`・`cache_key` の 1 項目・`lora_key`）、`config.py`、`cli.py`、`webui.py`（LoRA 欄・`Engine.key_for`）、README |

- 同じ関数を複数が触る箇所: `synth.cache_key`（B: `seed` 引数、D: `lora` 項目）、`pipeline.chapter_signature`（B: 行 seed の一覧、D: `readings_key` / `lora_key`）、`synthesize_book` の行ループ（B: 行 seed、C: break 行の skip）。いずれも数行で、マージ時にコーディネーターが解消する。
- マージ順: **A → C → D → B**。B のマージ後、`check` のキー計算に読み辞書を通す統合をコーディネーターが行う。
- `assemble.py` は C だけが編集する（B が必要とする文ごとの時刻 `segments` も C が書く）。

## 共通の前提（全ワーカー）

- Mac には torch / irodori_tts / GPU が無い。torch・transformers・irodori_tts はトップレベルで import しない。テストは `uv run pytest -q` で全部通す（着手時点で 484 件通過・10 件スキップ）。
- gradio のテストは `pytest.importorskip("gradio")`。ffmpeg のテストは `pytest.mark.skipif(not ffmpeg_available())`。
- 「手直しが正」: `source` 列は編集しない。既存の行の `labels` / `text`（B 以降は `seed` も）は `source` が一致する限り保つ。
- 文のキャッシュキー（`synth.cache_key`）と章署名（`pipeline.chapter_signature`）に入るものを変えるときは、理由をテストで固定する。
- **ハッシュの後方互換**: `cache_key` / `chapter_signature` に新しい項目を足すときは、**値があるときだけ**含める（`None` や空を無条件に足すと既存の全キャッシュ・全 `.sig`・連鎖で `bgm` の `.sig` が無効になる）。B の行 seed も「どれかの行に seed があるときだけ」署名に入れる。
- commit は日本語 1 行（例: `export: 章マーカー付き M4B と MP3 に書き出す`）、push しない。

## A. 書き出し `export`（`feature/export`）

### 決定

- 常に全章。narration は章 WAV と `<wav>.sig`・`.times.json`（frames が WAV と一致、段落の並びが `script.tsv` と一致）、bgm（`--bgm`）はさらに `bgm/chapters/<stem>.mp3`・`.sig`・`.sounds.json` を要求し、欠ければ `ValueError`（`synth --chapters N` / `bgm --chapters N` を案内）。検証は `bgm_mix.load_times` / `TimesMismatch` を再利用する（`bgm_mix` は torch を import しない。確認済み）。
- 章の開始時刻は、**実際にエンコードする入力ファイル**（narration は章 WAV、bgm は `bgm/chapters/*.mp3`）を `soundfile.info` で読んだ `frames` の累積 ÷ sr をミリ秒に丸める（`TIMEBASE=1/1000`）。`.times.json` の frames は朗読 WAV の検証にだけ使い、マーカーには使わない（bgm の MP3 は intro / bridge の分だけ WAV より長い）。libsndfile 1.2.2 は MP3 を読める（確認済み）。章で sr が違えば `ValueError`。
- 章題は `chapters.json` の生のタイトル（空なら `第N章`）。本の題名は `export.title`（null なら `out_dir.name` = 小説の stem）。
- 出力: m4b は `export/<book>.m4b`（bgm は `<book>_bgm.m4b`）。mp3 は `export/chapters/NN_title.mp3` + `export/<book>.mp3`（bgm は `chapters_bgm/` と `<book>_bgm.mp3`、既存の `concat_mp3` で通し）。narration `-ac 1`、bgm `-ac 2`。
- `<出力>.sig`（入力 `.sig` の中身の列 + source + format + bitrate + channels + 題名列の sha1）が一致すれば再エンコードしない。tmp → replace、失敗時は出力と `.sig` を消す。
- ffmpeg（`audio_io._run` 経由。`-nostdin -loglevel error` は `_run` が付ける）: `-y -f concat -safe 0 -i list.txt -f ffmetadata -i meta.txt -map 0:a -map_metadata 1 -map_chapters 1 -c:a aac -b:a 64k -ac 1 -movflags +faststart -f ipod out.tmp.m4b`。`list.txt` は `concat_mp3` と同じ書式。`meta.txt` は `;FFMETADATA1` + `title` / `album` + `[CHAPTER]` ブロック（`TIMEBASE=1/1000`、`START`、`END`、`title`。`= ; # \` と改行をエスケープ）。mp3 は `-c:a libmp3lame -b:a <bitrate> -ac N -metadata title=<章題> -metadata track=N`。

### ファイル

- 新規 `irodori_speaker/export.py`（純粋ロジック。torch・ffmpeg に触れない）: `FORMATS`, `ExportInput(idx, title, wav, bgm_mp3=None)`, `ExportChapter(idx, title, src, sr, frames, sig)`, `chapter_title(idx, title)`, `collect_chapters(inputs, rows, *, source)`（検証はすべてここ。`.sig` の場所は `pipeline._sig_path` を関数内 import）, `chapter_starts_ms(chapters)`, `ffmetadata(book, chapters)`, `escape_ffmeta(value)`, `export_paths(out_dir, *, source, fmt) -> (通し, 章ディレクトリ or None)`, `chapter_mp3_path(chapter_dir, ch)`, `export_signature(chapters, *, source, fmt, bitrate, channels)`。
- `audio_io.py` 追記: `encode_file(src, dst, args)`（`["-y", "-i", src, *args, tmp]` → replace）, `write_m4b(inputs, metadata, out, bitrate, channels)`（`TemporaryDirectory` に `list.txt` / `meta.txt`）, `_concat_list(paths)`（`concat_mp3` のリスト生成を切り出して共用）。
- `config.py`: `ExportConfig(format="m4b", bitrate="64k", title=None)` を `Config.bgm` の直後に `export:`、`_SECTIONS` 末尾、`_build` で format ∈ {m4b, mp3}・bitrate が `^\d+k$` を検証（日本語 `ValueError`）。`config.example.yaml` は `bgm:` の後・`output_dir` の前。
- `pipeline.py` 末尾: `ExportResult(path, chapter_files, skipped)`、`build_export(cfg, out_dir, *, source="narration", fmt=None, bitrate=None, progress=None) -> ExportResult`（`audio_io` / `export` は関数内 import。`load_book` → `_chapter_path` から `ExportInput` → `collect_chapters` → `export_paths` → 署名比較 → `write_m4b` / `encode_file` + `concat_mp3` → `.sig`。失敗時は出力と `.sig` を unlink）。progress は m4b が `(0,1,…)`→`(1,1,…)`、mp3 は章数 + 1。
- `cli.py`: `export` パーサーは `common(p)` を使わず `novel` / `-c` / `--format {m4b,mp3}` / `--bitrate` / `--bgm`（store_true）。`_run_export(args, cfg, out_dir)`（`_run_bgm` と同じ tqdm、`ValueError` / `RuntimeError` → 1、終了時にパスとサイズ MB を表示、`skipped` なら「変更なし」）。`main` の `bgm` 分岐の隣に `export`。
- `webui.py`: `from .pipeline import build_export`（テストが `webui.build_export` を monkeypatch できるように）。小説生成タブの末尾に `gr.Row`: `export_format = gr.Dropdown(["m4b","mp3"], value=cfg.export.format, label="書き出し形式")`、`export_bgm = gr.Checkbox(label="BGM 付き（bgm の MP3 から）")`、`export_btn = gr.Button("書き出し")`、`export_file = gr.File(label="書き出したファイル")`。ハンドラ `export_book(novel_file, novel_text, fmt, with_bgm)` は `_novel_path` → `build_export(cfg, book_dir(cfg, novel), source=..., fmt=fmt)` → `(str(result.path), "✅ 書き出しました: …")`。`ValueError` / `RuntimeError` は `fail()` で `gr.Error`。**`**gpu` を付けない**。出力は `[export_file, novel_status]`。
- README: CLI 冒頭ブロックに `# 3. 聴く形式に書き出す`、`### 出力` ツリーに `export/`、`### 私的な利用に限る` と `## WebUI` の間に新節 `## 聴く形式に書き出す（export）`（iPhone・Apple Books、AirDrop / 「ファイル」経由の転送、オプション表、`export.*`、前提、再実行時のスキップ、bgm は 192k→再エンコードなので 96k 以上を推奨）、`### タブ「小説生成」` に書き出し、設定表の `output_dir` 直前に `export.*` の行。

### テスト（`tests/test_export.py` のみ）

純粋: 章題の既定、開始ミリ秒の累積、ffmetadata の先頭と章ブロック、エスケープ（`= ; # \` 改行）、出力パス（narration / bgm × 形式）、`collect_chapters` の拒否 6 種（WAV 無し / `.sig` 無し / frames 不一致 / 段落不一致 / sr 不一致 / bgm の MP3・`.sig`・`.sounds.json` 無し）、署名の変化、設定の既定と検証、CLI（`cli.build_export` を monkeypatch、`--format mp3 --bitrate 96k --bgm` の受け渡し、終了コード 1、日本語 help、`--chapters` が無い）、WebUI（`_handlers` 方式、成功と `gr.Error`）。
`needs_ffmpeg`（入力は `tests/test_pipeline_bgm.py` の `make_book` と同様に `write_chapter` + 偽 `.sig` で作る）: m4b の章マーカーと題名（`audio_io._run(["-i", m4b, "-f", "ffmetadata", "-"])` の出力を `[CHAPTER]` ごとに解析、START は ±50 ms）、モノラル（`-f f32le` のサンプル数が frames の約 1 倍）、`encode_file` が tmp を残さない、mp3 の章ごと + 通し、bgm はステレオ、署名一致で再エンコードしない（`write_m4b` を boom に差し替え）、失敗時に出力と `.sig` を残さない。

### リスク

AAC のプライミング遅延で数十 ms ずれる（章マーカー用途では問題なし）。内蔵 `aac` は `libfdk_aac` より劣るが 64k モノラルの朗読では十分。`--bgm` は 192k ステレオを再エンコードするので劣化する（README で 96k 以上を勧める）。

## D. 読み辞書と話者 LoRA（`feature/readings-lora`）

### 決定

- 読み辞書はモデルに渡る経路で 1 回だけ適用: `_request(text)` が `apply_readings` → `tts_text`。`synthesize` のキーは適用後の本文で取る（`segment_cache_key`）。`_request` には生の `text` を渡す（二重適用しない）。`generate_once` も `_request` 経由なので試聴にも効く。`warnings.log` は生の `text`。
- 置換は最長キー優先に並べた `re.escape` の選択肢を 1 パスの `re.sub` で（逐次 `str.replace` だと `花→はな` の後に `はな→…` と連鎖する）。
- 辞書の場所: `readings: []`（YAML のパス、cwd 相対、`~` 展開、無ければ `ValueError`「読み辞書が見つかりません: …」）+ `out/<book>/readings.yaml`（あれば後勝ち）。書式は `漢字表記: よみ` の平らな mapping（例 `紫陽花: あじさい`）。空ファイルは `{}`。それ以外は日本語の `ValueError`。
- LoRA: `lora_adapter: null`（アダプタのディレクトリ）。`Synthesizer.__init__` で `resolve_lora_adapter`（ディレクトリ・`adapter_config.json`・`adapter_model.safetensors|bin` を検証、不足ファイル名を含む日本語 `ValueError`、runtime に触る前）。`lora_key` は `adapter_config.json` と `adapter_model.*` の（ファイル名 + 中身）の sha1（mtime は使わない）。`_request` は `lora_path` があるときだけ `lora_adapter=str(path)` を渡す。上流が「アダプタなし」と解釈する `none` / `off` / `base` は `normalize_lora` で `None` に寄せる。
- `load_runtime(cfg, checkpoint=None, lora_key=None)` は `_LOADED_KEY = (RuntimeKey, lora_key)` で管理し、組が変われば既存の clear → `check_vram` → get を通る（アダプタを VRAM に積み上げない。外したときも読み直す）。
- `cache_key(cfg, text, latent_key, lora_key=None)` は `"latent"` の次に `{"lora": lora_key}` を**あるときだけ**追加。`chapter_signature` は `latent_key` の直後に `*_synth_identity(synth)`（`readings_key` / `lora_key` のどちらかがあるときだけ `[{"readings": …, "lora": …}]`、どちらも無ければ空）。
- WebUI の `Engine.key_for(checkpoint, device, precision, voice, lora_adapter=None)` は 5 要素（空欄は `cfg.lora_adapter`、入力があれば `normalize_lora`、`none` で config を打ち消す）。`cfg_for` も `lora_adapter=` を replace。`get(key, segments_dir, warnings_path=None, out_dir=None)` / `load(...)` は呼ぶたびに `synth.readings = load_readings(cfg, out_dir)` を差し替える（Synthesizer はキーごとに使い回すが辞書は本ごと）。`ensure_loaded(key, segments_dir, warnings_path=None, out_dir=None)`。`stage_synth` は `out_dir` を渡す（試聴は渡さない = config の辞書だけ）。`lora` の Textbox は TTS 設定 2 行目（声の Dropdown の隣）、各 `.click` の inputs **末尾**に足し、ハンドラ（`load_model` / `generate` / `compare_emotions` / `run_novel` / `stage_synth`、`tts_key`）は末尾の kwarg `lora_adapter=None`（既存の位置引数のテストを壊さない）。

### ファイル

- 新規 `readings.py`: `load_readings(cfg, out_dir: Path | None) -> dict[str, str]`, `apply_readings(text, table) -> str`（空なら text をそのまま）, `readings_key(table) -> str | None`（空なら None、sorted items の sha1）。
- 新規 `lora.py`: `NO_LORA = ("none", "off", "base")`, `normalize_lora(value) -> str | None`, `resolve_lora_adapter(value) -> Path | None`, `lora_key(path) -> str`。
- `synth.py`: `cache_key` の 1 行、`segment_cache_key(cfg, text, latent_key, *, readings=None, lora_key=None)`（= `cache_key(cfg, apply_readings(text, readings), latent_key, lora_key=lora_key)`。B の `check` がマージ後に使う）、`load_runtime` の `lora_key`、`Synthesizer(..., readings=None)` に `readings` / `lora_path` / `lora_key` / `readings_key` プロパティ / `segment_key(text)`（`synthesize` もこれを使う）。`runtime` プロパティは `load_runtime(self.cfg, self._checkpoint, lora_key=self.lora_key)`。
- `config.py`: `Config.voice` の直後に `readings: list[str] = field(default_factory=list)` と `lora_adapter: str | None = None`。`_build` の `if cls is Config:` で `readings` が list[str] でなければ `ValueError`。`config.example.yaml` は `max_ref_seconds` の直後。
- `pipeline.py`: `chapter_signature` の 1 行と `_synth_identity(synth) -> list`。
- `cli.py`: `synth_opts` に `--lora-adapter`（help「話者 LoRA のアダプタディレクトリ（config を上書き）。none で無効」）、`_run_synth` で `--checkpoint` と同じ位置に `replace(cfg, lora_adapter=normalize_lora(...))`、`_make_synthesizer(cfg, out_dir, voice, checkpoint)` は 4 引数のまま中で `load_readings(cfg, out_dir)`（「読み辞書: N 語」「話者 LoRA: <path>」を表示）。
- README: 設定表の `voices_dir / voice` 直後に `readings` / `lora_adapter`、`### script.tsv を手直しする` の直後に `### 読み辞書`（書式、`readings:` と `out/<book>/readings.yaml`、最長一致で全文に適用、文脈で読みが変わる語はルビで、辞書を変えると本文が変わる文だけ再合成され章 WAV は組み直し、`script.tsv` は変えない）、`## 声の作り方` 末尾（`### キャプションで感情を試す` の前）に `### 話者 LoRA をアダプタのまま使う`（ディレクトリ構成、`lora_adapter:` / `--lora-adapter` / WebUI の欄、`voice: none` との組み合わせを推奨、アダプタを変えると runtime を読み直す、キャッシュキーにアダプタの中身が入る、統合 checkpoint の手順はそのまま使える）、`## 注意` の「漢字の読み間違い」に辞書を追記。

### テスト（新規 `tests/test_readings.py` / `tests/test_lora.py` のみ）

readings: 最長一致、連鎖しない、絵文字入り、空辞書、`readings_key` の順序非依存、config と本ごとの後勝ち、無いファイル・不正な形式（list / 数値キー）の拒否、本ごとの辞書が無ければ config だけ（`out_dir=None` も）、`readings` がリストでなければ拒否、辞書で本文が変わる文だけキーが変わる、`synthesize` が置換後でモデルを呼びキャッシュする（`make_synth` 相当のフィクスチャを複製。`rt.calls[0].text`、WAV 名が `segment_key`）、`warnings.log` は置換前、`generate_once` も適用、署名は辞書があるときだけ変わり無ければ従来と同じ、`_make_synthesizer` が本ごとの辞書を渡す、`Engine.get` が `out_dir` ごとに差し替える（voice `none`）。
lora: `normalize_lora`、ディレクトリ無し・`adapter_config.json` 無し・重み無しの拒否、`safetensors` か `bin` のどちらかを受け付ける、`lora_key` は中身で変わり mtime で変わらない、`cache_key` は lora が無ければ従来と同じ、runtime に触る前に検証（`fake_upstream` 相当で `order == []`）、`_request` は lora があるときだけ渡す、`load_runtime` の読み直し 3 通り（変更 / 外す / 同じなら clear しない）、署名、CLI の `--lora-adapter` と `none`、`Engine.key_for` / `cfg_for`、`generate` の末尾の `lora` 引数で既存の位置引数呼び出しも通る。

### リスク

1 パス置換は左端優先なので重なる語（`紫陽花火`）は意図どおりにならない（README でルビを勧める）。`readings_key` を署名に入れると 1 語足しただけで全章が組み直しになる（文キャッシュは効く。`bgm` の `.sig` も変わる）。

## C. 場面区切り・章題・間（`feature/scene-breaks`）

### 決定

- 区切り行の判定 `text_splitter.is_separator(line)`: `\w` にマッチする文字（かな・漢字・英数字）も括弧（`OPEN_BRACKETS + CLOSE_BRACKETS`）も Irodori の絵文字（`emoji_runs`）も含まない行。`＊`・`◇　◇　◇`・`* * *`・`……`・`――` が該当し、`「……」`・`1`・`😊` は該当しない。
- `build_chapters(text, regex, max_chars, *, read_chapter_title=True)`: 題が `AUTO_TITLES = ("序", "全編")` でなければ段落 0 に `Segment(kind="title", text=題)` を置き、本文は段落 1 から。区切り行は `Segment(kind="break", text=行そのもの)`（`source` になる）。連続する区切りは 1 つにまとめ、本文の先頭（題の直後を含む）と末尾の区切りは捨てる。区切りだけの章は従来どおり落ちる。`Segment.kind` は `"narration" | "dialogue" | "title" | "break"`。呼び出し元（`pipeline.build_script`、`cli._run_script`）は `cfg.read_chapter_title` を渡す。
- script.tsv: break 行は `labels=""`・`text=""`（手直しがあっても `text` は常に空）、title 行は `labels="neutral:1"`・`text=題`。どちらも LLM に渡さず、**LLM の文脈 `prev` / `nxt` にも入れない**（入れると `annotations.jsonl` のキーが隣の段落すべてで変わる）。`plan` は章ごとに `body = [segs for segs in chapter.paragraphs if segs[0].kind not in ("title", "break")]` を作り、文脈は `body` の中の隣。
- `SilenceConfig` に `title: 1.0`（章題の後）、`scene: 2.0`（区切りの位置）、`turn: 0.4`（同じ段落で台詞と地の文が切り替わる所）、`ellipsis: 0.6`（前の文が `…‥―—` で終わる所、閉じ括弧の前でも）。`Config.read_chapter_title: bool = True`。間は「基本（文 / 段落）と特別の**大きい方**」。
- `assemble.Clip(path: Path | None, chapter_idx, para_idx, seg_idx, kind="narration", text="")`。`ends_with_ellipsis(text)` と `gap_seconds(prev, clip, silence, after_break)` を関数に切り出す。`write_chapter` は `sr` を `path` のある最初のクリップから取り、break クリップは書かずに `paragraphs` に幅 0 の `[para, pos, pos]` を記録して `after_break = True`（bgm の段落集合の事前検証と `merge_short` / `build_layout` はこれで壊れない。確認済み）。`.times.json` に `"segments": [[para, seg, start, end], ...]`（音声のあるクリップだけ、サンプル単位）を追加し、`paragraphs` は変えない。章末が break なら末尾に無音は足さない。音声の文が 1 つも無い章は `ValueError`。
- **手直しの保全**: 題や区切りの挿入で後ろの段落番号がずれるので、`_match_saved_rows(chapters, overrides) -> dict[key, row]` を 2 パスで一度だけ解決する。パス 1: break 以外の全セグメントで `overrides.get(s.key)` があり `source == s.text` なら採用し `used` に入れる。パス 2: 未解決の文を文書順に、同じ章（`k[0] == s.chapter_idx`）で `source == s.text` かつ `used` に無い行を `sorted(overrides.items())` の順に 1 つ取り出して採用（消費）。2 パスにするのは、繰り返される文（`「はい」` など）で先の文の fallback が後の文のキー一致行を奪わないため。同文が複数あるときは文書順に対応づける（1 文消えていると隣の行に移る。許容し README に書く）。`kept = all(s.key in saved for s in segs)`。行の生成は `row_for(s)`: break → `_row(s, "", "")`、`saved.get(s.key)` → `_row_from_saved(s, r)`（`_row(s, r["labels"], r["text"])`）、それ以外 → `_row(s, "neutral:1", s.text)`。LLM ループ内も `saved.get` → `_row_from_saved`。`saved_row` / `neutral_row` は消す。`reannotate` は `saved = {}` で fallback も捨てる。`processed` の他章マージ、消えた章の削除、`--chapters` は変更なし。
- `synthesize_book` は `kind == "break"` の行を合成せず（text に何かあっても無視）`Clip(None, ch, para, seg, kind="break")` を積む。音声クリップには `kind=r.get("kind", "narration")`, `text=spoken_text(r)` を渡す。進捗の `total` は break を除く。`chapter_signature` は触らない（`(para, seg)` の並びと空 text で自然に変わる）。
- `bgm_scenes.chapter_paragraphs` はループ先頭で `if r.get("kind") == "break": continue`（title 行は段落 0 として LLM に見せてよい）。`webui.rows_from_dataframe` は `original.get("kind") == "break"` なら `text` を空にする。
- README: `## 仕組み` の分割の説明に「見出し行は 1 文の `title`、文字の無い行（`＊`・`◇　◇　◇`・`……` など）は `break` になり合成しない」、`### script.tsv を手直しする` の `kind` を 4 種に、`## 設定` の `silence.*`（7 値）と `read_chapter_title`、`## 注意` に区切り行の扱いと「区切りを足すと後ろの段落番号がずれるが、同じ章で原文が同じ行の手直しは引き継ぐ」、既存の本は章 WAV が組み直しになること（文のキャッシュは残る）。

### テスト

新しいテストは `tests/test_pipeline_layout.py` / `tests/test_webui_breaks.py`（`fake_emotion` は `tests/test_pipeline.py` から import せず、同じ fixture を小さくコピーするか `conftest.py` に移す）。`tests/sample_novel.txt` は変えない（区切りのテストは `tmp_path` に小説を書く）。既存の修正は `test_build_chapters_kinds`（先頭が title）、`test_write_chapter_writes_paragraph_times`（`segments` の期待値 `[[0,0,1000,1100],[0,1,1110,1310],[1,0,1410,1460]]` を足す）、`test_manual_edit_kept_per_row_within_changed_paragraph`（キーが 1 つずれる）に限る。
splitter: 区切り 5 種が break の 1 文になる（`text` が行そのもの）、文字・括弧・絵文字を含む行は区切りにしない、連続は 1 つにまとめ先頭と末尾は捨てる、区切りだけの章は落ちる、題のある章は para 0 が title で本文は 1 から、自動の題は読まない、`read_chapter_title=False`。
assemble: scene の無音と幅 0 の段落（a=100 / break / c=50、sr 1000、scene 0.5 → `paragraphs [[0,1000,1100],[1,1100,1100],[2,1600,1650]]`、`segments` に break 無し、frames 1650）、title の無音、同じ段落の台詞⇔地の文で turn（段落をまたぐ切り替えは paragraph のまま）、三点リーダで ellipsis（`そう……」` も）、大きい方の規則（scene < paragraph なら paragraph）、音声の文が無ければ `ValueError`、`Clip` は `kind` と `text` を省略できる。
pipeline: break 行の形、break と title を LLM に渡さず文脈にも入れない（`annotate_calls` の prev/nxt）、題の行は neutral で `text` は見出し、題の読みの手直しが保たれる、区切り挿入で段落がずれても後続の手直しが保たれ LLM も呼ばれない、同文複数の文書順対応、キー一致が fallback に奪われない、`--reannotate` は fallback も捨てる、`--chapters` 一部処理で他章の break と title が残る、消えた文と区切りの行が落ちる、`synthesize_book` が break を合成しない（`FakeSynth.texts` に入らない）、進捗が break を数えない、区切りのある章の `.times.json` の段落集合が `script.tsv` と一致する（`bgm_mix.load_times` で読み `_build_bgm` の事前検証と同じ比較）、`cli script` が `read_chapter_title: false` を反映。`tests/test_bgm_scenes.py` に break を飛ばす 1 件、`tests/test_config.py` に既定値 1 件、`tests/test_webui_breaks.py` に `rows_from_dataframe` 1 件。

### リスク

`asdict(cfg.silence)` が署名に入るので既存の章 WAV はすべて stale（文キャッシュは残る。`bgm` の `.sig` も変わり重ね直し）。題の挿入でも同じなので避けない。`is_separator` の `\w` は `_` も文字扱い。`！？` だけの行は区切りになる（意図どおり）。

## B. 品質チェック `check` と取り直し（`feature/check-retake`）

### 決定

- `synth.row_seed(row) -> int | None`（`seed` 列、空なら None）。`cache_key(cfg, text, latent_key, seed=None)` は `seed` が None なら `sampling.seed`。`Synthesizer.synthesize(text, force=False, seed=None)` は `seed` が None のとき `_request` に渡さない（`params.update` で config の seed が None に化けるのを防ぐ）。
- `TSV_FIELDS` 末尾に `seed`。`load_script` は読んだ行に `for f in TSV_FIELDS: r.setdefault(f, "")`（古いファイルも読める。`dataframe_rows` の `r[f]` が落ちない）。`_row(s, labels, text, seed="")`、saved 行からは `r.get("seed", "")` を引き継ぐ（C とのマージ後は `_row_from_saved` の中で渡す）。LLM で作った新しい文は空。`synthesize_book` は `seed=row_seed(r)` を渡す。`chapter_signature` は**どれかの行に seed があるときだけ** `[row_seed(r) for r in rows]` を末尾に足す。
- **`check` のキー計算は Synthesizer に寄せる**: `check_book` と WebUI の `select_row` は `Synthesizer(cfg, segments_dir, voice_paths, latent_cache_dir)` を作って（`__init__` は torch に触れない。`tests/test_synth.py` の `make_synth` と同じ）`synth.latent_key` を読み、キーは `check.segment_key(synth, row)` = `cache_key(synth.cfg, row["text"], synth.latent_key, row_seed(row))` とする。別実装の `book_latent_key` は持たない。マージ後は `check.segment_key` の中身を D の `synth.segment_key(text, seed=...)`（読み辞書と `lora_key` 込み）に替えるだけで済む。voice は `none` なら `[]`、それ以外は `resolve_voice`（`cli._make_synthesizer` と同じ）。
- 新規 `check.py`（numpy / soundfile だけ）: `CHECK_FIELDS = ["chapter","para","seg","key","seconds","chars","sec_per_char","z","flags","offset","text"]`、`ASR_FIELDS = ["cer","transcript"]`、定数 `FRAME_SECONDS = 0.02`, `SILENCE_DB = -40.0`, `TAIL_MAX_SECONDS = 0.35`, `TAIL_GAP_SECONDS = 0.4`, `CLIP_LEVEL = 0.99`（設定には入れない）。`AudioStats(seconds, leading_silence, trailing_silence, tail_noise, clipping)`、`CheckRow(chapter, para, seg, key, text, chars, stats, z, offset, flags, cer=None, transcript=None)`、`strip_emoji`、`count_chars`（`strip_emoji(tts_text(text))` から空白を除いた長さ）、`frame_rms_db`、`speech_runs`、`analyze_audio`、`analyze_file`（無ければ None、ステレオは平均）、`robust_z`（中央値 / MAD×1.4826、MAD 0 なら 0、None は飛ばす）、`flags_for(row, z_threshold, cer_threshold, warned)`、`segment_key(synth, row)`、`read_warned_keys`（各行の最初のタブまで）、`load_times_json`、`offset_for(times, para, seg)`（`segments` 優先、無ければ `paragraphs` の段落開始）、`normalize_for_cer`（NFKC → 絵文字・句読点・記号・空白を除く）、`cer(ref, hyp)`（編集距離 / len(ref)、ref 空なら 0）、`transcribe(paths, model, device, batch_size=8)`（transformers は関数内 import、`pipeline("automatic-speech-recognition", ...)`、`generate_kwargs={"language":"ja","task":"transcribe"}`、終わったら `del` と `empty_cache`）、`transcribe_with(check_cfg)`、`write_check_tsv(path, rows, asr)`。
- `tail_noise` の定義: 20 ms フレームの RMS dBFS で `SILENCE_DB` 超を発話とし、発話区間の最後の区間が `TAIL_MAX_SECONDS` 以下で、その直前の無音が `TAIL_GAP_SECONDS` 以上なら、その区間の秒数。それ以外は 0（「最後の 0.3 秒の RMS 比」は発話が末尾まで続くだけで鳴るので採らない）。`leading_silence` / `trailing_silence` は最初・最後の発話フレームまでの秒数。`clipping` は `|x| >= CLIP_LEVEL` のサンプル数。
- 印: `long`（z ≥ 閾値）/ `short`（z ≤ −閾値）/ `clip` / `tail` / `missing` / `warned`（`warnings.log` にキーがある）/ `cer`（`--asr` 時、`cer >= cer_threshold`）。`chars == 0` の行は z を None。
- `check.tsv`: `chapter / para / seg` は 0 始まり、`seconds` `.2f`、`sec_per_char` `.3f`、`z` `.2f`、`offset` `.2f`、None は空欄、`flags` は `|` 区切り。`--asr` のときだけ `cer` / `transcript` 列。ASR は全行（`--chapters` で量を絞る）。
- `pipeline.check_book(cfg, out_dir, *, chapters=None, asr=False, progress=None, transcriber=None) -> CheckResult(path, rows, total, flagged)`: `load_book` → 章で絞り `kind == "break"` を除く → `cfg.voice` が無ければ `ValueError("config の voice か --voice で声を指定してください")` → Synthesizer → `warned` → 行ごとに key / stats / chars → `robust_z` → 章ごとに `.times.json` から `offset` → `asr` なら `(transcriber or check.transcribe_with(cfg.check))` に存在する WAV をまとめて渡し `cer` → `flags` → `write_check_tsv`。`from . import check` は pipeline の先頭で可。
- `CheckConfig(z_threshold=3.0, cer_threshold=0.3, asr_model="kotoba-tech/kotoba-whisper-v2.0", asr_device="cuda")` を `Config.check` / `_SECTIONS["check"]`、`config.example.yaml` は `bgm:` の後。
- `cli check`: `common(p)` + `--asr` + `--voice` + `--checkpoint`（`synth_opts` から `--force` を除いた形に分ける）。`_run_check`: `--checkpoint` を `replace`、`voice = args.voice or cfg.voice` を反映 → tqdm で `check_book` → `ImportError` は `ASR_MISSING` を出して 2、`ValueError` / `RuntimeError` は 1 → `print(f"{total} 文中 {flagged} 件に印（{path}）")` → 0（印があっても 0）。
- WebUI（小説生成タブ、`table` と `script_rows` の直後）: `selected_idx = gr.State(None)`、`gr.Row` に `seg_audio = gr.Audio(type="filepath", label="選んだ文の音声（キャッシュ）", interactive=False)` と `retake_btn = gr.Button("取り直し（選んだ文を別の seed で合成）")` / `check_btn = gr.Button("チェック（長さ・無音・クリッピング）")`、`check_table = gr.Dataframe(headers=CHECK_VIEW_FIELDS, datatype="str", type="array", interactive=False)`（`CHECK_VIEW_FIELDS = ["chapter","para","seg","flags","seconds","chars","z","text"]`、印のある行だけ）。純粋ヘルパー `next_seed(current, default)`（空 → default + 1、default None → `random.randrange(2**31)`、"5" → 6）、`check_view_rows(rows)`。`dataframe_rows` は `r.get(f, "")`、`static_columns` は `text` と `seed` 以外、`rows_from_dataframe` は `seed` を `"" if _blank(seed) else str(_to_int(seed))`（整数でなければ `ValueError`）。
- イベント: `table.select(select_row, [table, script_rows, novel_file, novel_text, *tts_inputs], [selected_idx, seg_audio, novel_status])`、`retake_btn.click(retake_row, [selected_idx, table, script_rows, novel_file, novel_text, *tts_inputs], [table, script_rows, seg_audio, novel_status], **gpu)`、`check_btn.click(run_check, [novel_file, novel_text, chapters, *tts_inputs, table, script_rows], [check_table, novel_status])`。
- `select_row(df, original, novel_file, novel_text, checkpoint, device, precision, voice, evt: gr.SelectData)`: `idx = evt.index[0]`（空・None なら `gr.update()` ×3）。`rows_from_dataframe(df, original or _existing_rows(out_dir))[idx]` で表の今の text / seed を使う。break 行は「区切り行には音声がありません」。`engine.get(tts_key(...), out_dir/"segments", out_dir/"warnings.log")` で Synthesizer を作るだけ（モデルは読まない）→ `segments_dir / f"{check.segment_key(synth, row)}.wav"` → `(idx, path or None, 状態文)`。
- `retake_row(idx, df, original, novel_file, novel_text, checkpoint, device, precision, voice)`: generator。選択なし・break 行は `gr.Error`。`new_seed = next_seed(row["seed"], engine.cfg_for(key).sampling.seed)` → `ensure_loaded` の状態を流す → `path = synth.synthesize(row["text"], force=False, seed=new_seed)` → `rows[idx] = {**row, "seed": str(new_seed)}` → `gr.update(value=dataframe_rows(rows)), rows, str(path), "seed=N で合成しました。「音声にする」で章に反映します"` + 警告。script.tsv は保存しない。
- `run_check(novel_file, novel_text, chapters, checkpoint, device, precision, voice, df, original)`: 表が空でなければ保存してから `check_book(engine.cfg_for(key), out_dir, chapters=parse_chapter_range(chapters))`（ASR なし・GPU なし）→ `(check_view_rows(result.rows), "N 文中 M 件に印（check.tsv）")`。
- README: `## BGM を付ける` の後に `## 品質チェックと取り直し（check）`（コマンド、列と印の意味、`--asr` の前提と漢字/かなの表記差で CER が上がるので閾値は緩め、`offset` で章 WAV の位置に飛べる、`seed` 列を直して `synth` すると直した文だけ作り直し、WebUI の手順）、`### script.tsv を手直しする` に `seed` 行（空なら `sampling.seed`、文のキャッシュキーに入る）、CLI 表に `check`（`--asr` / `--voice` / `--checkpoint`）、`## 設定` に `check.*`、`### タブ「小説生成」` に「表の行をクリック → その文の音声 → 取り直し → 音声にする」と「チェック」。

### テスト

新しいテストは `tests/test_check.py` / `tests/test_pipeline_seed.py` / `tests/test_webui_retake.py`（`_handlers` と同じ取り出し方。声は `"none"` にして `resolve_voice` を避ける）。既存の修正は `tests/test_webui.py` の `_row` ヘルパーに `"seed": ""` と 7 列 → 8 列、`tests/test_pipeline.py` の `FakeSynth.synthesize` と `Failing` 2 箇所に `seed=None`、`tests/test_synth.py` に 3 件（`cache_key` は seed の上書きで変わる / `synthesize` は seed をキーと request に使う / seed None は config の seed のまま）、`tests/test_config.py` に `check` 節 1 件。
check: `count_chars`、`robust_z`（MAD 0、None）、前後の無音と秒数（無音 0.5 s + 正弦波 1 s + 無音 0.2 s）、末尾の孤立音は `tail` / 末尾まで続く発話は `tail` でない、クリッピング、無いファイルで None、`flags_for` 7 種、`segment_key` が `cache_key` と一致し seed 列を使う、`read_warned_keys`、`offset_for` の両経路（`segments` あり / なしを手書き JSON で）、`normalize_for_cer` / `cer`（同一 0・全部違い 1・空参照 0）、TSV の列は `asr` で増える。
seed: `TSV_FIELDS` 末尾、古いファイルを読める、`row_seed`、手直しした seed が `script` 再実行で保たれる、段落の一部が変わっても一致する行の seed は保たれる、新しい文は空、`synthesize_book` が行の seed を渡す（FakeSynth が `(text, seed)` を記録）、署名は seed があるときだけ変わる。`check_book`: 全行・`missing`・`warned`・break を飛ばす（手書きの行）・`--chapters`・偽 transcriber で `cer` と `transcript`・`voice` 無しで `ValueError`。CLI: `script` が無ければ 1、要約を表示して 0、`--asr` で transformers が無ければ 2（`check.transcribe_with` を ImportError に）、`--voice` と `--checkpoint` を受け取る。WebUI: `rows_from_dataframe` の seed と整数でない seed の拒否、`dataframe_rows` は seed の無い行でも動く、`static_columns`、`next_seed`、`select_row` 3 通り（`evt = types.SimpleNamespace(index=[1, 0])`）、`retake_row` の seed 増加と表・State の更新（`Engine.load` を偽 synth に差し替え `synthesize` の `seed` 引数を検証）、選択なしで `Error`、`run_check` は印の付いた行だけ表に出す。

### リスク

gradio `Dataframe.select` の `evt.index` は `[row, col]` で、ヘッダー操作や空表では空になり得る。表を丸ごと置き換えるとスクロール位置と選択が消える（`selected_idx` は State に残る）。`datatype="str"` でも `"43.0"` が来る → `_to_int`。CER は漢字/かなの表記差で高めに出る（閾値 0.3 は緩め）。`transformers` 5.x の ASR pipeline の引数差は関数内 import なので Mac のテストに影響しない。`tail` の定数は実データで調整が要る。

## マージ後の最終形（コーディネーターが解消する箇所）

`synthesize_book` の行ループ:
```python
for r in _sorted_rows(chapter_rows):
    if should_stop and should_stop(): ...
    if r.get("kind") == "break":                                   # C
        clips.append(Clip(None, ch, para, seg, kind="break")); continue
    path = synth.synthesize(spoken_text(r), force=force, seed=row_seed(r))   # B
    clips.append(Clip(path, ch, para, seg, kind=r.get("kind", "narration"), text=spoken_text(r)))  # C
    done += 1
```
`total` は break を除いて数える。`rows_from_dataframe`: break 行は `text=""`、`seed` は `_to_int`。`_row_from_saved` が `seed` を渡す。`cache_key(cfg, text, latent_key, seed=None, lora_key=None)`（B と D の 1 行ずつ）。`chapter_signature` の payload は `latent_key` の後に D の `*_synth_identity(synth)`、末尾に B の seed 列（あるときだけ）。

- `check.segment_key` の中身を D の `synth.segment_key(text, seed=row_seed(row))`（読み辞書と `lora_key` 込み）に替える。
- B の `select_row` / `retake_row` / `run_check` の inputs 末尾に D の `lora` を足し、`engine.get(...)` / `ensure_loaded(...)` に `out_dir=` を渡す（無いと取り直しがアダプタと辞書なしで合成し、「音声にする」が引かないキーで segment を書く）。
- CLAUDE.md の更新（サブコマンド一覧・script.tsv の列・キャッシュの項目・新モジュール）。

## 検証

- Mac: `uv run pytest -q` が全部通る。`uv run python -m irodori_speaker script <＊ 行入りの小説> -c config.example.yaml --no-emotion` で title 行と break 行（`text` 空）が出る。`tests/test_export.py` の ffmpeg 付きテストで m4b の章マーカーが読める。
- GPU 機（ユーザー）: `synth` → `export`（m4b を Mac にダウンロードして AirDrop → Books で章が出る）、`check`（印の妥当性、`--asr` のモデル取得）、`--lora-adapter`、`readings.yaml`。章題と区切りの間を聴いて `silence.*` の既定を調整。

## 見送り・後続

- 登場人物ごとの声（提案 6）、`bgm` の WebUI 化、ダッキング、効果音は対象外。
- `check` の `tail` の定数は実データで調整が要る（設定には入れず定数のまま）。
- `export` を `run` の最後に自動で行う案は、要望があれば後で。
