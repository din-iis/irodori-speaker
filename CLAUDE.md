# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

小説テキストを Irodori-TTS で読み上げる Python パッケージ `irodori_speaker`。感情は Ollama（既定）で句ごとに推定し、Irodori-TTS が解釈する絵文字として本文に挿入する。ユーザー向けの使い方は README.md（日本語）、設計は `docs/superpowers/specs/2026-10-24-irodori-speaker-design.md` にある。

## Commands

開発マシン（Mac）には torch / `irodori_tts` / GPU が無い。テストと `script` ステップはモデル無しで動くように作ってある。

```bash
# 依存関係とテスト（全体 / 1 ファイル / 1 件）
uv sync --extra dev
uv run pytest -q
uv run pytest tests/test_emotion.py -q
uv run pytest tests/test_pipeline.py::test_parse_chapter_range -q

# LLM 無しで分割と script.tsv の出力のみ（out/sample_novel/ に script.tsv と chapters.json）
uv run python -m irodori_speaker script tests/sample_novel.txt -c config.example.yaml --no-emotion

# WebUI の画面確認（生成には torch + irodori_tts + GPU が必要）
uv run --with gradio python -m irodori_speaker.webui
```

実際の合成は GPU マシン（WSL2）で行う。`--frozen` で Mac と同じ `uv.lock` を使い、以後は `--no-sync` で実行する。FFmpeg が必要（torchcodec）。

```bash
uv sync --frozen --extra tts --extra cu128 --extra webui
uv run --no-sync python -m irodori_speaker script novel.txt -c config.yaml     # ステップ 1（Ollama が要る）
uv run --no-sync python -m irodori_speaker synth  novel.txt -c config.yaml     # ステップ 2（TTS だけ）
uv run --no-sync python -m irodori_speaker run    novel.txt -c config.yaml     # 一括
uv run --no-sync python -m irodori_speaker.webui -c config.yaml --host 0.0.0.0
uv run --no-sync python -m irodori_speaker bgm    novel.txt -c config.yaml     # ステップ 3（synth 済みに環境音と音楽を重ねる。Ollama・Freesound のキー・CLAP が要る）
# キャプション（上流の SamplingRequest.caption）で感情を指示できるかの実験。本体の合成はまだキャプションを使わない
uv run --no-sync python scripts/caption_test.py -c config.yaml --checkpoint <A> --checkpoint <B>
```

サブコマンドは `script` / `synth` / `run` / `bgm`。`script` は `--chapters --no-emotion --reannotate --llm-model --think`、`synth` は `--chapters --force --voice --checkpoint`、`run` はその両方を取る。`bgm` は `--chapters --replan --llm-model --think`（`run` には入れない）。lint / formatter の設定は無い。`uv.lock` の `irodori-tts` を更新するときは `uv lock --upgrade-package irodori-tts`。

## Architecture

2 つのステップ（と、その後に独立して走る `bgm`）に分かれ、`cli.py` と `webui.py` は `pipeline.py` の同じ関数（`build_script` / `synthesize_book`）を呼ぶ別入口。ステップ 2 は `script.tsv` と `chapters.json` だけを読み、LLM に触れない。

1. `text_splitter.py`: 青空文庫のルビ・注記を読みに置換 → 章（正規表現）→ 段落（改行）→ 文（。！？、括弧の内側では切らない、1 文は `split.max_chars`=40 まで。長い文は後半で参照音声の声から離れるため短くしている）。`Segment(chapter_idx, para_idx, seg_idx, kind, text)` が全工程のキー。
2. `emotion.py`: `split_clauses` が文を句（`、。！？!?…‥，,`）に分け、**段落単位**で LLM を呼んで句ごとに感情 10 種 × 強度 1〜3 を得る。出力は `validate()` で検証し、不正値・欠落は neutral にフォールバック。成功結果だけ `annotations.jsonl` にキャッシュする（キーはモデル・段落・前後の文脈・`think`）。`apply_shift` で地の文の強度を下げ、`build_spoken_text` が `emotion_emoji` の表から絵文字を句の先頭（括弧で始まる句は括弧の直後）に置く。同じ文の中で直前の句と同じ絵文字は置かない。`LLMAnnotator` は `llm.api` で 2 系統（`ollama` / `openai`）、`LLMServer` が `server_command` のプロセスを起動・停止する。
3. `pipeline.annotate_chapters`: 結果を `out/<book>/script.tsv` に書く。LLM の起動・停止と `unload()` はこの関数の中で完結し、ステップ 2 に LLM の状態を持ち込まない。`build_script` が `chapters.json` も書く。
- `emoji.py`: Irodori-TTS が解釈する絵文字 45 種（`IRODORI_EMOJI`）。`emoji_runs` で本文中の連なりを判定し、`safe_cut` で文字数での分割が連なりの途中を切らないようにする。`webui.carry_emoji` と `text_splitter` が使う。
4. `voices.py`: `voices/<id>.<ext>` か `voices/<id>/`（複数クリップは名前順）を解決し、`voice_key` はファイル名と中身の sha1（mtime は使わない）。予約語 `NO_REF_VOICE = "none"`（`is_no_ref`）は参照音声なしを表し、`resolve_voice` を通さない。
5. `synth.py`: `check_vram` → `load_runtime` → `encode_voice` → `Synthesizer.synthesize`。文ごとに `segments/<sha1>.wav` に保存する。`voice_paths` が空なら参照音声なしで、`latent_key` は `codec_repo` だけを含む `no_ref_latent_key`、`latent_path` は None（encode しない）、`_request` は `ref_latent=None, no_ref=True` を渡す。
6. `pipeline.synthesize_book` + `assemble.py`: 文間・段落間・章頭の無音を挟んで章 WAV（`chapters/NN_タイトル.wav`）、続けて全体 WAV を書く。サンプリングレートは固定せず、キャッシュ WAV から読む。

### script.tsv

列は `chapter, para, seg, kind, labels, text, source`。`source` は分割直後の原文（一致判定用で編集しない）、`labels` は句ごとの `感情:強度` を `|` で並べた参考情報、`text` が絵文字入りの読み上げ本文で、合成とキャッシュキーに使われる。

**手直しが正**。`source` が小説本文と一致する行は、保存済みの `labels` / `text` をそのまま使い（絵文字を作り直さず、shift も再適用しない）、段落の全行が一致すれば LLM も呼ばない。`--reannotate` だけが手直しを捨てて作り直す（`emotion_emoji` を変えた後に使う）。そのとき LLM を呼ぶのは `annotations.jsonl` に無い段落だけ。`--chapters` で一部だけ処理しても、他の章の行は残す。小説から消えた章の行・文は落とす。

### キャッシュ

- 文のキャッシュ（`synth.cache_key`）: `text`（絵文字込み）・参照音声の潜在キー・checkpoint・`sampling` の各値のハッシュ。`device` / `precision` は含めない。設定を変えると自動で作り直しになり、中断後の再実行は続きから再開する。
- 潜在キャッシュ（`out/.voices/<latent_key>.pt`）: 参照音声を DACVAE で一度だけエンコードして保存し、以後は `ref_latent` で渡す。キーは参照音声の中身・`codec_repo`・`ref_normalize_db`・`max_ref_seconds`（`null` のときは checkpoint）。
- 章 WAV は `.sig` に署名（本文・サンプリング・無音・checkpoint・潜在キー）を持ち、全体 WAV は全章が揃って署名が一致するときだけ作る。章 WAV と全体 WAV の書き込みは一時ファイル経由で、失敗時は .sig が消えて stale 扱いになる。止めたとき・失敗時は古い全体 WAV を消す。

`webui.py`（Gradio、`-c` / `--host` / `--port`、既定 `127.0.0.1:7860`）は別の入口で、`Engine` が (checkpoint, device, precision, voice) の組ごとに `Synthesizer` を作り直し、変わると runtime を読み直す。生成ハンドラは generator で状態メッセージを yield する。

- 「台本を作る」「一括実行」の台本部分と「LLM で絵文字を付ける」は、LLM を動かす前に TTS の runtime を GPU から解放する（8GB で両方を載せない）。TTS は次の生成で読み直される。GPU を使う操作は `concurrency_id="gpu"`（上限 1）で直列にする。
- 「音声にする」は表の編集を `script.tsv` に保存してから `load_book` → `check_vram` → `synthesize_book` を回す。表が空なら `script.tsv` をそのまま使い、無ければ案内して止まる。表で編集できるのは `text` 列だけで、他の列は読み込み時の値を使う（`rows_from_dataframe`）。「一括実行」の合成は、直前に作った `script.tsv` をそのまま使う。
- 試聴の「生成」「総当たり生成」は、本文を `split_preview_text`（小説生成と同じ `split.max_chars` の分割）で文に分け、文ごとに `generate_once` を呼んで `join_audio` で `silence` の無音を挟んで結合する（長い文は後半で声が崩れるため）。
- 「停止」は `should_stop` のフラグを立てるだけで、現在の文が終わり次第止まる。文ごとのキャッシュがあるので、再実行で続きから進む。
- 「声を追加」は `voices/<ID><元の拡張子>` にコピーする。既存の ID は上書きせず拒否する。
- 絵文字パレット（`irodori_tts.gradio_emoji_palette`）は import できるときだけ使い、できなければ感情プリセットの Dropdown と「絵文字を文頭に挿入」ボタンにする。「総当たり生成」は同じ seed で文頭の絵文字だけを変えて 10 感情を生成する。

### bgm ステップ

`bgm` は `synth` の後に走る独立した工程で、`script.tsv` / `chapters.json` / 章 WAV と `.times.json` だけを読み、朗読の WAV を書き換えない。入口は `cli._run_bgm` → `pipeline.build_bgm`（`BgmDeps` で LLM・Freesound・CLAP・サーバー・ダウンロードを差し替える。`bgm_*` / `emotion` は関数の中で遅延 import し、`pipeline.py` のトップレベルに置かない）。設計は `docs/superpowers/specs/2026-10-31-bgm-design.md`。

1. `build_bgm` の順序: `load_book` → 対象の章すべてで `bgm_mix.load_times`（無ければ `synth --chapters N` を案内して `ValueError`、LLM を起動する前）→ `freesound.load_key`（LLM の前。`KeyMissing`）→ `with LLMServer`: `bgm_scenes`（`ScenePlanner.plan_chapter` → `merge_short` → `scenes.json`）、`bgm_select.search_slots` → `Gate` / `gate_slots` → `gate.tsv`、`unload` → **LLM を解放してから** `ClapScorer` を作り `choose` → `selection.json` / `missing.tsv` → 章ごとに `build_layout` → `render` → `write_mp3` → `CREDITS.txt` → 通し。
2. 8GB の GPU に LLM と CLAP を同時に載せない。`ClapScorer` は LLM の `unload` の後に作り、選び終えたら捨てる。torch / transformers は `ClapScorer` の中でだけ import する。
3. キャッシュ（`bgm/cache/`）: `scenes.jsonl`（場面）・`search.json`（検索語と条件ごと。上限で止まった後の再実行で API を呼ばない）・`gate.jsonl`（ふるい分け）・`scores.json`（CLAP の点）。`--replan` は `scenes.jsonl` だけ消す。章の MP3 は `<mp3>.sig`（朗読の署名・場面・選んだ音の ID・`bgm` 設定の `mix_signature`）が一致すれば重ね直さず、重ね直すときは選んだ音だけ `bgm/tmp/` にダウンロードして、重ね終えたら（例外でも）消す。通しは `--chapters` なしで全章の MP3 が揃ったときだけ作り、それ以外と失敗時は古い通しを消す。`--chapters` で一部だけ処理しても `selection.json` / `scenes.json` の他の章は残す。
4. 1 日の上限（`DailyLimit`）は捕まえずに上げ、MP3 は書かない（`cli` は終了コード 1）。`KeyMissing` / `ImportError` は終了コード 2、`ValueError` / `RuntimeError` は 1。
5. キーは `~/.config/irodori-speaker/secrets.env`（`bgm.secrets_file`）の `FREESOUND_API_KEY=` 行だけに置く。環境変数に置かない。`Authorization` ヘッダーで API ホストにだけ送り、URL・例外・ログ・キャッシュに含めない（試聴 MP3 の取得には付けない）。**実装・テストはこの実ファイルを読まない**（テストは `tmp_path` に偽のファイルを作る）。
6. `exclude.txt`（手で書く Freesound の ID、`bgm` は読むだけ）以外の `scenes.json` / `selection.json` は派生ファイルで、手で直さない。出力は私的な利用に限る（小説の著作権と Freesound のライセンス）。

## 注意点

- 文末の句点・読点（閉じ括弧の直前を含む）は、モデルに渡す前に `synth.tts_text` で外す（句点・読点で終わると文末に余計な音が入るため）。台本の `text` は変えない。加工規則を変えるときは `TEXT_RULE` を更新する（文のキャッシュと章署名に入る）。
- `torch` と `irodori_tts` は `synth.py` / `webui.py` 内（と `cli._make_synthesizer`）で遅延 import している。トップレベルに移すと Mac でのテストと `script` が壊れる。`emotion.py` も `pipeline.annotate_chapters` の中で遅延 import している（`--no-emotion` を動かすため）。TTS の import に失敗したら `cli.py` が案内を出して終了コード 2 で止まる。
- **8GB の GPU では LLM と TTS を同時に載せない**。`run` は台本 → LLM 解放 → 音声化の順。`load_runtime` はモデルを読み込む前に `check_vram`（`min_free_vram_gb`）で空きを確認し、足りなければ LLM サーバーの停止を案内する。モデルを切り替えるときは旧 runtime を先に解放してから確認する（上流は新旧を同時に持つため）。
- LLM の `ollama` は OpenAI 互換ではなく**ネイティブ `/api/chat`** を使う。`think: false` と `format`（JSON Schema）、`keep_alive: 0` が確実に効くため（OpenAI 互換経由では思考モードの無効化が届かない報告がある）。`openai` は `response_format`（400 なら `json_object`、さらに 400 なら余分なキーを外す）と `chat_template_kwargs` / `reasoning_effort` を送り、`<think>` は除去する。
- **絵文字の配置（句の先頭、連発しない、文の先頭は必ず置く）は本ツールの仮定**。上流の `EMOJI_ANNOTATIONS.md` は配置ルールを定めていない。変更時は WebUI の「総当たり生成」で効きを確認する。強度 1 は絵文字なし、強度 3 は同じ絵文字の繰り返し。😭 のような非言語音の絵文字は発話の前に泣き声が入ることがあるので、既定では sad の強度 3 だけに使う。
- 長文は不安定になり、`sampling.max_seconds`（30 秒）で切り詰められるので、文単位で合成して結合する設計を崩さない。切り詰めは `res.messages` の `info: predicted duration …` を `synth.collect_warnings` が読んで `warnings.log` に記録する。
- 参照音声（`ref_latent`）に対応しないチェックポイント（VoiceDesign 等）は対象外で、`load_runtime` が ValueError にする。
- `voice: none` は話者 LoRA 統合 checkpoint 向けの参照音声なし合成（ユーザー決定 2026-10-25）。`null` は none の意味にせず、従来どおりエラーで止める。`voices/none.wav` があっても none は参照音声なしとして扱い、WebUI の「声を追加」は ID `none` を拒否する。声の候補は `webui.voice_choices`（先頭が none）。ベースモデルでは声が定まらない。
- 出力には上流の仕様で SilentCipher の透かしが入る。他人の声を同意なく複製しないこと（モデルカードの倫理規定）。
- `voices/` と `config.yaml`、`out/` は git に入れない（`.gitignore`）。
