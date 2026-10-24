# irodori-speaker: Irodori-TTS で小説を読み上げる CLI + WebUI

## Context

小説のテキストを Irodori-TTS（日本語特化のオープンソース TTS、参照音声によるゼロショット声クローン、本文中の絵文字で演技を制御）で読み上げるツールを新規に作る。リポジトリ `irodori-speaker` は空（`.git` のみ）。

ユーザーには同じ目的の既存プロジェクト `../voxcpm-speaker`（VoxCPM2 + LoRA 版、パッケージ `novel_tts`）があり、その構成を Irodori-TTS 向けに移植する。voxcpm-speaker の文分割・章結合・script.tsv・キャッシュ・WebUI の設計は実績があるので、TTS エンジン部分と感情の与え方だけを差し替える。

### ブレインストーミングで決まったこと（ユーザーの決定）

| 項目 | 決定 |
|---|---|
| 実行環境 | 合成は **GPU マシン（Windows + WSL2、RTX 3060 Ti 8GB）**。Mac（M2/16GB）は開発・テスト・`--dry-run` のみ（voxcpm-speaker と同じ運用） |
| 感情制御 | **ローカル LLM で感情を推定し、対応する絵文字を本文に挿入**する（script.tsv で手直し可能） |
| 絵文字の単位 | **句読点ごと**。文を「、。！？…」で句に分け、句ごとに感情を推定して絵文字を置く（文頭 1 個ではない） |
| LLM エンジン | **Ollama を既定**（導入が最も簡単で、注釈後に VRAM を自動解放できる）。llama-server / vLLM へは設定だけで切り替えられるようにする |
| LLM モデル | **評価（Nejumi Leaderboard 4）を確認して選ぶ**。量子化版の劣化も考慮する → 下記「LLM モデルの選定」 |
| モデルの選択 | **既定だけでなく WebUI の画面で選べる**（LLM はサーバーにあるモデル一覧から、TTS はチェックポイント一覧 + 自由入力） |
| 実行の単位 | **「台本の書き出し（script.tsv）」と「音声化」を独立したステップとして別々に実行できる**。**一括実行**もできる（CLI も WebUI も） |
| 声 | **1 つの参照音声で全部読む**（地の文も台詞も同じ声） |
| WebUI の役割 | **試聴（短文で声・絵文字・パラメータを確認）と、小説全体の生成の両方** |

### 私（Claude）の判断で決めたこと

- **Irodori-TTS はライブラリとして直接組み込む**（`irodori_tts.inference_runtime`）。HTTP サーバー（Irodori-TTS-Server）は使わない。CLI も WebUI も GPU マシンで動かし、WebUI は `--host 0.0.0.0` で Mac のブラウザから開く。
- **感情→絵文字は対応表で変換**する。Ollama の語彙は voxcpm-speaker と同じ 10 感情 × 強度 1〜3 を維持し（検証済みの `validate` / 地の文の強度補正をそのまま使える）、`emotion_emoji` 表で絵文字に変換する。強度 1 は絵文字なし（地の文は強度を 1 下げるので、地の文はほぼ素の朗読になる）。強度 3 は原則「同じ絵文字の繰り返し」（上流の EMOJI_ANNOTATIONS.md に「同じ絵文字を複数回使うと効果を強調できる」とある）。😭 泣き声・😱 悲鳴のような非言語音の絵文字は、発話の前に泣き声や悲鳴が入る可能性があるので、既定では sad の強度 3 だけに使う。表は config で上書き可能で、WebUI の「総当たり」で調整する。
- **句の切り方と絵文字の置き方**。文（合成単位）を「、。！？!?…‥，」の直後で句に分け（句読点は前の句に付ける。括弧だけの断片は前の句に併合）、LLM には段落内の全句を `文番号-句番号: 句` の形で渡して句ごとの感情を得る。絵文字は句の先頭に置く（句が「『（で始まるなら開き括弧の直後）。同じ文の中で直前の句と絵文字が同じなら置かない（連発を避ける）。文の先頭の句は必ず置く（文ごとに別々に合成するため、前の文の絵文字は効かない）。上流の `EMOJI_ANNOTATIONS.md` は配置ルールを定めていないので、これは本ツールの仮定。「総当たり」で効きを確認する。
- **script.tsv は手直しが正**。`source` 列（分割直後の原文）で小説本文との一致を判定し、一致した行は `labels` / `text` を保存済みの値のまま使う（絵文字を作り直さない）。`text` 列は絵文字入りの読み上げ本文で、絵文字の位置変更・削除・追加や読み仮名の修正はここで行う。`labels` 列は句ごとの `感情:強度` を `|` で並べた参考情報。`emotion_emoji` 表を変えた後に反映したいときは `--reannotate`（script.tsv の手直しを捨て、`annotations.jsonl` のキャッシュから絵文字を作り直す。LLM は未キャッシュ段落のみ呼ぶ）。
- **参照音声は一度だけ DACVAE でエンコードして `.pt`（潜在）にキャッシュ**し、以後は `ref_latent` で渡す。文ごとに参照音声を再エンコードしない。
- **既定モデル** `Aratako/Irodori-TTS-v4.1-Small`（RF、40 ステップ）。CUDA では `precision: bf16`。
- **1 文の上限は 80 文字**（voxcpm は 120）。Irodori の `SamplingRequest.max_seconds` 既定が 30 秒なので、長い文が切れないようにする。
- パッケージ名は `irodori_speaker`。Python は上流に合わせ `.python-version` = 3.10。
- **8GB の GPU では LLM と TTS を同時に載せない**。CLI は「注釈（LLM）→ LLM を解放 → TTS 読み込み → 合成」の順に進め、TTS を読み込む前に空き VRAM を確認して、足りなければ LLM サーバーの停止を案内する。
- **LLM の呼び出しは 2 系統**を `llm.api` で切り替える。`ollama`（既定）は Ollama ネイティブ `/api/chat` を使う（`think: false`、`format: <JSON Schema>`、`keep_alive: 0` が確実に効く。OpenAI 互換経由だと思考モードの無効化が届かない報告がある）。`openai` は `/v1/chat/completions` を使い（llama-server / vLLM / LM Studio）、`response_format` の JSON Schema（400 なら `json_object`）と `chat_template_kwargs: {enable_thinking: false}` + `reasoning_effort: "none"` を送る。`server_command` が設定されていればプロセスをツールが起動・停止する。
- **思考モードは既定で無効**（`llm.think: false`）。有効にすると段落あたり数倍遅くなる。精度が足りないときの選択肢として残す。

### LLM モデルの選定（8GB、Nejumi Leaderboard 4 の 2026 年 10 月時点の集計より）

| モデル | 総合スコア | 8GB での量子化 | 備考 |
|---|---|---|---|
| **Qwen/Qwen3.5-4B**（推奨） | **0.7352**（10B 未満で 1 位） | **Q8_0 5.2GB** が載る（劣化ほぼ無し） | Apache 2.0、2026 年 2 月。Ollama `qwen3.5:4b-q8_0` / 軽量 `qwen3.5:4b-q4_K_M`（3.3GB） |
| nvidia/Nemotron-Nano-9B-v2-Japanese | 0.7111 | Q4 のみ（約 6GB） | Mamba ハイブリッド、ライセンスが NVIDIA 独自 |
| llm-jp/llm-jp-4.1-8b-thinking | 0.6921 | Q4 のみ（約 5.5GB） | Apache 2.0、日本語特化 |
| google/gemma-4-E4B-it | 0.6692 | Q8 可 | |
| tokyotech-llm/Qwen3-Swallow-8B-RL-v0.2 | 0.6552 | Q4 のみ | |

- 根拠: Qualiteg の集計（journal.qualiteg.com/llm-ranking-2026/、2026-10-22 スナップショット）と biton の 2026 年 10 月版比較。スコアは思考モード有効時のもので、非思考モードの公表値は無い。
- 量子化の目安: Q8_0 は FP16 とほぼ同等、Q4_K_M は 1〜3% の劣化（Llama-3.1-8B の perplexity: FP16 7.32 / Q8_0 7.33 / Q4_K_M 7.56）。4B は Q8 で 5.2GB なので 8GB に収まり、8B 級を Q4 で使うより有利。
- 既定は `qwen3.5:4b-q8_0`。voxcpm-speaker の `qwen2.5:14b-instruct` は 8GB に載らないので使わない。
- **TTS は `precision: bf16`** を既定にし（Ampere 対応）、OOM なら `Aratako/Irodori-TTS-v4.1-Small-Quantized/int8-weight-only` に切り替える。N 文ごとに `torch.cuda.empty_cache()` を呼ぶ。

### 参考にした一次情報

- Irodori-TTS 本体: https://github.com/Aratako/Irodori-TTS （MIT、`irodori_tts/inference_runtime.py` が公開 API）
- サーバー実装（pyproject の torch index 設定・voices フォルダ規約の参考）: https://github.com/Aratako/Irodori-TTS-Server
- モデルカード: https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small （MIT。参照音声は複数の短いクリップで合計 30 秒程度が推奨、上限 120 秒。日本語のみ。他人の声を同意なく複製しないこと）
- 絵文字一覧: `irodori_tts/gradio_emoji_palette.py` の `EMOJI_PALETTE_ITEMS`（45 種。例: 🤭 笑い、😭 泣き声、😠 怒り、😲 驚き、😰 慌てる、🫶 優しく、👂 囁き、⏩ 早口、🐢 ゆっくり、📖 朗読）

## Irodori-TTS の API（実装で使うもの）

```python
from irodori_tts.inference_runtime import (
    InferenceRuntime, RuntimeKey, SamplingRequest, SamplingResult,
    get_cached_runtime, download_hf_checkpoint, save_wav,
)
# チェックポイント解決: HF id ("Aratako/Irodori-TTS-v4.1-Small" / "repo/subfolder") -> download_hf_checkpoint(id) がローカルパスを返す。
#                        ローカル .safetensors / .pt はそのまま。
key = RuntimeKey(checkpoint=path, model_device="cuda", codec_device="cuda",
                 model_precision="bf16", codec_precision="bf16",
                 codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim")
runtime, loaded_now = get_cached_runtime(key)   # key が変わると旧 runtime を unload する。threading.Lock で保護
runtime.model_cfg.use_speaker_condition_resolved  # True なら参照必須（no_ref=True で無しも可）
runtime.codec.sample_rate                        # 出力サンプリングレート（codec 依存、48 kHz 想定）
latent = runtime.codec.encode_waveform(wav, sr, normalize_db=-16.0, ensure_max=True).cpu()  # (1, T, D)
res: SamplingResult = runtime.synthesize(SamplingRequest(
    text=text, ref_latent="path.pt",   # または ref_latents=[...] / ref_wav / ref_wavs（単複は同時指定不可）
    num_steps=None,                    # None = RF 40 / MeanFlow 4
    cfg_scale_text=3.0, cfg_scale_speaker=5.0, seed=42,
    duration_scale=1.0, max_seconds=30.0, t_schedule_mode="linear", sway_coeff=-1.0,
))
res.audio  # torch.Tensor (1, samples)   res.sample_rate   res.used_seed
save_wav(path, res.audio, res.sample_rate)
```

- `ref_latent` の `.pt` は `torch.save(tensor)` した `(T, D)` または `(1, T, D)`。読み込み側で `max_ref_seconds` 分に切り詰められる（v4: 120 秒）。
- `InferenceRuntime.synthesize` は内部で `_infer_lock` を取る。同時実行は 1 件。

## アーキテクチャ

```
ステップ 1「台本」（LLM だけ使う、torch 不要）
novel.txt ─ text_splitter ─▶ Segment(章, 段落, 文, kind, text) ─ split_clauses ─▶ 句
                │
                ▼
          emotion (LLM, 段落単位・句ごと) ─▶ Annotation ─▶ 絵文字入り本文 ─▶ out/<book>/script.tsv（手直し優先）
                                                                            out/<book>/chapters.json（章タイトル）
ステップ 2「音声化」（TTS だけ使う、LLM 不要。script.tsv と chapters.json だけを読む）
          script.tsv ─▶ synth (Irodori-TTS) ─ 文ごとに out/<book>/segments/<sha1>.wav にキャッシュ
                │                               参照音声は out/.voices/<sha1>.pt に潜在をキャッシュ
                ▼
          assemble ─▶ out/<book>/chapters/NN_タイトル.wav, out/<book>/<book>_full.wav
```

2 つのステップは独立に実行でき、「一括実行」はステップ 1 → LLM 解放 → ステップ 2 を続けて走らせる。`cli.py` と `webui.py` は `pipeline.py` の同じ関数（`build_script` / `synthesize_book`）を呼ぶ別入口。

## ファイル構成

```
irodori-speaker/
├── pyproject.toml            # 下記
├── .python-version           # 3.10
├── .gitignore                # .venv/ out/ __pycache__/ *.egg-info/ config.yaml voices/（voices/.gitkeep は残す）
├── README.md                 # 日本語。セットアップ（GPU マシン / Mac）、使い方、声の置き方、絵文字、注意（透かし・同意）
├── CLAUDE.md                 # voxcpm-speaker の CLAUDE.md と同じ粒度で、コマンドと設計の要点
├── config.example.yaml
├── voices/.gitkeep
├── irodori_speaker/
│   ├── __init__.py
│   ├── __main__.py           # from .cli import main
│   ├── config.py
│   ├── text_splitter.py      # voxcpm-speaker からそのままコピー
│   ├── emotion.py            # voxcpm-speaker から移植。to_instruction → to_emoji / insert_emoji
│   ├── voices.py             # voices/ の列挙・解決・声キー
│   ├── synth.py              # Irodori-TTS 呼び出し・潜在キャッシュ・セグメントキャッシュ
│   ├── assemble.py           # voxcpm-speaker からそのままコピー
│   ├── pipeline.py           # annotate_chapters / synthesize_book（CLI と WebUI の共通部分。進捗コールバック付き）
│   ├── cli.py
│   └── webui.py
└── tests/
    ├── sample_novel.txt      # voxcpm-speaker からコピー
    ├── test_text_splitter.py # コピー
    ├── test_emotion.py       # 句分割・句単位の検証・絵文字挿入・LLM 2 系統（モック）
    ├── test_voices.py
    ├── test_synth.py         # cache_key / 潜在キーの純粋関数のみ
    └── test_pipeline.py      # script.tsv の読み書き・手直し優先（Ollama と TTS はモック）
```

### pyproject.toml（要点）

Irodori-TTS-Server の pyproject を手本にする。`irodori-tts` は PyPI に無いので git 依存。torch の index は extra で切り替える（依存先の `[tool.uv.sources]` は引き継がれないため自前で宣言する）。

**確認済みの落とし穴**: Irodori-TTS は `dacvae` を素の名前で依存に書き、自分の `[tool.uv.sources]` で `git = "https://github.com/facebookresearch/dacvae"` を指している。`dacvae` は PyPI に存在しない（404）。依存先の source が推移的に効くかは uv の版に依存するので、こちらの `tts` extra で直接 URL として明示する（害はない）。`silentcipher` は上流が PEP 508 の直接 URL（コミット固定）で書いているので推移的に解決できる。

```toml
[project]
name = "irodori-speaker"
requires-python = ">=3.10,<3.13"
dependencies = ["numpy", "soundfile", "requests", "pyyaml", "tqdm"]

[project.optional-dependencies]
tts = ["irodori-tts @ git+https://github.com/Aratako/Irodori-TTS.git",
       "dacvae @ git+https://github.com/facebookresearch/dacvae",
       "torch>=2.10.0,<2.11.0", "torchaudio>=2.10.0,<2.11.0"]
cu128 = ["torch>=2.10.0,<2.11.0", "torchaudio>=2.10.0,<2.11.0", "torchao>=0.16.0,<0.17.0"]
webui = ["gradio>=5.0.0"]
dev = ["pytest"]

[tool.uv.sources]
torch = [{ index = "pytorch-cu128", extra = "cu128", marker = "sys_platform == 'linux' or sys_platform == 'win32'" }]
torchaudio = [{ index = "pytorch-cu128", extra = "cu128", marker = "sys_platform == 'linux' or sys_platform == 'win32'" }]

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true
```

- GPU マシン: `uv sync --frozen --extra tts --extra cu128 --extra webui`（`--frozen` で Mac と同じ `uv.lock` のコミットを使う）、以後 `uv run --no-sync ...`。FFmpeg が必要（torchcodec）。
- Mac: `uv sync --extra dev`（torch を入れない。`irodori_tts` は `synth.py` / `webui.py` 内で遅延 import する）。
- `uv.lock` は Mac で作り、`irodori-tts` の git コミットを固定する。上流を更新したいときは `uv lock --upgrade-package irodori-tts`。

### config.example.yaml

```yaml
checkpoint: Aratako/Irodori-TTS-v4.1-Small   # HF id / HF id/サブフォルダ / ローカル .safetensors
# 8GB で OOM になる場合: Aratako/Irodori-TTS-v4.1-Small-Quantized/int8-weight-only
codec_repo: Aratako/Semantic-DACVAE-Japanese-32dim
device: cuda            # cuda | mps | cpu（モデルとコーデックの両方）
precision: bf16         # fp32 | bf16（CUDA は bf16 が速く VRAM も半分。mps/cpu は fp32）
min_free_vram_gb: 3.0   # TTS 読み込み前の空き VRAM チェック（0 で無効）
empty_cache_interval: 10  # N 文ごとに torch.cuda.empty_cache()（0 で無効）

voices_dir: voices
voice: narrator         # voices/narrator.wav か voices/narrator/ 内の複数クリップ（名前順に連結、合計 30 秒程度が推奨）
ref_normalize_db: -16.0 # null で無効
max_ref_seconds: null   # null = チェックポイント既定（v4 は 120 秒）

sampling:
  num_steps: null       # null = RF 40 / MeanFlow 4
  t_schedule_mode: linear   # linear | sway
  sway_coeff: -1.0
  cfg_scale_text: 3.0
  cfg_scale_speaker: 5.0
  duration_scale: 1.0
  max_seconds: 30.0     # 1 文の最大長。split.max_chars と合わせる
  seed: 42              # null でランダム（文ごとに声がぶれるので固定推奨）

emotion_mode: emoji     # emoji | none
narration_intensity_shift: 1
# 感情 × 強度(1/2/3) → 挿入する絵文字。空文字は挿入なし。Irodori の対応絵文字のみ効く
emotion_emoji:
  neutral:   ["", "", ""]
  happy:     ["", "😊", "😆"]
  sad:       ["", "😟", "😭"]
  angry:     ["", "😠", "😠😠"]
  surprised: ["", "😲", "😲😲"]
  fearful:   ["", "😰", "😰😰"]
  tender:    ["", "🫶", "🫶🫶"]
  whisper:   ["", "👂", "👂👂"]
  excited:   ["", "💥", "💥💥"]
  serious:   ["", "", ""]

llm:
  api: ollama           # ollama: Ollama ネイティブ API / openai: OpenAI 互換（llama-server / vLLM / LM Studio）
  base_url: http://localhost:11434      # openai のときは http://localhost:8080/v1 など
  model: qwen3.5:4b-q8_0                # 8GB 向けの推奨。軽くするなら qwen3.5:4b-q4_K_M
  think: false          # 思考モード。true は精度が上がる可能性があるが数倍遅い
  api_key: null
  timeout: 120
  concurrency: 1        # Ollama は OLLAMA_NUM_PARALLEL、llama-server は -np に合わせる
  unload: true          # 注釈後にモデルを解放する（ollama: keep_alive=0 / openai: server_command を終了）
  # server_command:     # openai のとき、指定すると注釈の前に起動して /v1/models が応答するまで待ち、注釈後に終了させる
  #   ["llama-server", "-m", "models/Qwen3.5-4B-Q8_0.gguf", "-ngl", "99", "-c", "16384", "-np", "4", "--port", "8080"]
  server_start_timeout: 300

split:
  # chapter_regex: '^\s*第.+?[章話節].*$'
  max_chars: 80

silence:   # 秒
  sentence: 0.25
  paragraph: 0.6
  chapter: 1.0

output_dir: out
```

## モジュール設計

### `config.py`
voxcpm-speaker の `config.py` の形（dataclass + `_build` で不明キーを弾く）を踏襲。`Config` に `checkpoint, codec_repo, device, precision, min_free_vram_gb, empty_cache_interval, voices_dir, voice, ref_normalize_db, max_ref_seconds, sampling: SamplingConfig, emotion_mode, narration_intensity_shift, emotion_emoji, llm: LLMConfig, split, silence, output_dir`。`DEFAULT_EMOTION_EMOJI` は `emotion.py` 側に置き、config はそれを既定値にする。

### `text_splitter.py` / `assemble.py`
`../voxcpm-speaker/novel_tts/` から変更なしでコピー（`split_sentences` の `max_chars` 既定は呼び出し側で 80 を渡す）。

### `emotion.py`
voxcpm-speaker から `EMOTIONS`, `Annotation`, `validate` の考え方, `paragraph_hash`, `apply_shift` を移植し、句単位に拡張する。`OllamaAnnotator` は `LLMAnnotator` に、`DEFAULT_INSTRUCTIONS` / `to_instruction` は絵文字表に置き換える:

```python
CLAUSE_END = "、。！？!?…‥，,"
def split_clauses(sentence: str) -> list[str]
    # CLAUSE_END の直後で区切る（句読点は前の句に付ける）。「」』）だけになる断片は前の句に併合。"".join(clauses) == sentence を保証

SYSTEM_PROMPT  # voxcpm の文面を句単位に変更: 対象は「文番号-句番号: 句」の一覧。出力 {"segments":[{"s":0,"c":0,"emotion":"neutral","intensity":1}]}
JSON_SCHEMA    # 上の形を JSON Schema で（emotion は EMOTIONS の enum、intensity は 1..3 の integer）
def build_user_prompt(prev: list[str] | None, target: list[list[str]], nxt: list[str] | None) -> str
def validate(data, shape: list[int]) -> list[list[Annotation]]   # 文ごとの句数 shape に合わせて検証。欠落・不正は neutral

DEFAULT_EMOTION_EMOJI: dict[str, list[str]]  # 上の表
def to_emoji(ann: Annotation, table=None) -> str
def build_spoken_text(clauses: list[str], anns: list[Annotation], table=None) -> str
    # 句 k の先頭に to_emoji(anns[k]) を置く。句が「『（で始まるなら括弧の直後。k>0 で直前の句と同じ絵文字なら置かない
def labels_to_str(anns) -> str  # "happy:2|neutral:1"

class LLMAnnotator:
    # annotate(target: list[Segment], prev, nxt) -> list[list[Annotation]]（文ごと・句ごと）。annotations.jsonl キャッシュ（成功のみ、Lock）
    # annotate_many(items) が ThreadPoolExecutor(concurrency) で段落を並列に処理し、段落順に返す
    # api == "ollama": POST {base_url}/api/chat {"model", "messages", "think": cfg.think, "format": JSON_SCHEMA, "stream": false,
    #                  "options": {"temperature": 0}, "keep_alive": ...}。message.content を json.loads
    # api == "openai": POST {base_url}/chat/completions {"model", "messages", "temperature": 0,
    #                  "response_format": {"type": "json_schema", "json_schema": {...}}, "chat_template_kwargs": {"enable_thinking": cfg.think},
    #                  "reasoning_effort": "none"（think=false のとき）}。HTTP 400 なら response_format を json_object に、
    #                  それでも 400 なら余分なキーを外して再試行。choices[0].message.content を json.loads
    #                  （content が <think>…</think> を含むときは除去してから）
    def unload(self) -> None
        # unload=true かつ api == "ollama": POST {base_url}/api/generate {"model": model, "keep_alive": 0}。失敗は警告のみ
        # api == "openai" は LLMServer の __exit__ に任せる

class LLMServer:  # context manager
    # llm.server_command が設定されていて GET {base_url}/models が応答しないときだけ subprocess.Popen で起動し、
    # 応答するまで server_start_timeout 秒待つ。__exit__ で terminate → 10 秒待って kill。自分で起動していなければ何もしない
```

### `voices.py`

```python
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".aac", ".webm"}
def list_voices(voices_dir: Path) -> list[str]          # <id>.<ext> と <id>/（音声を含むディレクトリ）の ID を名前順に
def resolve_voice(voices_dir: Path, voice: str) -> list[Path]
    # ID のほか、ファイル/ディレクトリのパス指定も受ける。ディレクトリは名前順に並べる。見つからなければ ValueError（候補を列挙）
def voice_key(paths: list[Path]) -> str                  # ファイル名と中身（バイト列）の sha1。mtime は使わない（scp/rsync で変わるとキャッシュが全滅するため）
```

### `synth.py`

```python
def check_vram(cfg: Config) -> None
    # device が cuda で min_free_vram_gb > 0 のとき、torch.cuda.mem_get_info() の空きが不足なら
    # RuntimeError("空き VRAM が X GB です。LLM サーバー（Ollama なら `ollama stop <model>`）を停止してから再実行してください")

def latent_key(voice_key: str, codec_repo: str, normalize_db: float | None, max_ref_seconds: float | None) -> str
    # 潜在 .pt の識別子（sha1）。参照音声の中身・コーデック・正規化・上限秒数のどれかが変われば別ファイル

def cache_key(cfg: Config, text: str, latent_key: str) -> str
    # text(絵文字込み), latent_key, checkpoint, num_steps, t_schedule_mode, sway_coeff, cfg_scale_text,
    # cfg_scale_speaker, duration_scale, max_seconds, seed を sha1。device / precision は含めない

def load_runtime(cfg: Config):                           # irodori_tts を遅延 import
    # checkpoint が存在するパスならそのまま、そうでなければ download_hf_checkpoint()。
    # RuntimeKey(model_device=cfg.device, codec_device=cfg.device, model_precision=cfg.precision, codec_precision=cfg.precision, codec_repo=...)
    # get_cached_runtime(key) を返す（WebUI でモデル切替時に旧 runtime を自動 unload）

def encode_voice(runtime, paths: list[Path], cache_dir: Path, cfg) -> Path
    # cache_dir/<latent_key>.pt があれば返す。
    # 無ければクリップごとに読み込み（torchaudio.load、失敗時 soundfile）→ クリップが 1 本だけで max_ref_seconds を超えるときは
    # 波形を先に切り詰める（runtime は潜在を後から切るだけなので、10 分のファイルを丸ごとエンコードしない）→
    # runtime.codec.encode_waveform(wav, sr, normalize_db=cfg.ref_normalize_db, ensure_max=True).cpu()（上流で inference_mode 済み）→
    # dim=1 で連結 → tmp に torch.save(latent[0]) → rename

class Synthesizer:
    def __init__(self, cfg, segments_dir: Path, voice_paths: list[Path], latent_cache_dir: Path, warnings_path: Path | None)
    runtime (lazy)        # check_vram → load_runtime
    latent_path (lazy)    # encode_voice
    def synthesize(self, text: str, force=False) -> Path
        # segments_dir/<cache_key>.wav があれば返す。無ければ SamplingRequest(text, ref_latent=str(latent_path), ...) →
        # runtime.synthesize → save_wav(tmp) → tmp.replace(path)。empty_cache_interval 文ごとに torch.cuda.empty_cache()。
        # 警告の検知: res.messages のうち "info:" で始まらないもの（テキストのトークン超過、参照の切り詰め等）に加え、
        # 長さの上限は runtime が黙って切り詰める（messages に "info: predicted duration frames=X, scale=S, using_frames=Y" が残るだけ）ので、
        # この行を正規表現で読み round(X*S) > Y なら「max_seconds で切り詰め」の警告にする。
        # 警告は warnings_path に「cache_key \t text \t message」で追記し、直近の警告を self.last_warnings に保持（WebUI の状態欄に出す）
    def generate_once(self, text: str, **overrides) -> tuple[int, np.ndarray, list[str]]   # WebUI 試聴用（キャッシュしない。警告も返す）
```

`runtime.model_cfg.use_speaker_condition_resolved` が False のチェックポイント（VoiceDesign 等）は本ツールの対象外として起動時に ValueError。

### `pipeline.py`（CLI と WebUI の共通部分）

voxcpm-speaker の `cli.annotate_chapters` と `main` の合成ループをここに移す。

```python
TSV_FIELDS = ["chapter", "para", "seg", "kind", "labels", "text", "source"]
# source = 分割直後の原文（一致判定用、編集しない）。labels = 句ごとの "感情:強度" を | で連結（参考情報）。
# text = 絵文字入りの読み上げ本文（手直しはここ。絵文字の移動・削除・追加、読み仮名の修正）
def load_script / write_script
def build_script(novel: Path, cfg, out_dir, *, chapters: set[int] | None, use_emotion, reannotate=False,
                 llm_overrides: dict | None = None, progress=None) -> list[dict]          # ステップ 1
    # build_chapters → chapters.json（{idx: title}）を書く → annotate_chapters → script.tsv を書く
def annotate_chapters(chapters, cfg, out_dir, use_emotion, *, reannotate=False, progress=None) -> list[dict]
    # 段落内の全行が保存済みで source が一致（かつ reannotate でない）→ labels/text をそのまま使う（LLM も絵文字生成も呼ばない）
    # それ以外 → split_clauses → LLMAnnotator（annotations.jsonl キャッシュ）→ 句ごとに apply_shift（文の kind で）→ build_spoken_text
    # use_emotion=False → labels は全て neutral:1、text = source
    # --chapters で一部だけ処理しても他章の行は残す（voxcpm と同じ）
    # LLMServer の起動・停止と annotator.unload() はこの関数の中で完結させる（ステップ 2 に LLM の状態を持ち込まない）
def load_book(out_dir) -> tuple[dict[int, str], list[dict]]                                # chapters.json と script.tsv を読む。無ければ「先に script を実行」の ValueError
def spoken_text(row) -> str                               # row["text"] をそのまま返す。合成とキャッシュキーに使う
def synthesize_book(rows, titles, cfg, out_dir, synth, *, chapters: set[int] | None = None, force=False,
                    progress: Callable[[int, int, str], None] | None = None,
                    should_stop: Callable[[], bool] | None = None) -> BookResult(chapter_files, full_path, stopped)   # ステップ 2
    # rows を (chapter, para, seg) 順に章ごとにまとめ、文ごとに synth.synthesize(spoken_text(row)) → 章 WAV → 全体 WAV。
    # chapters 指定時はその章だけ合成し、全体 WAV は既存の章 WAV が揃っているときだけ作る。should_stop が True なら章の途中でも止め stopped=True
```

### `cli.py`

サブコマンドで 2 つのステップを分ける。どれも `novel.txt` を受け取り、`out/<novel の stem>/` を作業ディレクトリにする。

```
python -m irodori_speaker script novel.txt -c config.yaml [--chapters 1-3,5] [--no-emotion] [--reannotate] [--llm-model NAME] [--think]
    # ステップ 1: 分割 → LLM で句ごとの感情 → 絵文字入りの script.tsv と chapters.json を書く。torch 不要。終わったら LLM を解放
python -m irodori_speaker synth  novel.txt -c config.yaml [--chapters 1-3,5] [--force] [--voice ID] [--checkpoint ID]
    # ステップ 2: script.tsv と chapters.json を読み、check_vram → TTS 読み込み → 合成 → 章 WAV / 全体 WAV。LLM 不要。novel.txt は名前だけ使う
python -m irodori_speaker run    novel.txt -c config.yaml [script と synth の両方のオプション]
    # 一括: script → synth を続けて実行（間で LLM を解放してから TTS を読む）
```

`--reannotate` は script.tsv の手直しを捨てて感情と絵文字を作り直す（`emotion_emoji` 表の変更を反映するとき）。`script` は先頭で「N 章 / N 段落 / N 文」を、`synth` は使う声（クリップ数・合計秒数）を表示し、終了時に warnings.log の件数を出す。`synth` で script.tsv が無ければ「先に script を実行してください」と案内して終了する。

### `webui.py`（Gradio）

```
python -m irodori_speaker.webui [-c config.yaml] [--host 0.0.0.0] [--port 7860]
```

- **上部「TTS 設定」**: チェックポイント（Dropdown。候補 `Aratako/Irodori-TTS-v4.1-Small` / `…-Small-Quantized/int8-weight-only` / `…-Small-MF` に加えて自由入力可 `allow_custom_value=True`）、device / precision（Dropdown）、声（Dropdown = `list_voices`、更新ボタン、音声アップロード → `voices/<id>.wav` に保存して一覧更新）、「モデル読み込み」、状態欄。`Engine` は voxcpm-speaker の `Engine` と同じく、(checkpoint, device, precision) が変わったら `get_cached_runtime` で読み直す。生成ハンドラは generator で状態メッセージを yield する。
- **上部「LLM 設定」**（折りたたみ）: api（Radio: ollama / openai）、base_url、モデル（Dropdown + 「一覧を取得」ボタン。ollama は `GET /api/tags`、openai は `GET /v1/models` から埋める。自由入力可）、思考モード（Checkbox）、並列数。ここでの値は config の `llm` を上書きして `build_script` に渡す（`llm_overrides`）。config に無いモデルを選んでも動く。
- **タブ 1「試聴」**: 本文（絵文字パレット: `irodori_tts.gradio_emoji_palette.build_emoji_palette(textbox)` を import できれば使い、できなければ感情プリセット Dropdown + 強度 Slider → `to_emoji` をカーソル位置か文頭に挿入するボタン。「LLM で付ける」ボタンで本文 1 段落を `LLMAnnotator` に通し、句ごとの絵文字入りの本文に置き換える）、num_steps / cfg_scale_text / cfg_scale_speaker / duration_scale / seed（-1 でランダム）、「生成」→ Audio と使用 seed・所要時間。「感情プリセット総当たり」ボタン（同じ seed で 10 感情 × 指定強度を生成し、`emotion_emoji` の調整に使う）。
- **タブ 2「小説生成」**: 小説ファイル（`gr.File` アップロードまたはパス入力）、章の範囲、「感情を付けない」「感情を作り直す（reannotate）」「作り直す（force）」チェック。ボタンは 3 つで、それぞれ独立に押せる。
  1. 「台本を作る（script.tsv）」→ `build_script`（LLM 設定パネルの値を `llm_overrides` で渡す）。結果を `gr.Dataframe` に表示（text 列は編集可、labels / source 他は表示のみ）。「script.tsv を保存」で手直しを書き戻す（source が一致する行は次回以降もその値が使われる）。既に script.tsv があれば起動時・小説選択時に読み込んで表示する。
  2. 「音声にする」→ Dataframe の編集を script.tsv に保存 → `load_book` → `check_vram`（不足なら状態欄に案内して中止）→ `synthesize_book` を generator で回し `gr.Progress` と状態欄に「章 n / 文 i/N」を出す。完了後、章 WAV の `gr.Files` と全体 WAV の `gr.Audio`。script.tsv が無ければ案内して中止。
  3. 「一括実行」→ 1 → 2 を続けて実行（1 の中で LLM を解放してから 2 に進む）。
  「停止」ボタンは `should_stop` フラグを立てる（セグメントキャッシュがあるので再開は続きから）。

## 実装手順

1. 実装開始時に `obsidian-task-log` でタスクを登録する（ユーザーの運用ルール）。
2. 足場: `pyproject.toml`, `.python-version`, `.gitignore`, `voices/.gitkeep`, パッケージ雛形。Mac で `uv lock` と `uv sync --extra dev` が通ることを確認し、`uv.lock` に `irodori-tts` と `dacvae` が git で解決されていることを見る。
3. `text_splitter.py`, `assemble.py`, `tests/sample_novel.txt`, `tests/test_text_splitter.py` を voxcpm-speaker からコピーし、テストが通ることを確認。
4. `config.py`（新しいキー構成）+ `config.example.yaml`。
5. `emotion.py`: `split_clauses`（TDD: 句読点・括弧・連続句読点・結合して元に戻る）→ `validate`（句単位）→ `to_emoji` / `build_spoken_text`（連発抑止・括弧直後）→ `LLMAnnotator`（ollama / openai の 2 系統、json_object フォールバック、think 除去、並列、unload）→ `LLMServer`。`tests/test_emotion.py`（HTTP とサブプロセスは monkeypatch でモック）。
6. `voices.py` + `tests/test_voices.py`（tmp_path にダミー音声ファイルを置いて列挙・解決・キーを検証）。
7. `synth.py`: `cache_key` のテストを先に書き、`load_runtime` / `encode_voice` / `Synthesizer` を実装（irodori_tts は遅延 import。Mac ではテストが import しないことを確認）。
8. `pipeline.py` + `tests/test_pipeline.py`（Synthesizer と LLMAnnotator をモックし、`build_script` が script.tsv と chapters.json を書く・source 一致行の labels / text が保持される・本文変更行だけ再推定される・`reannotate` で絵文字が表から作り直される・`--chapters` 部分処理で他章の行を残す・`load_book` が無いときに案内する・`synthesize_book` が script.tsv だけから動く・`should_stop` を検証）。
9. `cli.py`（`script` / `synth` / `run` サブコマンド）+ `__main__.py`。Mac で `script … --no-emotion` を通す。
10. `webui.py`。Mac では `uv run --with gradio python -m irodori_speaker.webui` で画面だけ確認（生成不可）。
11. README.md（WSL2 での Ollama 導入、llama-server / vLLM への切替方法、8GB での注意、透かしと声の同意）/ CLAUDE.md。
12. GPU マシンで実機確認（下記）。

## 検証

**Mac（モデル不要）**
```bash
uv sync --extra dev
uv run pytest -q
uv run python -m irodori_speaker script tests/sample_novel.txt -c config.example.yaml --no-emotion   # script.tsv / chapters.json 出力（LLM 無し）
uv run python -m irodori_speaker synth tests/sample_novel.txt -c config.example.yaml                 # torch が無いので import エラーではなく案内が出ること
uv run --with gradio python -m irodori_speaker.webui -c config.example.yaml                              # 画面確認
```

**GPU マシン**
```bash
uv sync --frozen --extra tts --extra cu128 --extra webui
cp config.example.yaml config.yaml   # voice などを編集。voices/narrator/ に参照クリップを置く
curl -fsSL https://ollama.com/install.sh | sh && ollama pull qwen3.5:4b-q8_0                  # WSL2 側に Ollama（5.2GB）
uv run --no-sync python -m irodori_speaker script tests/sample_novel.txt -c config.yaml        # ステップ 1: script.tsv に句ごとの絵文字が入る。終了後 nvidia-smi で LLM が解放されている
uv run --no-sync python -m irodori_speaker synth  tests/sample_novel.txt -c config.yaml        # ステップ 2: out/sample_novel/ に章 WAV と全体 WAV
uv run --no-sync python -m irodori_speaker run    tests/sample_novel.txt -c config.yaml --force   # 一括
uv run --no-sync python -m irodori_speaker.webui -c config.yaml --host 0.0.0.0                  # Mac のブラウザから。LLM 一覧の取得、台本 → 音声 → 一括の各ボタン
```
確認項目: 参照音声の潜在 `.pt` が 1 回だけ作られる／2 回目の実行がキャッシュで即終了する／`--force` で作り直す／script.tsv の text 列の絵文字を手で動かすと当該文だけ再合成される／句読点ごとに絵文字が入り、同じ絵文字が連続しない／`think: false` で Qwen3.5 の応答に `<think>` が混ざらず JSON が返る／WebUI の停止後に再開すると続きから進む／1 文 80 文字が 30 秒に収まる（`out/<book>/warnings.log` に長さ切り詰めの警告が無い）／`voices/` を rsync し直しても再合成が起きない／注釈後に `nvidia-smi` で Ollama のモデルが解放されている／TTS が bf16 で OOM せずに動く（だめなら int8 版で動く）／`server_command` に llama-server を書いた場合に注釈の前後で起動・停止される／Ollama を止め忘れた状態で合成を始めると VRAM 不足の案内が出る／`synth` だけを単独で実行しても LLM に接続しない／WebUI の LLM 一覧に `ollama list` と同じモデルが出て、選んだモデルで台本が作られる。

## 実行方法（ユーザー指定: ワークツリーごとに Sonnet のワーカーへ並行で作業させる）

`worktree-orchestration` Skill の手順で進める。コーディネーター（このセッション）は指示書・確認・Codex レビュー・報告を担当し、実装は Orca のワーカー（`--model claude-sonnet-5-5`、TDD）に任せる。タスクごとに Obsidian Task Board に記録する（`obsidian-task-log`）。

**前提の整備（コーディネーターが行う）**: リポジトリはコミットが無く、origin（github.com/din-iis/irodori-speaker）も空。オーケストレーションのスクリプトは `origin/main` を前提にするので、最初に最小の初期コミット（`.gitignore`、`.python-version`、README の見出しだけ、`docs/superpowers/specs/2026-10-24-irodori-speaker-design.md` = この計画の設計部分、`tests/sample_novel.txt`）を `main` に作って push する。各波の終わりに、完了したブランチを `main` に取り込んで push し、次の波のワークツリーが最新を取り込めるようにする（PR は作らない。必要なら後で `worktree-pr`）。

**波（wave）と作業パッケージ**。1 タスク = 1 ワークツリー = 1 ワーカー。

| 波 | WP | 内容（計画の手順番号） | 触るファイル |
|---|---|---|---|
| 1 | WP0 足場 | 手順 2〜4: pyproject / uv.lock（Mac で `uv lock` が通ること）/ パッケージ雛形 / `text_splitter.py` `assemble.py` とそのテストのコピー / `config.py` / `config.example.yaml` | 上記すべて（他 WP の土台） |
| 2 | WP1 感情 | 手順 5: `emotion.py`（句分割、句単位の検証、絵文字表、`build_spoken_text`、`LLMAnnotator` 2 系統、`LLMServer`）+ `tests/test_emotion.py` | `irodori_speaker/emotion.py`, `tests/test_emotion.py` |
| 2 | WP2 合成 | 手順 6〜7: `voices.py`, `synth.py` + `tests/test_voices.py`, `tests/test_synth.py` | `irodori_speaker/voices.py`, `synth.py`, 対応テスト |
| 2 | WP3 パイプラインと CLI | 手順 8〜9: `pipeline.py`, `cli.py`, `__main__.py` + `tests/test_pipeline.py`。WP1/WP2 のクラスは計画のシグネチャどおりに**引数で受け取り**、`cli.py` の中で遅延 import する（テストはモックで通す。WP1/WP2 が無くてもテストが通る） | `irodori_speaker/pipeline.py`, `cli.py`, `__main__.py`, `tests/test_pipeline.py` |
| 3 | WP4 WebUI | 手順 10: `webui.py`（TTS 設定 / LLM 設定 / 試聴 / 小説生成の 3 ボタン） | `irodori_speaker/webui.py` |
| 3 | WP5 ドキュメント | 手順 11: README.md / CLAUDE.md（WP1〜3 の実装を読んで書く） | `README.md`, `CLAUDE.md` |

- 波 2 の 3 本は互いに別ファイルなので衝突しない。波 2 が全部 main に入ってから波 3 を起動する
- 各 WP の完了時にコーディネーターが差分・テスト・振る舞いを自分で確かめ、Codex レビューを指摘が無くなるまで回し、pr-lens の図をローカルに描く
- **統合確認（波 3 の後、コーディネーター）**: Mac で `uv sync --extra dev && uv run pytest`、`script … --no-emotion`、WebUI の画面確認。TTS を含む実機確認は GPU マシン（WSL2）で行う必要があり、このセッションからは触れないので、手順を README に書いてユーザーに依頼する。Mac で `uv sync --extra tts`（MPS、fp32）による 2〜3 文のスモークテストを行うかは、所要時間（torch 約 2GB + モデル約 3GB のダウンロード）を見てその時点で判断する

## 対象外（今回やらない）

- 登場人物ごとの声の割り当て（話者推定）
- Irodori-TTS-Server 経由の HTTP クライアント構成
- MP3 等への変換、字幕出力
- VoiceDesign（キャプション）チェックポイント
