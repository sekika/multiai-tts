# 仕様書: VOICEVOX プロバイダ対応

## 1. 概要

`multiai-tts` に、ローカルで動作する [VOICEVOX](https://voicevox.hifiman.jp/) エンジンを
TTS プロバイダとして追加する。既存の OpenAI / Google GenAI / Azure Speech と同じ
インターフェース（`speak()` / `save_tts()` / `get_wav()`）で利用できるようにする。

VOICEVOX エンジンは **すでに起動済み**（既定で `http://127.0.0.1:50021` で待受）であること
を前提とする。エンジンが起動していない・応答しない・音声生成に失敗した場合は、他プロバイダ
と同様に例外を送出せず、`self.error = True` と `self.error_message` に理由を格納して返す。

## 2. 前提と特徴

- **APIキー不要**：VOICEVOX はローカル HTTP サーバなので、認証情報は不要。
- **モデル指定不要**：Azure と同様、`set_tts_provider('voicevox')` だけで利用可能
  （`set_tts_model` も呼べるが `model` は使用しない）。
- **話者は整数の style ID（`speaker`）で指定する**。既定値は設定可能とする。
- **エンジンが WAV を返す**：`/synthesis` は WAV ヘッダ付きの音声を返すため、Google の
  ように生 PCM を WAV でラップする処理は不要。取得したバイト列をそのまま `self.wav` とする。

## 3. VOICEVOX エンジンの呼び出し手順

音声生成は 2 ステップで行う。

### 3.1 音声クエリの作成（`/audio_query`）

```python
query = requests.post(
    f"{self.tts_voicevox_url}/audio_query",
    params={
        "text": text,
        "speaker": speaker,
    },
).json()
```

- `text`：読み上げ対象テキスト（`self.prompt`。スタイルプロンプトを含む場合は連結後の文字列）。
- `speaker`：話者 style ID（`self.tts_voice_voicevox`）。

### 3.2 音声合成（`/synthesis`）

```python
synthesis = requests.post(
    f"{self.tts_voicevox_url}/synthesis",
    params={"speaker": speaker},
    json=query,
)
wav_bytes = synthesis.content   # WAV バイト列（既定 24000Hz / 16bit / mono）
```

`wav_bytes` を `self.wav` に格納する。

## 4. 実装内容

### 4.1 プロバイダ enum の追加

`TTS_Provider` に `VOICEVOX` を追加する。

```python
class TTS_Provider(enum.Enum):
    OPENAI = enum.auto()
    GOOGLE = enum.auto()
    AZURE = enum.auto()
    VOICEVOX = enum.auto()
```

### 4.2 設定属性（`__init__`）

以下を追加する。

| 属性 | 既定値 | 説明 |
|------|--------|------|
| `tts_voice_voicevox` | `1` | 話者 style ID（整数）。 |
| `tts_voicevox_url` | `"http://127.0.0.1:50021"` | VOICEVOX エンジンのベース URL。末尾スラッシュなし。 |
| `tts_voicevox_timeout` | `60` | HTTP リクエストのタイムアウト秒数。 |

### 4.3 合成メソッド `get_wav_voicevox()`

- 引数は取らない（`fmt` を受け取らない）。`self.prompt` を読み上げテキストとして使用する。
- 上記 3.1 → 3.2 の順に呼び出し、成功時は `self.wav = synthesis.content`、`self.error = False` とする。
- HTTP ステータスが 200 以外の場合、および接続不可・タイムアウト等の例外時は
  `self.error = True` とし、`self.error_message` に理由を格納する。`self.wav = None` とする。
  - 接続不可（`requests.exceptions.ConnectionError`）の場合は、エンジンが起動しているか
    確認を促す旨のメッセージを含める（例: `"VOICEVOX engine is not reachable at <URL>. Is it running?"`）。
  - それ以外の例外は既存の `handle_error(e)` に委譲する。

### 4.4 ディスパッチ（`get_wav`）の対応

`get_wav()` は `fmt` を渡すのは OpenAI のみとしている。VOICEVOX は Google / Azure と
同じく `func()`（引数なし）で呼ばれる分岐に入るため、既存の分岐で対応可能。追加変更は不要。

なお `save_tts()` の直接保存判定（`is_direct`）では VOICEVOX は WAV のみを直接返すため、
`fmt == 'wav'` のときだけ直接保存、それ以外は既存の ffmpeg 変換パスで処理される（追加変更不要）。

### 4.5 依存関係

`pyproject.toml` の `dependencies` に `requests` を明示的に追加する
（現状は他パッケージ経由で入るが、直接使用するため明示する）。

## 5. 使用例

```python
import sys
import multiai_tts

client = multiai_tts.Prompt()
client.set_tts_provider('voicevox')
client.tts_voice_voicevox = 3          # 話者 style ID
# client.tts_voicevox_url = "http://127.0.0.1:50021"  # 既定値。変更する場合のみ

# 直接再生
client.speak("こんにちは。VOICEVOX のテストです。")
if client.error:
    print(client.error_message)
    sys.exit(1)

# ファイル保存（mp3 変換には ffmpeg が必要）
client.save_tts("音声をファイルに保存します。", "output_voicevox.mp3")
if client.error:
    print(client.error_message)
    sys.exit(1)
```

長文チャンク分割（`chunk_size`）・スタイルプロンプト（`prompt`）は既存機構がそのまま働く。

## 6. エラーハンドリング方針

| 状況 | 挙動 |
|------|------|
| エンジン未起動・接続不可 | `self.error=True`、URL と起動確認を促すメッセージ |
| タイムアウト | `self.error=True`、タイムアウト旨のメッセージ |
| HTTP エラー（4xx/5xx） | `self.error=True`、ステータスコードと本文を含むメッセージ |
| 音声データが空 | `self.error=True`、"No audio data returned from VOICEVOX." |

いずれの場合も例外は送出せず、`self.wav = None` とする。

## 7. ドキュメント更新

- `README.md`：対応プロバイダに VOICEVOX を追記。
- `docs/index.md`：
  - 「Supported AI providers」表に VOICEVOX（ローカルエンジン・日本語）行を追加。
  - 「Usage」に VOICEVOX example を追加。
  - Notes に「VOICEVOX は API キー不要・ローカルエンジン起動が前提・`model` 不使用」を明記。

## 8. テスト

`tests/` に VOICEVOX 用のユニットテストを追加する。`requests.post` を mock し、
実エンジンなしで以下を検証する。

- `set_tts_provider('voicevox')` で `tts_provider` が `TTS_Provider.VOICEVOX` になる。
- `get_wav()` が `/audio_query` → `/synthesis` の順で呼ばれ、`speaker` に
  `tts_voice_voicevox` が渡ること。
- `/synthesis` の `content` が `self.wav` に格納されること。
- 接続不可（`ConnectionError`）時に `self.error=True` かつ `self.error_message` が設定され、
  例外が外に漏れないこと。
- HTTP エラー時に `self.error=True` になること。

## 9. 影響範囲

- 既存プロバイダ（OpenAI / Google / Azure）の挙動には影響しない（追加のみ）。
- `get_wav` / `save_tts` / `speak` / `split_text` の共通処理は変更なしで再利用する。
