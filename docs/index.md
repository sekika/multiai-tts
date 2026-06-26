# multiai-tts

`multiai-tts` is an extension library for [multiai](https://sekika.github.io/multiai/) that provides Text-to-Speech (TTS) capabilities using OpenAI, Google GenAI, and Azure Speech.

| Provider | Strengths | Docs |
|----------|-----------|------|
| [OpenAI](https://platform.openai.com/docs/guides/text-to-speech) | Simple API | [Voices](https://platform.openai.com/docs/guides/text-to-speech#voice-options) · [API](https://platform.openai.com/docs/api-reference/audio/createSpeech) |
| [Google GenAI](https://ai.google.dev/gemini-api/docs/audio) | Emotion tags, multi-speaker | [Voices](https://ai.google.dev/gemini-api/docs/audio#voices) · [API](https://ai.google.dev/api/generate-content) |
| [Azure Speech](https://learn.microsoft.com/azure/ai-services/speech-service/text-to-speech) | SSML, extensive voice selection | [Voices](https://learn.microsoft.com/azure/ai-services/speech-service/language-support?tabs=tts) · [API](https://learn.microsoft.com/azure/ai-services/speech-service/rest-text-to-speech) |

## Prerequisites

**API Key Configuration**

This library relies on the configuration provided by `multiai`. You must set up your API keys (OpenAI API Key, Google API Key, Azure TTS Key and Region) using `multiai`'s configuration files or environment variables before using this library.

For details on how to configure API keys, please refer to the **[multiai documentation](https://sekika.github.io/multiai/)**.

**System Requirements**

- `ffmpeg` must be installed if you want to save audio in formats other than WAV (e.g., MP3).

## Installation

```bash
pip install multiai-tts
```

## Usage

### Google GenAI Example

```python
import sys
import multiai_tts

client = multiai_tts.Prompt()
client.set_tts_model('google', 'gemini-2.5-flash-preview-tts')
client.tts_voice_google = 'charon'

# Speak directly
client.speak("Please speak the following. Hello, this is a test from Google model.")
if client.error:
    print(client.error_message)
    sys.exit(1)

# Save to file
client.save_tts("Please speak the following. Saving this audio to mp3.", "output_google.mp3")
if client.error:
    print(client.error_message)
    sys.exit(1)
```

### OpenAI Example

```python
import sys
import multiai_tts

client = multiai_tts.Prompt()
client.set_tts_model('openai', 'gpt-4o-mini-tts')
client.tts_voice_openai = 'marin'

# Speak directly
client.speak("Hello, this is a test from OpenAI model.")
if client.error:
    print(client.error_message)
    sys.exit(1)

# Save to file
client.save_tts("Saving this audio to mp3.", "output_openai.mp3")
if client.error:
    print(client.error_message)
    sys.exit(1)
```

### Azure TTS Example

```python
import sys
import multiai_tts

client = multiai_tts.Prompt()
client.set_tts_provider('azure')
client.tts_voice_azure = 'en-US-JennyNeural'

# Speak directly
client.speak("Hello, this is a test from Azure TTS.")
if client.error:
    print(client.error_message)
    sys.exit(1)

# Save to file
client.save_tts("Saving this audio to mp3.", "output_azure.mp3")
if client.error:
    print(client.error_message)
    sys.exit(1)
```

## Long text (automatic chunking)

When the text exceeds a provider's request length limit, `speak()` and
`save_tts()` can automatically split the text into chunks, synthesize each
chunk, and join the resulting audio.

```python
# Split into chunks of at most ~1000 characters and join the audio
client.save_tts(long_text, "output.mp3", chunk_size=1000)
if client.error:
    print(client.error_message)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `chunk_size` | `int` or `None` | `None` | Maximum characters per chunk. `None` disables splitting (original behavior). |
| `split_chars` | `str` | `"。．.!！?？\n"` | Candidate split characters. The split point is just after the rightmost candidate found within `chunk_size`. |
| `chunk_overflow` | `str` | `"extend"` | Behavior when no candidate is found within `chunk_size`: `"extend"` reads on until the next candidate (or end of text); `"error"` sets `client.error` and stops. |

`split_text()` is also exposed directly if you only need the chunk boundaries:

```python
chunks = client.split_text(long_text, chunk_size=1000)
```

**Caveats**

* At chunk boundaries pitch, tempo, and trailing reverberation may shift
  slightly. This is an inherent limitation of the TTS APIs (each chunk is
  synthesized independently); no silence is inserted between chunks.
* With `chunk_overflow="extend"`, the actual chunk size may significantly
  exceed `chunk_size`.
* Per-provider API character limits are not managed by this library — you are
  responsible for choosing an appropriate `chunk_size`.

## Notes

* For OpenAI and Google TTS, use `set_tts_model(provider, model)` to select both provider and model.
* For Azure, `set_tts_provider('azure')` is sufficient; the model parameter is not used.
* In Google’s example, the prompt includes “Please speak the following.” In the OpenAI and Azure examples, it does not. Whether you include this phrase depends on the model you use.
* `Prompt.get_wav()` fetches the raw audio data in memory. Playback is separate from retrieval.
* Error handling: After `speak()` or `save_tts()`, always check `client.error` and `client.error_message`.
* WAV output is default; `ffmpeg` is used for converting to other formats.
