# multiai-tts

`multiai-tts` is an extension library for [multiai](https://sekika.github.io/multiai/) that provides Text-to-Speech (TTS) capabilities using OpenAI, Google GenAI, and Azure Speech.

## Prerequisites

**API Key Configuration**

This library relies on the configuration provided by `multiai`. You must set up your API keys (OpenAI API Key, Google API Key, Azure TTS Key and Region) using `multiai`'s configuration files or environment variables before using this library.

For details on how to configure API keys, please refer to the **[multiai documentation](https://sekika.github.io/multiai/)**.

**System Requirements**

- `ffmpeg` must be installed if you want to save audio in formats other than WAV (e.g., MP3).
- `pydub` is required for audio conversion.

## Installation

```bash
pip install multiai-tts
````

## Usage

### Google GenAI Example

```python
import sys
import multiai_tts

client = multiai_tts.Prompt()
client.set_tts_model('google', 'gemini-2.5-flash-preview-tts')
client.tts_voice_google = 'charon'

# Speak directly
client.speak("Hello, this is a test from Google model.")
if client.error:
    print(client.error_message)
    sys.exit(1)

# Save to file
client.save_tts("Saving this audio to mp3.", "output_google.mp3")
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

## Notes

* `Prompt.get_wav()` fetches the raw audio data in memory. Playback is separate from retrieval.
* Error handling: After `speak()` or `save_tts()`, always check `client.error` and `client.error_message`.
* WAV output is default; use `pydub`/`ffmpeg` for other formats.