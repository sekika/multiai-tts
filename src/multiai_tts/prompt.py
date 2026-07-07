import io
import enum
import os
import wave
import multiai
import subprocess
import tempfile
import sounddevice as sd
import soundfile as sf
import requests
from openai import OpenAI
from google import genai
import azure.cognitiveservices.speech as speechsdk

# Azure TTS error code mapping
# https://learn.microsoft.com/en-us/javascript/api/microsoft-cognitiveservices-speech-sdk/cancellationerrorcode
AZURE_ERROR_CODES = {
    0: "NoError",
    1: "AuthenticationFailure",
    2: "BadRequestParameters",
    3: "TooManyRequests",
    4: "ConnectionFailure",
    5: "ServiceTimeout",
    6: "ServiceError",
    7: "RuntimeError",
    8: "Forbidden",
}


class Prompt(multiai.Prompt):
    """
    Extends multiai.Prompt to include Text-to-Speech capabilities
    using OpenAI and Google GenAI providers.
    """

    def __init__(self):
        super(Prompt, self).__init__()
        self.tts_voice_openai = 'marin'
        self.tts_voice_google = 'charon'
        self.tts_framerate_google = 24000
        self.tts_voice_azure = 'en-US-AriaNeural'
        self.tts_voice_voicevox = 1
        self.tts_voicevox_url = 'http://127.0.0.1:50021'
        self.tts_voicevox_timeout = 60

    def set_tts_provider(self, provider):
        """Sets the active TTS provider."""
        try:
            self.tts_provider = TTS_Provider[provider.upper()]
        except Exception:
            self.error = True
            self.error_message = f'multiai-tts system error: TTS provider "{provider}" is not available.'
            return

    def set_tts_model(self, provider, model):
        """Sets the TTS provider and the specific model to use."""
        self.set_tts_provider(provider)
        self.tts_model = model
        setattr(self, 'model_' + provider.lower(), model)

    def speak(self, text: str,
              prompt: str = "",
              chunk_size: int = None,
              split_chars: str = "。．.!！?？\n",
              chunk_overflow: str = "extend"):
        """Generates audio from ``text`` and plays it using sounddevice.

        ``prompt`` is an optional style instruction (voice, tone, speed, …).
        It is *not* part of the spoken text and is *not* subject to chunk
        splitting: when ``chunk_size`` splits ``text`` into chunks, ``prompt``
        is re-applied to every chunk so the style stays consistent. An empty
        ``prompt`` reproduces the original behavior (synthesize ``text`` only).

        When ``chunk_size`` is a positive integer, ``text`` is split into
        chunks (see :meth:`split_text`) and the generated audio of every chunk
        is concatenated before playback.
        """
        self.error = False

        if chunk_size is None:
            # Request WAV format specifically for playback compatibility
            wav_bytes = self.get_wav(text, fmt='wav', prompt=prompt)
        else:
            wav_bytes = self._get_chunked_wav(
                text, chunk_size, split_chars, chunk_overflow, prompt=prompt)

        if self.error or not wav_bytes:
            return

        try:
            wav_io = io.BytesIO(wav_bytes)
            data, samplerate = sf.read(wav_io)
            sd.play(data, samplerate)
            sd.wait()
        except Exception as e:
            self.error = True
            self.error_message = f"Playback error: {e}"

    def save_tts(self, text: str, filename: str,
                 prompt: str = "",
                 chunk_size: int = None,
                 split_chars: str = "。．.!！?？\n",
                 chunk_overflow: str = "extend"):
        """
        Generates audio from ``text`` and saves it to a file.
        Automatically handles format conversion based on file extension.

        ``prompt`` is an optional style instruction (voice, tone, speed, …).
        It is *not* part of the spoken text and is *not* subject to chunk
        splitting: when ``chunk_size`` splits ``text`` into chunks, ``prompt``
        is re-applied to every chunk so the style stays consistent. An empty
        ``prompt`` reproduces the original behavior (synthesize ``text`` only).

        When ``chunk_size`` is a positive integer, ``text`` is split into
        chunks (see :meth:`split_text`), each chunk is synthesized as WAV and
        the results are concatenated before being written/converted to
        ``filename``.
        """
        # Determine format from extension
        _, ext = os.path.splitext(filename)
        fmt = ext.lower().replace('.', '')
        if not fmt:
            fmt = 'wav'

        self.error = False

        if chunk_size is not None:
            # Chunked mode: always synthesize WAV, concatenate, then convert.
            wav_bytes = self._get_chunked_wav(
                text, chunk_size, split_chars, chunk_overflow, prompt=prompt)
            if self.error or not wav_bytes:
                return
            try:
                self._save_wav_as(wav_bytes, filename, fmt)
            except Exception as e:
                self.error = True
                self.error_message = f"Failed to save/convert audio: {str(e)}"
            return

        # Fetch audio bytes (OpenAI attempts native format, Google returns WAV)
        audio_bytes = self.get_wav(text, fmt=fmt, prompt=prompt)

        if self.error or not audio_bytes:
            return

        try:
            # Check if we can save directly without conversion
            openai_direct_formats = [
                'mp3', 'opus', 'aac', 'flac', 'wav', 'pcm']
            is_direct = False

            if self.tts_provider == TTS_Provider.OPENAI:
                if fmt in openai_direct_formats:
                    is_direct = True
                elif fmt == 'ogg':
                    # ogg is requested as opus from OpenAI
                    is_direct = True

            if fmt == 'wav':
                is_direct = True

            if is_direct:
                with open(filename, "wb") as f:
                    f.write(audio_bytes)
            else:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(audio_bytes)
                    tmp_path = tmp.name

                try:
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-y",
                            "-i",
                            tmp_path,
                            filename,
                        ],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                finally:
                    os.remove(tmp_path)

        except Exception as e:
            self.error = True
            self.error_message = f"Failed to save/convert audio: {str(e)}"

    def split_text(self, text: str, chunk_size: int,
                   split_chars: str = "。．.!！?？\n",
                   chunk_overflow: str = "extend"):
        """Split ``text`` into chunks no longer than ``chunk_size`` characters.

        ``text`` is the spoken body only; style instructions (the ``prompt``
        argument of :meth:`speak` / :meth:`save_tts`) are never passed here, so
        ``chunk_size`` is measured against the body length and is unaffected by
        the prompt length.

        Within each ``chunk_size`` window the split position is the position
        immediately after the last (rightmost) character contained in
        ``split_chars``. When no such character exists in the window the
        behaviour is controlled by ``chunk_overflow``:

        - ``"extend"`` keeps reading past ``chunk_size`` until the next
          ``split_chars`` character is found (or the end of the text);
        - ``"error"`` sets ``self.error`` and ``self.error_message`` and
          returns ``None``.

        Returns the list of chunks, or ``None`` on error.
        """
        self.error = False
        if chunk_overflow not in ("extend", "error"):
            self.error = True
            self.error_message = (
                "multiai-tts system error: chunk_overflow must be "
                f'"extend" or "error", got "{chunk_overflow}".')
            return None

        if not isinstance(chunk_size, int) or chunk_size <= 0:
            self.error = True
            self.error_message = (
                "multiai-tts system error: chunk_size must be a positive "
                f"integer, got {chunk_size!r}.")
            return None

        chunks = []
        remaining = text
        while remaining:
            if len(remaining) <= chunk_size:
                chunks.append(remaining)
                break

            window = remaining[:chunk_size]
            split_pos = -1
            for i in range(len(window) - 1, -1, -1):
                if window[i] in split_chars:
                    split_pos = i
                    break

            if split_pos != -1:
                cut = split_pos + 1
            elif chunk_overflow == "error":
                self.error = True
                self.error_message = (
                    "multiai-tts error: no split character found within "
                    f"chunk_size={chunk_size}.")
                return None
            else:  # extend: search beyond chunk_size
                cut = len(remaining)
                for i in range(chunk_size, len(remaining)):
                    if remaining[i] in split_chars:
                        cut = i + 1
                        break

            chunks.append(remaining[:cut])
            remaining = remaining[cut:]

        return chunks

    def _get_chunked_wav(self, text, chunk_size,
                         split_chars, chunk_overflow, prompt=""):
        """Split the body text, synthesize each chunk as WAV and concatenate.

        Only ``text`` is split; ``prompt`` (a style instruction) is re-applied
        to every chunk so the style stays consistent across the whole audio.

        Returns the combined WAV bytes, or ``None`` on error (with
        ``self.error`` set).
        """
        self.chunks = self.split_text(
            text, chunk_size, split_chars, chunk_overflow)
        if self.error or not self.chunks:
            return None

        wav_list = []
        for chunk in self.chunks:
            wav_bytes = self.get_wav(chunk, fmt='wav', prompt=prompt)
            if self.error or not wav_bytes:
                return None
            wav_list.append(wav_bytes)

        try:
            return self._combine_wav(wav_list)
        except Exception as e:
            self.error = True
            self.error_message = f"Failed to combine audio: {str(e)}"
            return None

    def _combine_wav(self, wav_list):
        """Concatenate a list of WAV byte strings into a single WAV byte string.

        Concatenation is done in memory using ``soundfile``; no silence is
        inserted between chunks.
        """
        import numpy as np

        if len(wav_list) == 1:
            return wav_list[0]

        arrays = []
        samplerate = None
        for wav_bytes in wav_list:
            data, sr = sf.read(io.BytesIO(wav_bytes))
            if samplerate is None:
                samplerate = sr
            arrays.append(data)

        combined = np.concatenate(arrays)
        out = io.BytesIO()
        sf.write(out, combined, samplerate, format='WAV')
        return out.getvalue()

    def _save_wav_as(self, wav_bytes, filename, fmt):
        """Write WAV bytes to ``filename``, converting via ffmpeg if needed."""
        if fmt == 'wav':
            with open(filename, "wb") as f:
                f.write(wav_bytes)
            return

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_bytes)
            tmp_path = tmp.name
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_path, filename],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        finally:
            os.remove(tmp_path)

    def get_wav(self, text: str, fmt: str = 'wav', prompt: str = ""):
        """Dispatch method to generate audio bytes using the selected provider.

        ``text`` is the spoken body. ``prompt`` is an optional style
        instruction prepended to the text before synthesis. The same rule is
        used for every provider; an empty (or ``None``) ``prompt`` leaves the
        request unchanged.
        """
        if prompt:
            self.prompt = f"{prompt}\n\n{text}"
        else:
            self.prompt = text
        func_name = 'get_wav_' + self.tts_provider.name.lower()
        try:
            func = getattr(self, func_name)
        except AttributeError:
            self.error = True
            self.error_message = f'multiai-tts system error: {func_name}() function is not defined.'
            return None

        # OpenAI accepts a format argument, others do not
        if self.tts_provider == TTS_Provider.OPENAI:
            func(fmt)
        else:
            func()

        if self.error:
            return None
        return self.wav

    def get_wav_openai(self, fmt: str = 'wav'):
        """Internal method to fetch audio from OpenAI."""
        try:
            client = OpenAI(api_key=self.openai_api_key)

            # Map requested format to OpenAI API supported formats
            api_fmt = 'wav'
            if fmt in ['mp3', 'aac', 'flac', 'wav', 'pcm']:
                api_fmt = fmt
            elif fmt == 'ogg':
                api_fmt = 'opus'

            response = client.audio.speech.create(
                model=self.model_openai,
                voice=self.tts_voice_openai,
                input=self.prompt,
                response_format=api_fmt
            )
            self.error = False
            self.wav = response.content

        except Exception as e:
            self.handle_error(e)

    def get_wav_google(self):
        """Internal method to fetch audio from Google GenAI."""
        if not getattr(self, 'google_api_key', None):
            self.error = True
            self.error_message = "Google API key is not set."
            return

        client = genai.Client(api_key=self.google_api_key)
        config = genai.types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=genai.types.SpeechConfig(
                voice_config=genai.types.VoiceConfig(
                    prebuilt_voice_config=genai.types.PrebuiltVoiceConfig(
                        voice_name=self.tts_voice_google
                    )
                )
            )
        )

        try:
            response = client.models.generate_content(
                model=self.model_google,
                contents=[self.prompt],
                config=config
            )
            self.error = False

            raw_audio = None

            # Safely check for content existence to avoid NoneType errors
            if (response.candidates and
                len(response.candidates) > 0 and
                response.candidates[0].content and
                    response.candidates[0].content.parts):

                for part in response.candidates[0].content.parts:
                    if part.inline_data:
                        raw_audio = part.inline_data.data
                        break

            if raw_audio:
                # Wrap raw PCM in WAV container
                buffer = io.BytesIO()
                with wave.open(buffer, "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(self.tts_framerate_google)
                    wav.writeframes(raw_audio)
                self.wav = buffer.getvalue()
            else:
                self.error = True
                msg = "No audio data returned from Google."
                if response.prompt_feedback:
                    msg += f" Feedback: {response.prompt_feedback}"
                self.error_message = msg

        except Exception as e:
            self.handle_error(e)

    def get_wav_azure(self, fmt: str = 'wav'):
        """Fetch WAV bytes from Azure TTS using self.prompt, do not play audio."""

        if not getattr(self, 'azure_tts_api_key', None):
            self.error = True
            self.error_message = "Azure TTS API key is not set."
            self.wav = None
            return

        try:
            speech_config = speechsdk.SpeechConfig(
                subscription=self.azure_tts_api_key,
                region=self.azure_tts_region
            )
            speech_config.speech_synthesis_voice_name = self.tts_voice_azure

            # **Critical:** output only to memory, no default speaker
            audio_stream = speechsdk.audio.PullAudioOutputStream()
            audio_output_config = speechsdk.audio.AudioOutputConfig(
                stream=audio_stream)
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=speech_config,
                audio_config=audio_output_config
            )

            # Generate speech
            result = synthesizer.speak_text_async(self.prompt).get()

            if result.reason == speechsdk.ResultReason.Canceled:
                cancellation = result.cancellation_details
                code_int = cancellation.error_code.value
                code_str = AZURE_ERROR_CODES.get(
                    code_int, f"UnknownError({code_int})")
                self.error = True
                self.error_message = (
                    f"Azure TTS failed: {cancellation.reason} ({code_str})\n"
                    f"Details: {cancellation.error_details}"
                )
                self.wav = None
                return

            # Read the audio from the PullAudioOutputStream
            buffer = io.BytesIO()
            buffer.write(result.audio_data)
            self.wav = buffer.getvalue()
            self.error = False

        except Exception as e:
            self.handle_error(e)
            self.wav = None

    def get_wav_voicevox(self):
        """Fetch WAV bytes from a locally running VOICEVOX engine.

        The engine is assumed to be already running (default
        ``http://127.0.0.1:50021``). Synthesis is a two-step call:
        ``/audio_query`` builds the query from the text and speaker, then
        ``/synthesis`` returns the WAV audio. On any failure (engine not
        reachable, timeout, HTTP error, empty audio) no exception is raised;
        ``self.error`` and ``self.error_message`` are set and ``self.wav`` is
        cleared.
        """
        url = self.tts_voicevox_url.rstrip('/')
        speaker = self.tts_voice_voicevox
        timeout = self.tts_voicevox_timeout

        try:
            query_resp = requests.post(
                f"{url}/audio_query",
                params={"text": self.prompt, "speaker": speaker},
                timeout=timeout,
            )
            query_resp.raise_for_status()
            query = query_resp.json()

            synth_resp = requests.post(
                f"{url}/synthesis",
                params={"speaker": speaker},
                json=query,
                timeout=timeout,
            )
            synth_resp.raise_for_status()

            if not synth_resp.content:
                self.error = True
                self.error_message = "No audio data returned from VOICEVOX."
                self.wav = None
                return

            self.wav = synth_resp.content
            self.error = False

        except requests.exceptions.ConnectionError:
            self.error = True
            self.error_message = (
                f"VOICEVOX engine is not reachable at {url}. Is it running?")
            self.wav = None
        except requests.exceptions.Timeout:
            self.error = True
            self.error_message = (
                f"VOICEVOX request timed out after {timeout}s ({url}).")
            self.wav = None
        except requests.exceptions.HTTPError as e:
            resp = e.response
            status = resp.status_code if resp is not None else "Error"
            body = resp.text if resp is not None else str(e)
            self.error = True
            self.error_message = f"VOICEVOX HTTP error {status}\n{body}"
            self.wav = None
        except Exception as e:
            self.handle_error(e)
            self.wav = None

    def handle_error(self, e):
        """Parses exception or result details into a readable error message."""
        self.error = True

        # --- Azure TTS ResultReason.Canceled handling ---
        if hasattr(e, 'reason') and e.reason == speechsdk.ResultReason.Canceled:
            cancellation = getattr(e, 'cancellation_details', None)
            if cancellation:
                code = getattr(cancellation, 'error_code', 'Canceled')
                details = getattr(
                    cancellation, 'error_details', 'No details provided')
                self.error_message = f"Azure TTS canceled: {code}\n{details}"
                return

        # --- OpenAI structured body ---
        body = getattr(e, 'body', None)
        if isinstance(body, dict) and 'error' in body:
            err = body['error']
            code = err.get('code') or getattr(e, 'code', 'Error')
            message = err.get('message')
            if message:
                self.error_message = f"Error {code} Error\n{message}"
                return

        # --- Standard attributes (Google, OpenAI fallback) ---
        code = getattr(e, 'code', None)
        message = getattr(e, 'message', None)
        if code and message:
            msg_str = str(message)
            # Attempt to parse verbose OpenAI dump
            if msg_str.startswith("Error code:") and " - {'error':" in msg_str:
                try:
                    import ast
                    dict_str = msg_str.split(" - ", 1)[1]
                    err_data = ast.literal_eval(dict_str)
                    if 'error' in err_data and 'message' in err_data['error']:
                        message = err_data['error']['message']
                except Exception:
                    pass  # keep original message

            status = getattr(e, 'status', 'Error')
            self.error_message = f"Error {code} {status}\n{message}"
            return

        # --- Fallback regex for unstructured exceptions ---
        import re
        raw_text = str(e)
        code_match = re.search(r"'code':\s*(\d+|'[^']+')", raw_text)
        status_match = re.search(r"'status':\s*'([^']+)'", raw_text)
        msg_match = re.search(r"'message':\s*'([^']+)'", raw_text)

        if code_match and msg_match:
            p_code = code_match.group(1).replace("'", "")
            p_status = status_match.group(1) if status_match else "Error"
            p_msg = msg_match.group(1)
            self.error_message = f"Error {p_code} {p_status}\n{p_msg}"
        else:
            self.error_message = raw_text


class TTS_Provider(enum.Enum):
    OPENAI = enum.auto()
    GOOGLE = enum.auto()
    AZURE = enum.auto()
    VOICEVOX = enum.auto()
