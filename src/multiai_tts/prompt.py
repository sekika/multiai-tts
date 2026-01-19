import io
import enum
import os
import wave
import multiai
import sounddevice as sd
import soundfile as sf
from pydub import AudioSegment
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

    def speak(self, prompt: str):
        """Generates audio from the prompt and plays it using sounddevice."""
        # Request WAV format specifically for playback compatibility
        self.error = False
        wav_bytes = self.get_wav(prompt, fmt='wav')
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

    def save_tts(self, prompt: str, filename: str):
        """
        Generates audio and saves it to a file.
        Automatically handles format conversion based on file extension.
        """
        # Determine format from extension
        _, ext = os.path.splitext(filename)
        fmt = ext.lower().replace('.', '')
        if not fmt:
            fmt = 'wav'

        # Fetch audio bytes (OpenAI attempts native format, Google returns WAV)
        self.error = False
        audio_bytes = self.get_wav(prompt, fmt=fmt)

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
                # Convert using pydub (requires ffmpeg)
                segment = AudioSegment.from_wav(io.BytesIO(audio_bytes))
                segment.export(filename, format=fmt)

        except Exception as e:
            self.error = True
            self.error_message = f"Failed to save/convert audio: {str(e)}"

    def get_wav(self, prompt: str, fmt: str = 'wav'):
        """Dispatch method to generate audio bytes using the selected provider."""
        self.prompt = prompt
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
