import unittest
import io
import sys
from unittest.mock import MagicMock, patch, mock_open
from multiai_tts import Prompt, TTS_Provider

class TestTTSPrompt(unittest.TestCase):
    def setUp(self):
        """Setup run before each test."""
        self.client = Prompt()
        # Set dummy API keys to avoid errors during tests
        self.client.openai_api_key = "dummy-openai-key"
        self.client.google_api_key = "dummy-google-key"

    def test_initialization(self):
        """Check initialization and default values."""
        self.assertEqual(self.client.tts_voice_openai, 'marin')
        self.assertEqual(self.client.tts_voice_google, 'charon')
        self.assertIsNone(getattr(self.client, 'tts_provider', None))

    def test_set_tts_model_openai(self):
        """Verify OpenAI model configuration."""
        self.client.set_tts_model('openai', 'gpt-4o-mini-tts')
        self.assertEqual(self.client.tts_provider, TTS_Provider.OPENAI)
        self.assertEqual(self.client.tts_model, 'gpt-4o-mini-tts')
        self.assertEqual(self.client.model_openai, 'gpt-4o-mini-tts')

    def test_set_tts_model_google(self):
        """Verify Google model configuration."""
        self.client.set_tts_model('google', 'gemini-2.5-flash-preview-tts')
        self.assertEqual(self.client.tts_provider, TTS_Provider.GOOGLE)
        self.assertEqual(self.client.tts_model, 'gemini-2.5-flash-preview-tts')
        self.assertEqual(self.client.model_google, 'gemini-2.5-flash-preview-tts')

    def test_set_tts_provider_invalid(self):
        """Verify error handling for invalid provider."""
        self.client.set_tts_provider('invalid_provider')
        self.assertTrue(self.client.error)
        self.assertIn('not available', self.client.error_message)

    @patch('multiai_tts.prompt.OpenAI')
    def test_get_wav_openai(self, MockOpenAI):
        """Mock test for OpenAI API call."""
        # Setup mock
        mock_client = MockOpenAI.return_value
        mock_response = MagicMock()
        mock_response.content = b'fake_audio_data'
        mock_client.audio.speech.create.return_value = mock_response

        self.client.set_tts_model('openai', 'tts-1')
        
        # Execute
        wav_bytes = self.client.get_wav('Hello', fmt='mp3')

        # Verify
        self.assertEqual(wav_bytes, b'fake_audio_data')
        self.assertFalse(self.client.error)
        # Check if API was called with correct arguments
        mock_client.audio.speech.create.assert_called_with(
            model='tts-1',
            voice='marin',
            input='Hello',
            response_format='mp3'
        )

    @patch('multiai_tts.prompt.genai')
    def test_get_wav_google(self, mock_genai):
        """Mock test for Google GenAI API call."""
        # Mock Google's complex response structure
        mock_client = mock_genai.Client.return_value
        mock_response = MagicMock()
        
        mock_part = MagicMock()
        mock_part.inline_data.data = b'raw_pcm_data'
        
        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]
        
        mock_response.candidates = [mock_candidate]
        mock_client.models.generate_content.return_value = mock_response

        self.client.set_tts_model('google', 'gemini-test')

        # Mock wave module to simulate header generation
        with patch('wave.open') as mock_wave:
             # Simulate writing to buffer
             mock_wave_instance = mock_wave.return_value.__enter__.return_value
             
             wav_bytes = self.client.get_wav('Hello')
             
             self.assertFalse(self.client.error)
             # Verify API call
             mock_client.models.generate_content.assert_called()

    @patch('multiai_tts.prompt.Prompt.get_wav')
    @patch('builtins.open', new_callable=mock_open)
    def test_save_tts_wav(self, mock_file, mock_get_wav):
        """Test saving as WAV (no conversion)."""
        # Assume get_wav returns audio data
        mock_get_wav.return_value = b'wav_header_and_data'
        self.client.set_tts_provider('openai')

        self.client.save_tts('test prompt', 'output.wav')

        # Verify call to get_wav
        mock_get_wav.assert_called_with('test prompt', fmt='wav')
        
        # Verify file write
        mock_file.assert_called_with('output.wav', 'wb')
        mock_file().write.assert_called_with(b'wav_header_and_data')

    @patch('multiai_tts.prompt.Prompt.get_wav')
    @patch('multiai_tts.prompt.AudioSegment')
    def test_save_tts_conversion(self, mock_audio_segment, mock_get_wav):
        """Test saving as MP3 (conversion via pydub)."""
        # Assume API returns WAV data
        mock_get_wav.return_value = b'wav_data'
        self.client.set_tts_provider('google') # Google always returns WAV

        self.client.save_tts('test prompt', 'music.mp3')

        # Verify AudioSegment loaded the WAV
        mock_audio_segment.from_wav.assert_called()
        # Verify export was called with correct format
        mock_audio_segment.from_wav.return_value.export.assert_called_with('music.mp3', format='mp3')

    @patch('multiai_tts.prompt.Prompt.get_wav')
    @patch('multiai_tts.prompt.sd')
    @patch('multiai_tts.prompt.sf')
    def test_speak(self, mock_sf, mock_sd, mock_get_wav):
        """Test speak method (verify playback library calls)."""
        mock_get_wav.return_value = b'wav_data'
        mock_sf.read.return_value = (b'audio_array', 24000)
        
        self.client.set_tts_provider('openai')
        self.client.speak("Hello")

        # Verify playback execution
        mock_sd.play.assert_called_with(b'audio_array', 24000)
        mock_sd.wait.assert_called()

if __name__ == '__main__':
    unittest.main()
