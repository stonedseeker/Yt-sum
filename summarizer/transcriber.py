from pathlib import Path
from typing import Optional

from faster_whisper import WhisperModel


class Transcriber:
    """Offline speech-to-text transcriber using Faster Whisper with CTranslate2."""
    
    def __init__(self, model_size: str = "base", device: str = "cpu"):
        """
        Initialize transcriber with specified model.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            device: Device to run on (cpu or cuda)
        """
        self.model_size = model_size
        self.device = device
        self.model: Optional[WhisperModel] = None
    
    def _load_model(self) -> None:
        """Lazy load the Whisper model (downloads on first use)."""
        if self.model is None:
            print(f"📥 Loading Whisper '{self.model_size}' model (first run may download ~140MB)...")
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type="int8"  # Quantized for faster inference
            )
            print("✅ Model loaded successfully")
    
    def transcribe(self, audio_path: Path, language: Optional[str] = None) -> str:
        """
        Transcribe audio file to text.
        
        Args:
            audio_path: Path to audio file
            language: Optional language code (e.g., 'en', 'es')
        
        Returns:
            Transcribed text
        """
        self._load_model()
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Check file size to warn about long videos
        file_size_mb = audio_path.stat().st_size / (1024 * 1024)
        if file_size_mb > 50:  # ~50+ minutes
            print(f"   ⚠️  Large audio file ({file_size_mb:.1f}MB) - this may take several minutes...")
        
        # Transcribe with Faster Whisper
        segments, info = self.model.transcribe(
            str(audio_path),
            language=language,
            beam_size=5,
            vad_filter=True,  # Voice activity detection to skip silence
        )
        
        # Combine all segments into full transcript
        transcript = " ".join([segment.text.strip() for segment in segments])
        
        return transcript
    
    def transcribe_with_timestamps(self, audio_path: Path) -> list:
        """
        Transcribe with word-level timestamps.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            List of segments with text, start, and end times
        """
        self._load_model()
        
        segments, _ = self.model.transcribe(str(audio_path), word_timestamps=True)
        
        return [
            {
                "text": segment.text.strip(),
                "start": segment.start,
                "end": segment.end
            }
            for segment in segments
        ]

# """Speech-to-text transcription using Faster Whisper."""

# from pathlib import Path
# from typing import Optional

# from faster_whisper import WhisperModel


# class Transcriber:
#     """Offline speech-to-text transcriber using Faster Whisper with CTranslate2."""

#     def __init__(self, model_size: str = "base", device: str = "cpu"):
#         """
#         Initialize transcriber with specified model.

#         Args:
#             model_size: Whisper model size (tiny, base, small, medium, large)
#             device: Device to run on (cpu or cuda)
#         """
#         self.model_size = model_size
#         self.device = device
#         self.model: Optional[WhisperModel] = None

#     def _load_model(self) -> None:
#         """Lazy load the Whisper model (downloads on first use)."""
#         if self.model is None:
#             print(f"📥 Loading Whisper '{self.model_size}' model (first run may download ~140MB)...")
#             self.model = WhisperModel(
#                 self.model_size,
#                 device=self.device,
#                 compute_type="int8"  # Quantized for faster inference
#             )
#             print("✅ Model loaded successfully")

#     def transcribe(self, audio_path: Path, language: Optional[str] = None) -> str:
#         """
#         Transcribe audio file to text.

#         Args:
#             audio_path: Path to audio file
#             language: Optional language code (e.g., 'en', 'es')

#         Returns:
#             Transcribed text
#         """
#         self._load_model()

#         if not audio_path.exists():
#             raise FileNotFoundError(f"Audio file not found: {audio_path}")

#         # Transcribe with Faster Whisper
#         segments, info = self.model.transcribe(
#             str(audio_path),
#             language=language,
#             beam_size=5,
#             vad_filter=True,  # Voice activity detection to skip silence
#         )

#         # Combine all segments into full transcript
#         transcript = " ".join([segment.text.strip() for segment in segments])

#         return transcript

#     def transcribe_with_timestamps(self, audio_path: Path) -> list[dict[str, any]]:
#         """
#         Transcribe with word-level timestamps.

#         Args:
#             audio_path: Path to audio file

#         Returns:
#             List of segments with text, start, and end times
#         """
#         self._load_model()

#         segments, _ = self.model.transcribe(str(audio_path), word_timestamps=True)

#         return [
#             {
#                 "text": segment.text.strip(),
#                 "start": segment.start,
#                 "end": segment.end
#             }
#             for segment in segments
#         ]
