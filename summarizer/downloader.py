import tempfile
from pathlib import Path
from typing import Optional

import yt_dlp


def download_audio(url: str, output_dir: Optional[Path] = None) -> Path:
    """
    Download audio from YouTube video.

    Args:
        url: YouTube video URL
        output_dir: Directory to save audio (default: temp directory)

    Returns:
        Path to downloaded audio file

    Raises:
        Exception: If download fails
    """
    if output_dir is None:
        output_dir = Path(tempfile.gettempdir()) / "yt_summarizer"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure yt-dlp options
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': str(output_dir / '%(id)s.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_id = info['id']
            audio_file = output_dir / f"{video_id}.mp3"

            if not audio_file.exists():
                raise FileNotFoundError(f"Downloaded audio not found: {audio_file}")

            return audio_file

    except Exception as e:
        raise Exception(f"Failed to download audio: {str(e)}")
