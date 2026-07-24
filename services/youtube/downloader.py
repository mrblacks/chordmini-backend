import subprocess
import tempfile
import os


def download_youtube_audio(youtube_url: str, output_path: str) -> None:
    result = subprocess.run(
        ["yt-dlp", "-x", "--audio-format", "mp3", "--audio-quality", "0",
         "-o", output_path, "--no-playlist", youtube_url],
        capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp 실패: {result.stderr[-300:]}")
