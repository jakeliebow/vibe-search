import os
import yt_dlp
from typing import List, Dict, Any, Optional, Tuple


def download_video(url: str, output_path: str = '.') -> Tuple[bool, str]:
    """
    Downloads a video from a given URL using yt-dlp.

    Args:
        url: The URL of the video to download (can be full URL or just video ID).
        output_path: The directory to save the video in.

    Returns:
        A tuple containing:
        - bool: True if download was successful, False otherwise.
        - str: Success message with video title, or error message if failed.
    """
    
    # Handle YouTube video ID format (convert to full URL if needed)
    if not url.startswith(('http://', 'https://')):
        # Assume it's a YouTube video ID
        url = f"https://www.youtube.com/watch?v={url}"
    
    output_template = os.path.join(output_path, f'test.%(ext)s')

    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': output_template,
        'quiet': True,
        'no_warnings': True,
        'merge_output_format': 'mp4',
        'noplaylist': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if not info:
                raise ValueError(f"Failed to extract info for video: {url}")
            title = info.get('title')
        return True, f"Successfully downloaded: {title}"
    except Exception as e:
        return False, f"Failed to download {url}: {e}"
