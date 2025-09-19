import os
import yt_dlp
from typing import List, Dict, Any, Optional, Tuple

def download_video(url: str, output_path: str = ".") -> str:
    import os, yt_dlp
    if not url.startswith(("http://", "https://")):
        url = f"https://www.youtube.com/watch?v={url}"

    ydl_opts = {
        # Only pick MP4-compatible codecs; fall back to progressive MP4
        "format": '(bv*[vcodec~="^avc1|^h264|^hev1|^h265"]+ba[acodec~="^mp4a|^aac"])/b[ext=mp4]',
        "outtmpl": os.path.join(output_path, "test.%(ext)s"),
        "noplaylist": True,
        "prefer_ffmpeg": True,
        "merge_output_format": "mp4",
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        rd = info.get("requested_downloads") or []
        fp = (rd[0].get("filepath") if rd else info.get("filepath")) or ydl.prepare_filename(info)
        fp = os.path.abspath(fp)
        
    return fp


