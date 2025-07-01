import logging
import os
import subprocess
import json
import whisper
import yt_dlp
from yt_dlp.utils import DownloadError

# This will get the logger that was configured in mcp_server.py
logger = logging.getLogger(__name__)

# A simple cache for the Whisper model so we don't reload it every time
_whisper_model = None

def _check_whisper_cpp():
    """Check if whisper.cpp is available in the system PATH."""
    try:
        result = subprocess.run(['whisper-cpp', '--version'], 
                              capture_output=True, 
                              text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def _transcribe_with_whisper_cpp(audio_path):
    """
    Transcribe audio using whisper.cpp.
    Returns the transcript text if successful, None if failed.
    """
    try:
        logger.info("Attempting transcription with whisper.cpp...")
        result = subprocess.run(
            ['whisper-cpp', audio_path, '--output-json'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            try:
                output = json.loads(result.stdout)
                return output.get('text', '')
            except json.JSONDecodeError:
                logger.error("Failed to parse whisper.cpp JSON output")
                return None
        else:
            logger.error(f"whisper.cpp failed with error: {result.stderr}")
            return None
    except Exception as e:
        logger.error(f"Error running whisper.cpp: {e}")
        return None

def _yt_dlp_hook(d):
    """
    A hook for yt-dlp to capture its progress. We use this to prevent it from
    printing directly to stdout, which would corrupt the JSON-RPC communication.
    """
    if d['status'] == 'finished':
        # The 'filename' key is provided once the download is complete.
        logger.info(f"yt-dlp hook: Finished downloading '{d.get('filename', 'unknown file')}'")
    # We ignore the 'downloading' status to prevent log spam.

def get_youtube_transcript(query: str, force_whisper: bool = False) -> dict:
    """
    Searches for a YouTube video, downloads it, and returns the transcript.
    It will try to get an official transcript first, unless force_whisper is True.
    If force_whisper is True, it will try whisper.cpp first, then fall back to Python whisper.
    """
    global _whisper_model
    logger.info(f"get_youtube_transcript called with query: '{query}', force_whisper: {force_whisper}")

    try:
        # --- 1. Search for the video and get its info ---
        logger.info("Searching for video with yt-dlp...")
        ydl_opts_info = {
            'format': 'bestaudio/best',
            'noplaylist': True,
            'default_search': 'ytsearch',
            'quiet': True,
            'noprogress': True,
            'logger': logger,
        }
        with yt_dlp.YoutubeDL(ydl_opts_info) as ydl:
            info = ydl.extract_info(query, download=False)
            video_info = info['entries'][0] if 'entries' in info else info

        video_url = video_info.get("webpage_url")
        video_title = video_info.get("title", "Unknown Title")
        logger.info(f"Found video: '{video_title}' at {video_url}")

        transcript_text = None
        transcript_source = "Not Available"

        # --- 2. Try to get the official transcript ---
        if not force_whisper:
            logger.info("Official transcript check skipped for this debug version. Proceeding to Whisper.")

        # --- 3. Use Whisper if needed ---
        if transcript_text is None:
            output_dir = os.path.join(os.path.dirname(__file__), "testing", "audio_cache")
            os.makedirs(output_dir, exist_ok=True)
            audio_path = os.path.join(output_dir, f"{video_info.get('id', 'default_id')}.mp3")

            logger.info(f"Downloading audio to: {audio_path}")
            ydl_opts_audio = {
                'format': 'bestaudio/best',
                'outtmpl': audio_path,
                'quiet': True,
                'noprogress': True,
                'overwrites': True,
                'logger': logger,
                'progress_hooks': [_yt_dlp_hook],
            }
            with yt_dlp.YoutubeDL(ydl_opts_audio) as ydl:
                ydl.download([video_url])
            logger.info("Audio download complete.")

            # First try whisper.cpp if available
            if _check_whisper_cpp():
                transcript_text = _transcribe_with_whisper_cpp(audio_path)
                if transcript_text:
                    transcript_source = "whisper.cpp (AI Generated)"
                    logger.info("Successfully transcribed using whisper.cpp")

            # Fall back to Python whisper if whisper.cpp failed or isn't available
            if transcript_text is None:
                logger.info("Falling back to Python whisper...")
                transcript_source = "Python Whisper (AI Generated)"
                
                # Load the Whisper model if it's not already in memory
                if _whisper_model is None:
                    logger.info("Whisper model not loaded. Loading 'tiny' model now...")
                    _whisper_model = whisper.load_model("tiny")
                    logger.info("Whisper model 'tiny' loaded successfully.")

                # Run the transcription
                logger.info("Starting Python Whisper transcription...")
                result = _whisper_model.transcribe(audio_path, fp16=False)
                transcript_text = result["text"]
                logger.info("Python Whisper transcription complete.")

        return {
            "status": "success",
            "title": video_title,
            "url": video_url,
            "source": transcript_source,
            "transcript": transcript_text
        }

    except DownloadError as e:
        logger.error(f"yt-dlp download error: {e}", exc_info=True)
        return {"status": "error", "message": f"Could not find or download the video: {e}"}
    except Exception as e:
        logger.error(f"An unexpected error occurred in get_youtube_transcript: {e}", exc_info=True)
        return {"status": "error", "message": f"An unexpected error occurred: {str(e)}"}
