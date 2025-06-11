from langchain_core.tools import tool
import os
import base64
import uuid
import tempfile
from typing import Dict, Any, Optional
from openai import OpenAI
from pydub import AudioSegment

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Helper functions for audio processing


def decode_audio(base64_string: str, file_extension: str = "mp3") -> str:
    """Convert a base64 string to an audio file and return the path."""
    audio_data = base64.b64decode(base64_string)
    temp_dir = tempfile.gettempdir()
    audio_id = str(uuid.uuid4())
    audio_path = os.path.join(temp_dir, f"{audio_id}.{file_extension}")

    with open(audio_path, "wb") as audio_file:
        audio_file.write(audio_data)

    return audio_path


def convert_to_supported_format(audio_path: str) -> str:
    """Convert audio to a format supported by gpt-4o-transcribe (mp3, mp4, mpeg, mpga, m4a, wav, webm)."""
    supported_formats = ['mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'wav', 'webm']
    file_extension = audio_path.split('.')[-1].lower()

    if file_extension in supported_formats:
        return audio_path

    # Convert to wav format if not supported
    try:
        audio = AudioSegment.from_file(audio_path)
        temp_dir = tempfile.gettempdir()
        audio_id = str(uuid.uuid4())
        converted_path = os.path.join(temp_dir, f"{audio_id}.wav")
        audio.export(converted_path, format="wav")
        return converted_path
    except Exception as e:
        raise ValueError(f"无法转换音频格式: {str(e)}")


@tool
def transcribe_audio(audio_base64: str, language: Optional[str] = None) -> str:
    """
    Transcribes audio to text using OpenAI gpt-4o-transcribe.

    Args:
        audio_base64 (str): Base64 encoded audio string.
        language (str, optional): Language code of the audio (e.g., 'zh', 'en').

    Returns:
        The transcribed text string.
    """
    try:
        # Decode audio from base64
        audio_path = decode_audio(audio_base64)

        # Convert to supported format if necessary
        converted_path = convert_to_supported_format(audio_path)

        # Transcribe audio using OpenAI gpt-4o-transcribe
        with open(converted_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model='gpt-4o-transcribe',
                file=audio_file,
            )

        # Clean up temporary files
        if os.path.exists(audio_path):
            os.remove(audio_path)
        if converted_path != audio_path and os.path.exists(converted_path):
            os.remove(converted_path)

        return transcript.text

    except Exception as e:
        return f"音频转录失败: {str(e)}"
