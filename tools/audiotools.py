from langchain_core.tools import tool
import os
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


@tool
def transcribe_audio(file_path: str) -> str:
    """
    Transcribe audio file to text using OpenAI gpt-4o-transcribe.
    Args:
        file_path (str): the path to the audio file.
    Returns:
        The transcribed text string.
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            return f"Error: file not found - {file_path}"

        # Check if file is supported by gpt-4o-transcribe
        supported_extensions = ['.mp3', '.mp4',
                                '.mpeg', '.mpga', '.m4a', '.wav', '.webm']
        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext not in supported_extensions:
            return f"Error: unsupported audio format - {file_ext}. Supported formats: {', '.join(supported_extensions)}"

        # Transcribe audio using OpenAI gpt-4o-transcribe
        with open(file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=audio_file,
            )

        return transcript.text

    except Exception as e:
        return f"Error: audio transcription failed: {str(e)}"
