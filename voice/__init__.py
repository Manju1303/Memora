"""
Voice module for AI Assistant
- Speech-to-Text (Whisper)
- Text-to-Speech (pyttsx3)
"""
from .speech_to_text import SpeechToText
from .text_to_speech import TextToSpeech

__all__ = ["SpeechToText", "TextToSpeech"]
