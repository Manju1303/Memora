"""
Speech-to-Text Module
Uses OpenAI Whisper for voice input transcription.
"""
import os
import tempfile
from typing import Optional
import numpy as np

import sys
sys.path.append('..')
from config import WHISPER_MODEL_SIZE, AUDIO_SAMPLE_RATE, AUDIO_CHANNELS


class SpeechToText:
    """
    Speech-to-Text using OpenAI Whisper.
    
    Algorithm: Encoder-Decoder Neural Network
    - Audio is converted to mel spectrogram
    - Encoder processes audio features
    - Decoder generates text tokens
    
    Features:
    - Works offline (local model)
    - Supports multiple languages
    - High accuracy transcription
    """
    
    def __init__(self, model_size: str = WHISPER_MODEL_SIZE):
        self.model_size = model_size
        self.model = None
        self._is_loaded = False
    
    def load_model(self) -> bool:
        """
        Load the Whisper model.
        Called lazily on first use to avoid slow startup.
        """
        if self._is_loaded:
            return True
        
        try:
            import whisper
            print(f"Loading Whisper model: {self.model_size}...")
            self.model = whisper.load_model(self.model_size)
            self._is_loaded = True
            print("Whisper model loaded successfully!")
            return True
        except ImportError:
            print("Whisper not installed. Run: pip install openai-whisper")
            return False
        except Exception as e:
            print(f"Error loading Whisper: {e}")
            return False
    
    def transcribe(self, audio_data: np.ndarray, sample_rate: int = AUDIO_SAMPLE_RATE) -> str:
        """
        Transcribe audio data to text.
        
        Args:
            audio_data: NumPy array of audio samples
            sample_rate: Audio sample rate (default 16000 Hz)
        
        Returns:
            Transcribed text
        """
        if not self._is_loaded:
            if not self.load_model():
                return "Error: Whisper model not available"
        
        try:
            # Ensure audio is float32 and normalized
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Normalize if needed
            if audio_data.max() > 1.0:
                audio_data = audio_data / 32768.0
            
            # Transcribe
            result = self.model.transcribe(audio_data, fp16=False)
            return result["text"].strip()
        except Exception as e:
            print(f"Transcription error: {e}")
            return f"Error transcribing audio: {e}"
    
    def transcribe_file(self, audio_path: str) -> str:
        """
        Transcribe an audio file to text.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Transcribed text
        """
        if not self._is_loaded:
            if not self.load_model():
                return "Error: Whisper model not available"
        
        try:
            result = self.model.transcribe(audio_path, fp16=False)
            return result["text"].strip()
        except Exception as e:
            print(f"Transcription error: {e}")
            return f"Error transcribing file: {e}"
    
    def is_ready(self) -> bool:
        """Check if STT is ready (model loaded)."""
        return self._is_loaded
    
    def get_available_models(self) -> list:
        """Get list of available Whisper model sizes."""
        return ["tiny", "base", "small", "medium", "large"]
    
    def __repr__(self) -> str:
        status = "loaded" if self._is_loaded else "not loaded"
        return f"SpeechToText(model={self.model_size}, status={status})"


def record_audio(duration: float = 5.0, sample_rate: int = AUDIO_SAMPLE_RATE) -> Optional[np.ndarray]:
    """
    Record audio from microphone.
    
    Args:
        duration: Recording duration in seconds
        sample_rate: Audio sample rate
    
    Returns:
        NumPy array of audio data, or None if recording failed
    """
    try:
        import sounddevice as sd
        
        print(f"Recording for {duration} seconds...")
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype=np.float32
        )
        sd.wait()  # Wait for recording to complete
        print("Recording complete!")
        
        return audio.flatten()
    except ImportError:
        print("sounddevice not installed. Run: pip install sounddevice")
        return None
    except Exception as e:
        print(f"Recording error: {e}")
        return None


def record_until_silence(
    silence_threshold: float = 0.01,
    silence_duration: float = 1.5,
    max_duration: float = 30.0,
    sample_rate: int = AUDIO_SAMPLE_RATE
) -> Optional[np.ndarray]:
    """
    Record audio until silence is detected.
    
    Args:
        silence_threshold: RMS threshold for silence
        silence_duration: How long silence must last to stop
        max_duration: Maximum recording duration
        sample_rate: Audio sample rate
    
    Returns:
        NumPy array of audio data
    """
    try:
        import sounddevice as sd
        
        print("Recording... (speak now, recording stops after silence)")
        
        chunk_duration = 0.1  # 100ms chunks
        chunk_samples = int(chunk_duration * sample_rate)
        
        audio_chunks = []
        silent_chunks = 0
        silent_chunks_needed = int(silence_duration / chunk_duration)
        max_chunks = int(max_duration / chunk_duration)
        
        for _ in range(max_chunks):
            chunk = sd.rec(chunk_samples, samplerate=sample_rate, channels=1, dtype=np.float32)
            sd.wait()
            audio_chunks.append(chunk.flatten())
            
            # Check for silence
            rms = np.sqrt(np.mean(chunk**2))
            if rms < silence_threshold:
                silent_chunks += 1
                if silent_chunks >= silent_chunks_needed:
                    break
            else:
                silent_chunks = 0
        
        print("Recording complete!")
        return np.concatenate(audio_chunks)
    except Exception as e:
        print(f"Recording error: {e}")
        return None
