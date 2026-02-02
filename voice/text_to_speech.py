"""
Text-to-Speech Module
Uses pyttsx3 for offline voice output.
"""
import threading
from typing import Optional
import sys
sys.path.append('..')
from config import TTS_RATE


class TextToSpeech:
    """
    Text-to-Speech using pyttsx3.
    
    Algorithm: Neural TTS / Platform-native synthesis
    - Text is analyzed for pronunciation
    - Speech is synthesized using system voices
    - Audio is played through speakers
    
    Features:
    - Works completely offline
    - Uses system's native TTS engine
    - Supports multiple voices and rates
    """
    
    def __init__(self, rate: int = TTS_RATE):
        self.rate = rate
        self.engine = None
        self._is_initialized = False
        self._is_speaking = False
    
    def _initialize(self) -> bool:
        """Initialize the TTS engine."""
        if self._is_initialized:
            return True
        
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', self.rate)
            self._is_initialized = True
            return True
        except ImportError:
            print("pyttsx3 not installed. Run: pip install pyttsx3")
            return False
        except Exception as e:
            print(f"TTS initialization error: {e}")
            return False
    
    def speak(self, text: str, wait: bool = True) -> bool:
        """
        Convert text to speech and play it.
        
        Args:
            text: Text to speak
            wait: If True, block until speech is complete
        
        Returns:
            True if speech started successfully
        """
        if not self._initialize():
            return False
        
        try:
            self._is_speaking = True
            self.engine.say(text)
            
            if wait:
                self.engine.runAndWait()
                self._is_speaking = False
            else:
                # Run in background thread
                thread = threading.Thread(target=self._run_and_wait)
                thread.daemon = True
                thread.start()
            
            return True
        except Exception as e:
            print(f"TTS error: {e}")
            self._is_speaking = False
            return False
    
    def _run_and_wait(self):
        """Helper for async speech."""
        try:
            self.engine.runAndWait()
        finally:
            self._is_speaking = False
    
    def speak_async(self, text: str) -> bool:
        """Speak text without blocking."""
        return self.speak(text, wait=False)
    
    def stop(self) -> None:
        """Stop current speech."""
        if self._is_initialized and self.engine:
            try:
                self.engine.stop()
            except:
                pass
        self._is_speaking = False
    
    def set_rate(self, rate: int) -> None:
        """Set speech rate (words per minute)."""
        self.rate = rate
        if self._is_initialized and self.engine:
            self.engine.setProperty('rate', rate)
    
    def set_volume(self, volume: float) -> None:
        """Set volume (0.0 to 1.0)."""
        if self._is_initialized and self.engine:
            self.engine.setProperty('volume', max(0.0, min(1.0, volume)))
    
    def get_voices(self) -> list:
        """Get available voices."""
        if not self._initialize():
            return []
        
        try:
            voices = self.engine.getProperty('voices')
            return [{"id": v.id, "name": v.name} for v in voices]
        except:
            return []
    
    def set_voice(self, voice_id: str) -> bool:
        """Set the voice by ID."""
        if not self._initialize():
            return False
        
        try:
            self.engine.setProperty('voice', voice_id)
            return True
        except:
            return False
    
    def is_speaking(self) -> bool:
        """Check if currently speaking."""
        return self._is_speaking
    
    def is_ready(self) -> bool:
        """Check if TTS is ready."""
        return self._initialize()
    
    def save_to_file(self, text: str, filename: str) -> bool:
        """
        Save speech to audio file.
        
        Args:
            text: Text to convert
            filename: Output file path
        
        Returns:
            True if saved successfully
        """
        if not self._initialize():
            return False
        
        try:
            self.engine.save_to_file(text, filename)
            self.engine.runAndWait()
            return True
        except Exception as e:
            print(f"Save to file error: {e}")
            return False
    
    def __repr__(self) -> str:
        status = "ready" if self._is_initialized else "not initialized"
        return f"TextToSpeech(rate={self.rate}, status={status})"


# Convenience function
def speak(text: str, wait: bool = True) -> bool:
    """Quick function to speak text."""
    tts = TextToSpeech()
    return tts.speak(text, wait=wait)
