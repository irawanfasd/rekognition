"""
Text-to-Speech: mengubah kata/kalimat hasil deteksi jadi suara,
supaya khalayak di sekitar penyandang disabilitas bisa "mendengar" isyaratnya.
"""
import threading

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False


class TTSHandler:
    def __init__(self, lang="id"):
        self.lang = lang
        self.enabled = PYTTSX3_AVAILABLE
        if not self.enabled:
            print("[TTSHandler] pyttsx3 tidak terpasang -> TTS dimatikan sementara. "
                  "Jalankan: pip3 install pyttsx3")

    def speak(self, text):
        """Bicara di thread terpisah supaya tidak nge-block loop deteksi kamera."""
        if not self.enabled or not text:
            return
        threading.Thread(target=self._speak_blocking, args=(text,), daemon=True).start()

    def _speak_blocking(self, text):
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            print(f"[TTSHandler] Gagal bicara: {e}")
