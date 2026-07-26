"""
Klasifikasi gerakan berurutan (sequence) jadi kata/frasa.

Berbeda dari deteksi gesture statis (1 pose = 1 label), bahasa isyarat kalimat
sehari-hari itu dinamis: kita kumpulkan landmark selama beberapa frame (SEQUENCE_LENGTH),
baru dilempar ke model LSTM/GRU yang mengenali pola gerakan dari waktu ke waktu.

CATATAN PENTING:
Model (models/sign_lstm.tflite) dan kamus kata (data/vocab.json) BELUM ada di skeleton ini
karena butuh proses training terpisah pakai dataset BISINDO tingkat kata (lihat README).
Sebelum model tersedia, kelas ini otomatis jalan di MOCK MODE supaya server tetap bisa
dites end-to-end (kamera -> landmark -> "kata" placeholder -> TTS -> histori).
"""
import json
import os
from collections import deque

import numpy as np

try:
    import tflite_runtime.interpreter as tflite
    TFLITE_AVAILABLE = True
except ImportError:
    try:
        import tensorflow.lite as tflite
        TFLITE_AVAILABLE = True
    except ImportError:
        TFLITE_AVAILABLE = False


class SequenceClassifier:
    def __init__(self, model_path, vocab_path, sequence_length, confidence_threshold=0.7):
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        self.buffer = deque(maxlen=sequence_length)

        self.vocab = self._load_vocab(vocab_path)
        self.interpreter = self._load_model(model_path)
        self.mock_mode = self.interpreter is None

        if self.mock_mode:
            print("[SequenceClassifier] Model tidak ditemukan -> jalan di MOCK MODE. "
                  "Lihat README untuk cara melatih & memasang model asli.")

    def _load_vocab(self, vocab_path):
        if os.path.exists(vocab_path):
            with open(vocab_path, "r", encoding="utf-8") as f:
                return json.load(f)
        # kamus default sementara, ganti setelah punya dataset sendiri
        return {
            "0": "halo",
            "1": "terima kasih",
            "2": "tolong",
            "3": "sakit",
            "4": "makan",
        }

    def _load_model(self, model_path):
        if not TFLITE_AVAILABLE or not os.path.exists(model_path):
            return None
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter

    def add_frame(self, feature_vector):
        """Tambahkan satu frame landmark ke buffer. Panggil ini tiap frame kamera."""
        self.buffer.append(feature_vector)

    def is_ready(self):
        return len(self.buffer) == self.sequence_length

    def predict(self):
        """
        Jalankan prediksi atas buffer sequence saat ini.
        Return (label:str, confidence:float) atau (None, 0.0) kalau belum yakin/belum siap.
        """
        if not self.is_ready():
            return None, 0.0

        sequence = np.array(self.buffer, dtype=np.float32)  # shape: (sequence_length, n_features)

        if self.mock_mode:
            return self._mock_predict(sequence)

        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        self.interpreter.set_tensor(input_details[0]["index"], sequence[np.newaxis, ...])
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(output_details[0]["index"])[0]

        idx = int(np.argmax(output))
        confidence = float(output[idx])

        if confidence < self.confidence_threshold:
            return None, confidence

        label = self.vocab.get(str(idx), "?")
        return label, confidence

    def _mock_predict(self, sequence):
        """Placeholder: pilih kata acak berbobot berdasarkan seberapa banyak gerakan terdeteksi,
        supaya demo tetap terasa 'hidup' sebelum model asli terpasang."""
        motion = float(np.std(sequence))
        if motion < 0.01:
            return None, 0.0
        idx = str(hash(round(motion, 2)) % len(self.vocab))
        return self.vocab.get(idx, "?"), 0.75

    def reset(self):
        self.buffer.clear()
