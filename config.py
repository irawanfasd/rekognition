"""
Konfigurasi utama sistem Detektor Bahasa Isyarat
Dioptimasi untuk ARM device (Armbian STB) dengan RAM terbatas
"""
import os

# ---- Optimasi ARM / Low-RAM ----
os.environ["OPENCV_OPENCL_RUNTIME"] = ""     # matikan OpenCL (STB biasanya tidak punya GPU CV)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"     # kurangi log TensorFlow

# ---- Kamera ----
CAMERA_INDEX = 0
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
CAMERA_FPS = 10

# ---- Sequence model (untuk kalimat sehari-hari, bukan alfabet statis) ----
SEQUENCE_LENGTH = 30          # jumlah frame per satu "kata/gerakan" (±1-2 detik @ 15fps)
LANDMARK_FEATURES = 225        # pose (33x3) + tangan kiri (21x3) + tangan kanan (21x3)
MODEL_PATH = "models/sign_lstm.tflite"
VOCAB_PATH = "data/vocab.json"
CONFIDENCE_THRESHOLD = 0.7

# ---- Database ----
DB_PATH = "history.db"

# ---- Server ----
HOST = "0.0.0.0"
PORT = 5000
DEBUG = False

# ---- Fitur ----
ENABLE_TTS = True
TTS_LANG = "id"        # Bahasa Indonesia
