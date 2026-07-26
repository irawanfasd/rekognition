"""
Detektor Bahasa Isyarat - Server Utama
Jalan di Armbian STB. Kamera -> landmark -> sequence model -> kata -> TTS + histori.
Frontend (di /static) menampilkan stream video + teks hasil terjemahan + histori,
diakses lewat browser di jaringan lokal, atau di-embed ke lms moodletest.xyz/detektor.
"""
import threading
import time

import cv2
from flask import Flask, Response, jsonify, render_template

import config
from modules.landmark_extractor import LandmarkExtractor
from modules.sequence_classifier import SequenceClassifier
from modules.db_handler import DBHandler
from modules.tts_handler import TTSHandler

app = Flask(__name__)

extractor = LandmarkExtractor()
classifier = SequenceClassifier(
    model_path=config.MODEL_PATH,
    vocab_path=config.VOCAB_PATH,
    sequence_length=config.SEQUENCE_LENGTH,
    confidence_threshold=config.CONFIDENCE_THRESHOLD,
)
db = DBHandler(config.DB_PATH)
tts = TTSHandler(lang=config.TTS_LANG) if config.ENABLE_TTS else None

state_lock = threading.Lock()
state = {
    "latest_frame": None,      # bytes JPEG untuk streaming
    "latest_text": "",         # kata/kalimat terakhir terdeteksi
    "latest_confidence": 0.0,
    "hands_detected": False,
}

# Cooldown supaya kata yang sama tidak diucapkan berulang tiap frame
LAST_SPOKEN = {"text": "", "time": 0}
SPEAK_COOLDOWN_SEC = 2.0


def camera_loop():
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)

    if not cap.isOpened():
        print("[camera_loop] Kamera tidak bisa dibuka. Cek koneksi/permission kamera di STB.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        annotated, feature_vector, has_hands = extractor.extract(frame)

        if has_hands:
            classifier.add_frame(feature_vector)
        else:
            # tidak ada tangan -> reset buffer supaya sequence tidak tercampur
            classifier.reset()

        label, confidence = classifier.predict()
        if label:
            handle_detection(label, confidence)

        ok, jpeg = cv2.imencode(".jpg", annotated)
        if ok:
            with state_lock:
                state["latest_frame"] = jpeg.tobytes()
                state["hands_detected"] = has_hands

        time.sleep(1.0 / config.CAMERA_FPS)

    cap.release()


def handle_detection(label, confidence):
    with state_lock:
        state["latest_text"] = label
        state["latest_confidence"] = confidence

    now = time.time()
    if label != LAST_SPOKEN["text"] or (now - LAST_SPOKEN["time"]) > SPEAK_COOLDOWN_SEC:
        LAST_SPOKEN["text"] = label
        LAST_SPOKEN["time"] = now

        db.add_entry(label, confidence)
        if tts:
            tts.speak(label)

    classifier.reset()  # mulai buffer baru untuk gerakan berikutnya


def mjpeg_generator():
    while True:
        with state_lock:
            frame = state["latest_frame"]
        if frame is not None:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(1.0 / config.CAMERA_FPS)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/current")
def api_current():
    with state_lock:
        return jsonify({
            "text": state["latest_text"],
            "confidence": state["latest_confidence"],
            "hands_detected": state["hands_detected"],
        })


@app.route("/api/history")
def api_history():
    return jsonify(db.get_recent(limit=50))


if __name__ == "__main__":
    cam_thread = threading.Thread(target=camera_loop, daemon=True)
    cam_thread.start()
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG, threaded=True)
