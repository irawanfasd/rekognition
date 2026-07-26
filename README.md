# Detektor Bahasa Isyarat — Setup di Armbian STB

Sistem ini membaca gerakan bahasa isyarat lewat kamera, mengenalinya (kata/kalimat
sehari-hari, bukan cuma alfabet), lalu menerangkannya ke khalayak sekitar lewat
teks di layar + suara (TTS), dan menyimpan histori percakapan.

## 1. Install dependencies di STB

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-opencv espeak-ng
pip3 install --upgrade pip
pip3 install --break-system-packages -r requirements.txt
```

Catatan ARM:
- `tflite-runtime` kadang tidak punya wheel siap pakai untuk semua varian ARM.
  Kalau gagal install, coba `pip3 install tensorflow-aarch64` sebagai gantinya,
  atau cari wheel `tflite-runtime` yang cocok dengan arsitektur STB kamu (`uname -m`).
- `pyttsx3` di Linux butuh `espeak-ng` terpasang di sistem (sudah termasuk di atas).

## 2. Jalankan servernya

```bash
cd detektor
python3 main.py
```

Server jalan di `http://<ip-stb>:5000`. Buka dari browser HP/laptop di jaringan yang
sama untuk tes kamera + deteksi.

**Status saat ini: MOCK MODE.** Karena belum ada model terlatih (`models/sign_lstm.tflite`),
sistem menampilkan kata acak dari `data/vocab.json` setiap ada gerakan tangan terdeteksi —
ini supaya kamu bisa tes alur kamera → tampilan → suara → histori dulu, sebelum model asli
jadi. Lihat bagian 4 untuk melatih model sungguhan.

## 3. Jalankan otomatis saat STB nyala (systemd)

Buat file `/etc/systemd/system/detektor.service`:

```ini
[Unit]
Description=Detektor Bahasa Isyarat
After=network.target

[Service]
WorkingDirectory=/home/<user>/detektor
ExecStart=/usr/bin/python3 main.py
Restart=always
User=<user>

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable detektor
sudo systemctl start detektor
```

## 4. Melatih model sungguhan (di luar STB, bukan di STB)

STB kamu tidak cukup kuat untuk training. Alurnya:

1. **Kumpulkan dataset**: rekam video untuk tiap kata di `data/vocab.json` (idealnya
   20-30 pengulangan per kata, orang berbeda-beda kalau bisa). Simpan landmark per video
   pakai `LandmarkExtractor` (bisa dijalankan offline di laptop).
2. **Susun jadi sequence**: tiap sample = array shape `(SEQUENCE_LENGTH, 225)`.
3. **Latih model LSTM/GRU** (di Google Colab / laptop, pakai TensorFlow/Keras):
   ```python
   model = tf.keras.Sequential([
       tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(30, 225)),
       tf.keras.layers.LSTM(32),
       tf.keras.layers.Dense(len(vocab), activation="softmax"),
   ])
   ```
4. **Convert ke TFLite**:
   ```python
   converter = tf.lite.TFLiteConverter.from_keras_model(model)
   tflite_model = converter.convert()
   open("sign_lstm.tflite", "wb").write(tflite_model)
   ```
5. Copy `sign_lstm.tflite` ke `models/` di STB, update `data/vocab.json` sesuai urutan
   label saat training. Restart server — otomatis keluar dari MOCK MODE.

Kalau belum sempat rekam dataset sendiri, cari dataset publik BISINDO/SIBI tingkat kata
di Kaggle/GitHub untuk starting point.

## 5. Deploy ke lms moodletest.xyz/detektor

Karena ini aplikasi Flask (bukan file statis), ada 2 opsi:

**A. Reverse proxy (disarankan)** — kalau web Moodle kamu di server yang sama/terjangkau,
setup Nginx reverse proxy supaya path `/detektor` meneruskan ke Flask yang jalan di STB:
```nginx
location /detektor/ {
    proxy_pass http://<ip-stb>:5000/;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
}
```

**B. Iframe embed** — kalau STB dan server Moodle beda jaringan, buat halaman di Moodle
yang isinya `<iframe>` mengarah ke `http://<ip-publik-stb>:5000` (perlu port forwarding/
dynamic DNS kalau STB di jaringan rumah).

## Struktur proyek

```
detektor/
├── main.py                    # Server Flask + camera loop
├── config.py                  # Semua pengaturan
├── modules/
│   ├── landmark_extractor.py  # MediaPipe: kamera -> landmark
│   ├── sequence_classifier.py # Landmark sequence -> kata (LSTM/TFLite)
│   ├── db_handler.py          # SQLite: histori percakapan
│   └── tts_handler.py         # Kata -> suara
├── models/sign_lstm.tflite    # Model hasil training (belum ada, lihat bagian 4)
├── data/vocab.json            # Kamus kata
├── templates/index.html       # Tampilan web (stream + histori)
└── history.db                 # Dibuat otomatis saat server pertama jalan
```
