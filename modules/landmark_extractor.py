"""
Ekstraksi landmark tangan + pose dari tiap frame kamera menggunakan MediaPipe.
Output-nya adalah vektor angka (bukan gambar) yang jadi input ke sequence classifier.
"""
import numpy as np
import mediapipe as mp


class LandmarkExtractor:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        # Holistic = gabungan pose + hand landmark sekaligus, lebih efisien
        # daripada load model hands + pose terpisah di device terbatas.
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0,  # 0 = paling ringan, cocok untuk ARM/STB
        )
        self.mp_draw = mp.solutions.drawing_utils

    def extract(self, frame_bgr):
        """
        Return:
            frame_annotated: frame dengan landmark digambar (untuk preview/stream)
            feature_vector: np.array 1D berisi koordinat landmark (untuk sequence model)
            has_hands: bool, apakah ada tangan terdeteksi di frame ini
        """
        rgb = frame_bgr[:, :, ::-1]  # BGR -> RGB tanpa dependensi cv2 di sini
        results = self.holistic.process(rgb)

        frame_annotated = frame_bgr.copy()
        if results.left_hand_landmarks:
            self.mp_draw.draw_landmarks(
                frame_annotated, results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
            )
        if results.right_hand_landmarks:
            self.mp_draw.draw_landmarks(
                frame_annotated, results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
            )
        if results.pose_landmarks:
            self.mp_draw.draw_landmarks(
                frame_annotated, results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
            )

        feature_vector = self._to_vector(results)
        has_hands = bool(results.left_hand_landmarks or results.right_hand_landmarks)

        return frame_annotated, feature_vector, has_hands

    def _to_vector(self, results):
        """Susun landmark jadi satu vektor tetap panjangnya, isi 0 kalau tidak terdeteksi."""
        def landmarks_to_array(landmarks, n_points):
            if landmarks is None:
                return np.zeros(n_points * 3)
            return np.array([[p.x, p.y, p.z] for p in landmarks.landmark]).flatten()

        pose = landmarks_to_array(results.pose_landmarks, 33)
        left_hand = landmarks_to_array(results.left_hand_landmarks, 21)
        right_hand = landmarks_to_array(results.right_hand_landmarks, 21)

        return np.concatenate([pose, left_hand, right_hand])  # total 33*3 + 21*3 + 21*3 = 225

    def close(self):
        self.holistic.close()
