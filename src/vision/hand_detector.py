"""Reusable hand detection and landmark extraction utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class DetectedHand:
    """Landmarks and handedness information for one detected hand."""

    landmarks: np.ndarray
    handedness: Optional[str] = None
    confidence: Optional[float] = None


class HandDetector:
    """Detect hands and extract their 21 normalized landmarks."""

    def __init__(
        self,
        static_image_mode: bool = True,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._drawer = mp.solutions.drawing_utils
        self._styles = mp.solutions.drawing_styles

    def detect(self, image_bgr: np.ndarray) -> List[DetectedHand]:
        """Detect hands in a BGR OpenCV image."""
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("image_bgr must contain a valid image")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self._hands.process(image_rgb)
        detected: List[DetectedHand] = []

        if not results.multi_hand_landmarks:
            return detected

        handedness_list = results.multi_handedness or []
        for index, hand_landmarks in enumerate(results.multi_hand_landmarks):
            points = np.array(
                [[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark],
                dtype=np.float32,
            )
            handedness = None
            confidence = None
            if index < len(handedness_list):
                classification = handedness_list[index].classification[0]
                handedness = classification.label
                confidence = float(classification.score)
            detected.append(DetectedHand(points, handedness, confidence))

        return detected

    def draw_landmarks(self, image_bgr: np.ndarray, results) -> np.ndarray:
        """Draw MediaPipe landmarks onto a BGR image using raw process results."""
        output = image_bgr.copy()
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self._drawer.draw_landmarks(
                    output,
                    hand_landmarks,
                    self._mp_hands.HAND_CONNECTIONS,
                    self._styles.get_default_hand_landmarks_style(),
                    self._styles.get_default_hand_connections_style(),
                )
        return output

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._hands.close()

    def __enter__(self) -> "HandDetector":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
