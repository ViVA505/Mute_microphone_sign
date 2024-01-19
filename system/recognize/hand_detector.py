import cv2
import mediapipe as mp
import time
from typing import (Optional,
                    Tuple, Callable)
import numpy as np
class HandGestureDetector:
    def __init__(self) -> None:
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_draw = mp.solutions.drawing_utils
        self.last_v_sign_time = 0
        self.last_one_finger_up_time = 0
        self.gesture_hold_threshold = 1.0  # время в секундах

    def process_frame(self, image: np.ndarray) -> Tuple[np.ndarray, Optional[mp.solutions.hands.HandLandmark]]:
        image_rgb = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        hand_landmarks_detected = None
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                hand_landmarks_detected = hand_landmarks
                break  # Принимаем первое обнаруженное изображение руки

        return image, hand_landmarks_detected

    def is_gesture_detected(self, hand_landmarks: mp.solutions.hands.HandLandmark, gesture_check_func: Callable) -> bool:
        if gesture_check_func(hand_landmarks):
            current_time = time.time()
            if current_time - getattr(self, gesture_check_func.__name__ + '_time') > self.gesture_hold_threshold:
                setattr(self, gesture_check_func.__name__ + '_time', current_time)
                return True
        return False
    def is_v_sign(self, hand_landmarks) -> bool:
        index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]

        index_finger_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        middle_finger_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

        # Проверяем, что указательный и средний пальцы подняты
        is_index_up = index_finger_tip.y < index_finger_mcp.y
        is_middle_up = middle_finger_tip.y < middle_finger_mcp.y

        # Проверяем, что безымянный и мизинец опущены
        is_ring_down = ring_finger_tip.y > middle_finger_mcp.y
        is_pinky_down = pinky_tip.y > middle_finger_mcp.y

        current_time = time.time()
        # Условие для жеста ✌️: указательный и средний подняты, остальные опущены
        if is_index_up and is_middle_up and is_ring_down and is_pinky_down:
            if current_time - self.last_v_sign_time > self.gesture_hold_threshold:
                self.last_v_sign_time = current_time
                return True
        return False

    def is_one_finger_up(self, hand_landmarks) -> bool:
        # Получение координат кончиков и MCP пальцев
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]

        index_finger_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        middle_finger_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        ring_finger_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_MCP]
        pinky_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP]

        # Проверка, что только указательный палец поднят
        is_index_up = index_finger_tip.y < index_finger_mcp.y
        is_middle_down = middle_finger_tip.y > middle_finger_mcp.y
        is_ring_down = ring_finger_tip.y > ring_finger_mcp.y
        is_pinky_down = pinky_tip.y > pinky_mcp.y

        current_time = time.time()
        if is_index_up and is_middle_down and is_ring_down and is_pinky_down:
            if current_time - self.last_one_finger_up_time > self.gesture_hold_threshold:
                self.last_one_finger_up_time = current_time
                return True
        return False
