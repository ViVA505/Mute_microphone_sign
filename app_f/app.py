import cv2
import numpy as np
import time
import json
from typing import Set

from system.micro.micro_trigger import MicrophoneController
from system.recognize.hand_detector import HandGestureDetector


class Application:
    def __init__(self) -> None:
        self.detector = HandGestureDetector()
        self.mic_controller = MicrophoneController()
        self.cap = cv2.VideoCapture(0)
        self.gesture_options = ['✌️', '☝️']
        self.gesture_images = {}
        self.load_gesture_images()
        self.mute_gestures: Set[str] = set()
        self.unmute_gestures: Set[str] = set()
        self.load_gesture_settings()


    def load_gesture_images(self) -> None:
        self.gesture_images = {
            '✌️': cv2.imread('app_f/images/is_v_sign.png', cv2.IMREAD_COLOR),
            '☝️': cv2.imread('app_f/images/up-pointing.png', cv2.IMREAD_COLOR),
        }


        for gesture, image in self.gesture_images.items():
            if image is None:
                print(f"Warning: Unable to load image for gesture {gesture}")
            else:

                self.gesture_images[gesture] = cv2.resize(image, (50, 50))

    def draw_gesture_buttons(self, img: np.ndarray, y_start: int, gestures: list, selected_gestures: Set[str],
                             section_title: str) -> None:
        title_y_offset = 20
        cv2.putText(img, section_title, (10, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_start += title_y_offset
        for i, gesture in enumerate(gestures):
            x_start = 10
            y_position = y_start + (i * 70)
            color = (0, 255, 0) if gesture in selected_gestures else (200, 200, 200)
            cv2.rectangle(img, (x_start, y_position), (x_start + 60, y_position + 60), color,
                          -1)
            gesture_img = self.gesture_images.get(gesture)
            if gesture_img is not None:
                img[y_position:y_position + gesture_img.shape[0], x_start:x_start + gesture_img.shape[1]] = gesture_img

    def load_gesture_settings(self) -> None:
        try:
            with open("system/configurations/gesture_settings.json", "r") as file:
                settings = json.load(file)
                mute_gesture = settings.get("mute_gesture")
                unmute_gesture = settings.get("unmute_gesture")

                if mute_gesture:
                    self.mute_gestures.add(mute_gesture)
                if unmute_gesture:
                    self.unmute_gestures.add(unmute_gesture)
        except FileNotFoundError:
            print("Gesture settings file not found. Using default settings.")
        except json.JSONDecodeError:
            print("Error decoding gesture settings. Using default settings.")

    def save_gesture_settings(self) -> None:
        mute_gesture = next(iter(self.mute_gestures), None)
        unmute_gesture = next(iter(self.unmute_gestures), None)
        with open("system/configurations/gesture_settings.json", "w") as file:
            file.write(json.dumps({"mute_gesture": mute_gesture, "unmute_gesture": unmute_gesture}))
        print("Gesture settings saved")



    def on_gesture_click(self, event: int, x: int, y: int, flags: int, param: any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            button_height = 70
            title_height = 30
            title_button_space = 20
            section_height = title_height + len(self.gesture_options) * button_height + title_button_space

            mute_section_y_start = title_height
            unmute_section_y_start = section_height + title_button_space

            if mute_section_y_start <= y < mute_section_y_start + len(self.gesture_options) * button_height:
                gesture_index = (y - mute_section_y_start) // button_height
                if 0 <= gesture_index < len(self.gesture_options):
                    gesture = self.gesture_options[gesture_index]
                    self.toggle_gesture_selection(gesture, self.mute_gestures)

            elif unmute_section_y_start <= y < unmute_section_y_start + len(self.gesture_options) * button_height:
                gesture_index = (y - unmute_section_y_start) // button_height
                if 0 <= gesture_index < len(self.gesture_options):
                    gesture = self.gesture_options[gesture_index]
                    self.toggle_gesture_selection(gesture, self.unmute_gestures)

    def toggle_gesture_selection(self, gesture: str, gesture_set: Set[str]) -> None:
        gesture_set.clear()
        gesture_set.add(gesture)

    def select_gestures(self) -> None:
        button_height = 70
        title_height = 30
        title_button_space = 20
        section_height = title_height + len(self.gesture_options) * button_height + title_button_space
        selection_window_height = section_height * 2 + title_button_space
        selection_window = np.zeros((selection_window_height, 400, 3), dtype=np.uint8)

        cv2.namedWindow('Gesture Selection')
        cv2.setMouseCallback('Gesture Selection', self.on_gesture_click)

        while True:
            self.draw_gesture_buttons(selection_window, title_height, self.gesture_options, self.mute_gestures,
                                      "Mute Micro:")
            self.draw_gesture_buttons(selection_window, section_height + title_button_space, self.gesture_options,
                                      self.unmute_gestures, "Unmute Micro:")
            cv2.imshow('Gesture Selection', selection_window)

            key = cv2.waitKey(1)
            if key == 27:  # ESC
                break
            elif key == ord('s'):
                self.save_gesture_settings()
                break

        cv2.destroyWindow('Gesture Selection')

    def handle_gesture_action(self, gesture: str, current_time: float, last_action_time: float) -> float:
        if gesture in self.mute_gestures:
            self.mic_controller.toggle_microphone(False)
            return current_time
        elif gesture in self.unmute_gestures:
            self.mic_controller.toggle_microphone(True)
            return current_time
        return last_action_time

    def run(self) -> None:
        last_action_time = 0
        action_cooldown = 4.0  # Время в секундах

        while self.cap.isOpened():
            success, image = self.cap.read()
            if not success:
                continue

            image, hand_landmarks = self.detector.process_frame(image)
            current_time = time.time()

            if hand_landmarks and current_time - last_action_time > action_cooldown:
                if self.detector.is_v_sign(hand_landmarks):
                    last_action_time = self.handle_gesture_action('✌️', current_time, last_action_time)
                elif self.detector.is_one_finger_up(hand_landmarks):
                    last_action_time = self.handle_gesture_action('☝️', current_time, last_action_time)

            cv2.imshow('Micro tracking', image)
            if cv2.waitKey(1) & 0xFF == ord('g'):
                self.select_gestures()
            elif cv2.waitKey(1) & 0xFF == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()
