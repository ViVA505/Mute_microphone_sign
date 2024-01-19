import subprocess
import platform
from typing import Optional

if platform.system() == "Windows":
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

class MicrophoneController:
    def __init__(self) -> None:
        self.mic_muted = False
        if platform.system() == "Windows":
            self.init_windows_audio()

    def init_windows_audio(self) -> None:
        self.device = AudioUtilities.GetSpeakers()
        self.interface = self.device.Activate(
            IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume = cast(self.interface, POINTER(IAudioEndpointVolume))

    def toggle_microphone(self, enable_mic: bool) -> None:
        if platform.system() == 'Linux':
            self.toggle_microphone_linux(enable_mic)
        elif platform.system() == "Windows":
            self.toggle_microphone_windows(enable_mic)

    def toggle_microphone_linux(self, enable_mic: bool) -> None:
        mute_state = '0' if enable_mic else '1'
        subprocess.run(['pactl', 'set-source-mute', '@DEFAULT_SOURCE@', mute_state])
        print(f"{'ВКЛЮЧЕНИЕ' if enable_mic else 'ВЫКЛЮЧЕНИЕ'} МИКРОФОНА")

    def toggle_microphone_windows(self, enable_mic: bool) -> None:
        self.volume.SetMute(0 if enable_mic else 1, None)
        print(f"{'ВКЛЮЧЕНИЕ' if enable_mic else 'ВЫКЛЮЧЕНИЕ'} МИКРОФОНА")
