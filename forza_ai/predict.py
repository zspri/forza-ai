from PIL import Image
from queue import Queue
from threading import Thread
from win32gui import GetForegroundWindow, GetWindowText
from . import canny, datatypes
import keyboard
import logging
import mss.windows
import numpy as np
import time


def press_for_duration(key: str, dur: float):
    keyboard.press(key)
    time.sleep(dur)
    keyboard.release(key)


class PredictionHandler:
    def __init__(self):
        self.thread = Thread(target=self.pr_thread, daemon=True)
        self.data_queue = Queue()

        self.do_canny = True
        self.ui_elements = {}

    def pr_thread(self):
        frame_count = 0
        last_fps = time.time()

        fg = 'Forza Horizon 4'

        sct: mss.windows.MSS
        with mss.mss() as sct:
            while fg == 'Forza Horizon 4':
                fg = GetWindowText(GetForegroundWindow())

                ss = sct.grab({'top': 0, 'left': 0, 'width': 1920, 'height': 1080})
                img = Image.frombytes('RGB', (ss.width, ss.height), ss.rgb)

                try:
                    if self.do_canny:
                        prediction = canny.canny_all(img)
                        self.data_queue.put(prediction)
                    else:
                        prediction = datatypes.PredictionData(img, np.ndarray([]))
                        self.data_queue.put(prediction)

                except:
                    logging.exception('canny_all failed')

                frame_count += 1
                if time.time() >= last_fps + 1:
                    fps, frame_count = frame_count, 0
                    logging.info(f'fps={fps}')
                    last_fps = time.time()

            print('Window focus removed')

