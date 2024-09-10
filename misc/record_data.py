import os
import time
import datetime
import cv2
import keyboard
import pandas as pd
from ctypes import windll, Structure, c_long, byref
import win32api as wapi
import win32gui
from screen import grab_window
import json
from warnings import filterwarnings

filterwarnings('ignore')

class POINT(Structure):
    _fields_ = [("x", c_long), ("y", c_long)]

class DataRecorder:
    def __init__(self):
        # Load configuration
        with open('configs/sekiro_config.json') as config_file:
            self.config = json.load(config_file)

        self.game_title = self.config.get('game_title', 'none_specified')
        self.session_name = self.config.get('session_name', 'session_1')
        self.frame_size = self.config.get('frame_size', (1280, 720))
        self.video_record_fps = self.config.get('video_record_fps', 30)
        self.time_format = self.config.get('time_format')
        self.counter = len(os.listdir(f"data/{self.game_title}/{self.session_name}/images")) \
            if os.path.exists(f"data/{self.game_title}/{self.session_name}/images") else 0
        self.max_frames = self.config["max_frames"]
        
        self.window = win32gui.FindWindow(None, self.game_title)    

        
        # Virtual key codes for special keys
       # Virtual key codes for special keys
        self.space_key  = 0x20
        self.shift_key  = 0x10
        self.ctrl_key   = 0x11
        self.esc_key    = 0x1B
        self.enter_key  = 0x0D
        self.alt_key    = 0x12
        self.tab_key    = 0x09
        # create output dataframe
        self.columns = ['frame_name', 'record_time',
                   'space', 'shift', 'r', 'k', 'j', 'w', 'a', 's', 'd']
        self.df = pd.read_csv(f"data/{self.game_title}/{self.session_name}/label.csv") if os.path.exists(
                    f"data/{self.game_title}/{self.session_name}/label.csv") else pd.DataFrame(columns=self.columns)
        self.save_path = f"data/{self.game_title}/{self.session_name}"

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            os.makedirs(os.path.join(self.save_path, "images"))

        self.timestamp = None
        self.frame_buffer = []
        self.key_buffer = []
        self.record_data = False

    def fetch_frame(self):
        frame = grab_window(self.window, self.config.get('game_resolution', [1280, 720]))
        return frame

    def key_check(self):
        pressed_keys = {}

        # Include space, shift, and specific keys
        special_keys = {
            "space": self.space_key,
            "shift": self.shift_key,   
        }

        for key_name, key_code in special_keys.items():
            if wapi.GetAsyncKeyState(key_code):
                pressed_keys[key_name] = 1
            else:
                pressed_keys[key_name] = 0

        for key_code in range(97, 123):
            key_char = chr(key_code)
            if keyboard.is_pressed(key_char):
                pressed_keys[key_char] = 1
            else:
                pressed_keys[key_char] = 0


        return pressed_keys

    def save_data(self, frame, pressed_keys):
        if self.record_data:
            self.counter += 1
            file_name = f"{self.counter}.jpg"
            self.save_metadata(self.save_path, "meta.json")

            # Set timestamp format
            timestamp = time.time()
            if self.time_format != "unix":
                timestamp = datetime.datetime.now().strftime(self.time_format)

            # Create a new row with the frame data and key data
            row_data = {
                "frame_name": file_name,
                "record_time": timestamp,
                **pressed_keys,
            }

            # Append the new row to the DataFrame and save it to the CSV file
            self.df = pd.concat([self.df, pd.DataFrame([row_data])], ignore_index=True)
            self.df.to_csv(os.path.join(self.save_path, "label.csv"), index=False)

            # Save the captured frame as an image
            cv2.imwrite(os.path.join(self.save_path, "images", file_name), frame)

    def save_metadata(self, save_path, json_name):
        capture_resolution = self.frame_size
        fps = self.video_record_fps

        meta_data = {
            "capture_resolution": capture_resolution,
            "Game name": self.game_title,
            "last_record_frame_index": self.counter,
            "FPS": fps
        }

        with open(os.path.join(save_path, json_name), 'w') as f:
            json.dump(meta_data, f, indent=4)

    # Start recording
    def record(self):
        while True:
            # Calculate loop_start_time for controlling FPS
            loop_start_time = time.time()

            # Get the current state of pressed keys
            pressed_keys = self.key_check()

            # set up a boolean for the *value* of dictionary keys, since all keys
            # are present in pressed_keys regardless of value
            q    = pressed_keys.get("q")     != 0
            p    = pressed_keys.get("p")     != 0
            b    = pressed_keys.get("b")     != 0
            ctrl = pressed_keys.get("ctrl")  != 0
            shift = pressed_keys.get("shift") != 0

            # Begin recording if the "B" key is pressed
            if shift and ctrl and b:
                print("Begin recording")
                self.record_data = True
            # Stop the recording if the "Q and P" keys are pressed together
            if shift and ctrl and q:
                print("Quiting.")
                break
            # Pause the recording if the "P" key is pressed (and the "Q" key is not)
            if shift and ctrl and p:
                print("Pause the recording process")
                self.record_data = False

            # Print the current state of pressed keys
            if self.record_data:
                print(f'Pressed Keys: {pressed_keys}')

            # Fetch and save the current frame along with the key data
            frame = self.fetch_frame()
            self.save_data(frame, pressed_keys)

            # Control the loop execution based on the specified video_record_fps
            while time.time() < loop_start_time + 1 / self.video_record_fps:
                pass

            # If we set a maximum frame number (anything other than 0) in our config,
            # stop recording once that many frames are captured
            if self.max_frames != 0 and self.counter >= self.max_frames:
                print(f"- - - - - - - -\nReached max frame count: {self.max_frames}\nQuiting.\n- - - - - - - -")
                break


if __name__ == "__main__":
    
    
    data_recorder = DataRecorder()
    win32gui.SetForegroundWindow(data_recorder.window)
    data_recorder.record()
