import torch
from torchvision import transforms
import cv2 
import torch.nn as nn
import numpy as np
import random
import win32gui, win32ui, win32con, win32api, win32process
from misc.screen import grab_window
import os, time, sys, pymem
from PIL import Image
from network import DQN
from pyautogui import press, typewrite, hotkey
from misc.action_interface import action_interface





class Sekiro_Env():
    def __init__(self, game_resolution=(1280, 720)):
        self.action_space = None
        self.observation_space = None
        self.state = torch.zeros(1280,720).cuda()
        self.game_resolution = game_resolution
        self.action_interface = action_interface()
        self.template_path_death = cv2.imread('asset/dead.png', 0)
        self.threshold = 0.6

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    

    def get_state(self):
        ## get the current window and grab raw opencv image
        hwin_sekiro = win32gui.FindWindow(None, 'Sekiro')
        win32gui.SetForegroundWindow(hwin_sekiro)
        img = grab_window(hwin_sekiro, game_resolution=self.game_resolution, SHOW_IMAGE=False)
        
        ## transform opencv image/array to pytorch compatible format
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        # img_transpose = img_rgb.transpose(2,1,0) 
        img_tensor = self.transform(img_pil).cuda()
        
        return img_tensor, img_rgb
        


    def take_action(self, action):
        message, action = self.action_interface.action_map.get(action)
        print(f"{message}")
        action()

    
    def cal_reward(self, new_state):
        pass

    def check_done(self, new_state):
        gray_state = cv2.cvtColor(new_state, cv2.COLOR_RGB2GRAY)
        res = cv2.matchTemplate(gray_state, self.template_path_death, cv2.TM_CCOEFF_NORMED)

        if np.max(res) >= self.threshold:
            print("Sekiro is dead")
            return True
        else:
            return False


    def step(self, action):
        self.take_action(action)
        new_state, cv2_img = self.get_state()
        reward = self.cal_reward(new_state)
        done = self.check_done(cv2_img)
        
        return new_state, reward, done
        

    def reset(self):
        self.action_interface.reset_state()
        
        





if __name__ == "__main__":
    env = Sekiro_Env()
    while True:
        tensor, _ = env.get_state()