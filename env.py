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
import cv2
from misc.key_input import key_check
from misc.key_output import HoldKey, ReleaseKey, k_char, l_char

class Sekiro_Env:
    """
    A class to represent the game environment for Sekiro.
    This class handles the interaction with the game, including capturing the game state,
    performing actions, and checking game conditions.
    """

    def __init__(self, game_resolution=(1280, 720)):
        """
        Initialize the Sekiro_Env class.
        
        Parameters:
        - game_resolution (tuple): The resolution of the game window (width, height).
        
        Attributes:
        - action_space (None): Placeholder for the action space, to be defined based on the game.
        - observation_space (None): Placeholder for the observation space, to be defined based on the game.
        - state (torch.Tensor): The initial state of the game as a torch tensor.
        - game_resolution (tuple): Stores the resolution of the game window.
        - action_interface (ActionInterface): Interface to map actions to game inputs.
        - template_path_death (numpy.ndarray): Template image for detecting the death screen.
        - threshold (float): Threshold for template matching when checking if the game character is dead.
        - transform (torchvision.transforms.Compose): Transformations to be applied to the game image.
        """
        self.action_space = None
        self.observation_space = None
        self.state = torch.zeros(1280, 720).cuda()
        self.game_resolution = game_resolution
        self.action_interface = action_interface()
        self.template_path_death = cv2.imread('assets/dead.png', 0)
        self.threshold = 0.57
        self.counter = 0
        self.use_color_gesture = False
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def get_state(self, if_tensor=False):
        """
        Capture the current state of the game window.
        
        - Grabs the current game window and converts the image to a format suitable for processing.
        - Converts the captured image from BGR to RGB.
        - Applies transformations to the image for further use with PyTorch models.
        
        Returns:
        - img_tensor (torch.Tensor): The transformed image tensor.
        - img_rgb (numpy.ndarray): The original captured image in RGB format.
        """
        hwin_sekiro = win32gui.FindWindow(None, 'Sekiro')
        win32gui.SetForegroundWindow(hwin_sekiro)
        img = grab_window(hwin_sekiro, game_resolution=self.game_resolution, SHOW_IMAGE=False)
        
        # not sure if we need torch tensor at this stage
        if if_tensor:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tensor = self.transform(img_pil).cuda()
        
        return img

    def take_action(self, action):
        """
        Execute the given action in the game.
        
        - Maps the action to a corresponding game input using the action interface.
        - Prints the message corresponding to the action.
        - Executes the action.
        
        Parameters:
        - action (int): The action to be taken, as an integer index.
        """
        message, action_func = self.action_interface.action_map.get(action)
        print(f"{message}")
        action_func()

    def extract_self_health(self, next_state):
        img = cv2.cvtColor(next_state, cv2.COLOR_BGR2GRAY)
        x_min, x_max = 66, 494
        y_min, y_max = 666, 674
        screen_roi = img[y_min:y_max, x_min:x_max]
        cond1 = np.where(screen_roi[4] > 60, True, False)
        cond2 = np.where(screen_roi[4] < 90, True, False)
        health = np.logical_and(cond1, cond2).sum() / screen_roi.shape[1]
        health *= 100
        print(f"Sekrio_health is {health}")
        return health
    
    def extract_boss_health(self, next_state):
        img = cv2.cvtColor(next_state, cv2.COLOR_BGR2GRAY)
        x_min, x_max = 67, 349
        y_min, y_max = 53, 61
        screen_roi = img[y_min:y_max, x_min:x_max]
        cond1 = np.where(screen_roi[4] > 50, True, False)
        cond2 = np.where(screen_roi[4] < 80, True, False)
        health = np.logical_and(cond1, cond2).sum() / screen_roi.shape[1]
        health *= 100
        print(f"boss_health is: {health}")
        return health
    

    def extract_self_gesture(self, next_state):
        img = cv2.cvtColor(next_state, cv2.COLOR_BGR2GRAY)
        x_min, x_max = 523, 779
        y_min, y_max = 636, 644
        screen_roi = img[y_min:y_max, x_min:x_max]
        cond1 = np.where(screen_roi[4] > 80, True, False)
        cond2 = np.where(screen_roi[4] < 120, True, False)
        gesture = np.logical_and(cond1, cond2).sum() / screen_roi.shape[1]
        gesture *= 100
        print(f"Sekiro_gesture is: {gesture}")
        return gesture


    def extract_boss_gesture(self, next_state):
        if self.use_color_gesture:
            img = next_state
            x_min, x_max = 427, 869
            y_min, y_max = 32, 38
            screen_roi = img[y_min:y_max, x_min:x_max]
            hsv = cv2.cvtColor(screen_roi, cv2.COLOR_BGR2HSV) 
            lower = np.array([22, 93, 0])
            upper = np.array([45, 255, 255])
            mask = cv2.inRange(hsv, lower, upper) 
            cv2.imshow("Mask", mask) 
        else:
            img = cv2.cvtColor(next_state, cv2.COLOR_BGR2GRAY)
            x_min, x_max = 427, 869
            y_min, y_max = 32, 38
            screen_roi = img[y_min:y_max, x_min:x_max]
            cond1 = np.where(screen_roi[3] > 60, True, False)
            cond2 = np.where(screen_roi[3] < 90, True, False)
            gesture = np.logical_and(cond1, cond2).sum() / screen_roi.shape[1]
            gesture *= 100
        
        
        print(f"boss_gesture is: {gesture}")
        return gesture
    
    def cal_reward(self, new_state):
        """
        Calculate the reward based on the new state of the game.
        
        - This method is currently a placeholder and should be implemented based on the specific
          game logic for reward calculation.
        
        Parameters:
        - new_state (torch.Tensor): The new state of the game after taking an action.
        
        Returns:
        - reward (float): The calculated reward (currently not implemented).
        """
        self_health = self.extract_self_health(new_state)
        self_gesture = self.extract_self_gesture(new_state)

        boss_health = self.extract_boss_health(new_state)

        return 0

    def check_done(self, new_state):
        """
        Check if the game has reached a terminal state (e.g., the player character is dead).
        
        - Converts the new state to grayscale.
        - Uses template matching to detect if the death screen is visible.
        
        Parameters:
        - new_state (numpy.ndarray): The new game state as a RGB image.
        
        Returns:
        - done (bool): True if the game is over, False otherwise.
        """
        gray_state = cv2.cvtColor(new_state, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(gray_state, self.template_path_death, cv2.TM_CCOEFF_NORMED)

        if np.max(res) >= self.threshold:
            print("Sekiro is dead")
            return True
        else:
            return False

    def step(self, action):
        """
        Perform a step in the environment by taking an action and observing the result.
        
        - Takes the specified action.
        - Captures the new state of the game.
        - Calculates the reward for the new state.
        - Checks if the game is over.
        
        Parameters:
        - action (int): The action to be performed.
        
        Returns:
        - new_state (torch.Tensor): The new state of the game.
        - reward (float): The reward obtained from the new state (currently not implemented).
        - done (bool): Whether the game has reached a terminal state.
        """
        self.take_action(action)
        cv2_img = self.get_state()
        reward = self.cal_reward(cv2_img)
        done = self.check_done(cv2_img)
        if (self.counter%5) ==0 :
            cv2.imwrite(f'test_imgs/check_{self.counter}.png', cv2_img)
        self.counter += 1
        return cv2_img, reward, done

    def reset(self):
        """
        Reset the game environment to its initial state.
        
        - Resets the state of the action interface, preparing the environment for a new game session.
        """
        self.action_interface.reset_state()

if __name__ == "__main__":
    # cv2.imshow('test_health', next_state)
    # cv2.waitKey(0)
    def model(state):
        action = random.randint(0, 9)
        return action

    env = Sekiro_Env()
    obs = env.get_state()
    
    while True:
        action = model(obs)
        obs, reward, done = env.step(action)
        if done:
            time.sleep(4)
            HoldKey(k_char)
            time.sleep(0.3)
            ReleaseKey(k_char)
            time.sleep(10)
            env.reset()
            time.sleep(10)
            HoldKey(l_char)
            time.sleep(0.3)
            ReleaseKey(l_char)

        keys_pressed_tp = key_check()
        if 'Q' in keys_pressed_tp:
            # exit loop
            print('exiting...')
            
            break


