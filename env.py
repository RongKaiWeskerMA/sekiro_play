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
from misc.key_output import HoldKey, ReleaseKey, j_char, l_char


class Sekiro_Env:
    """
    A class to represent the game environment for Sekiro.
    This class handles the interaction with the game, including capturing the game state,
    performing actions, and checking game conditions.

    Attributes:
        game_resolution (tuple): The resolution of the game window (width, height).
        action_interface (ActionInterface): Interface to map actions to game inputs.
        action_space (int): The number of possible actions in the game.
        observation_space (None): Placeholder for the observation space, not currently used.
        template_path_death (numpy.ndarray): Template image for detecting the death screen.
        threshold (float): Threshold for template matching when checking if the game character is dead.
        counter (int): Counter to keep track of the number of steps taken.
        use_color_gesture (bool): Flag to determine whether to use color-based gesture detection.
        dead_counter (int): Counter to keep track of consecutive frames where the character appears dead.
        hwin_sekiro (int): Handle to the Sekiro game window.
        boss_health_prev (float): Previous health value of the boss, used for reward calculation.
    """

    def __init__(self, game_resolution=(1280, 720)):
        """
        Initialize the Sekiro_Env class.
        
        Args:
            game_resolution (tuple): The resolution of the game window (width, height).
        """
        self.game_resolution = game_resolution
        self.action_interface = action_interface()
        self.action_space = len(self.action_interface.action_map)
        self.observation_space = None
        self.template_path_death = cv2.imread('assets/death_crop.png', 0)
        self.threshold = 0.55
        self.counter = 0
        self.use_color_gesture = True
        self.dead_counter = 0
        self.hwin_sekiro = win32gui.FindWindow(None, 'Sekiro')
        self.boss_health_prev = 86
    
    def get_state(self, if_tensor=False):
        """
        Capture the current state of the game window.

        Args:
            if_tensor (bool): Flag to determine if the output should be a tensor (not used in current implementation).

        Returns:
            numpy.ndarray: The captured image of the game window in BGR format.
        """
        
        
        img = grab_window(self.hwin_sekiro, game_resolution=self.game_resolution, SHOW_IMAGE=False)
        
        return img

    def take_action(self, action):
        """
        Execute the given action in the game.
        
        Args:
            action (int): The action to be taken, as an integer index.
        """
        message, action_func = self.action_interface.action_map.get(action)
        print(f"{message}")
        action_func()

    def extract_self_health(self, next_state):
        """
        Extract the player's health from the game state image.

        Args:
            next_state (numpy.ndarray): The current game state image.

        Returns:
            float: The extracted health value as a percentage.
        """
        img = cv2.cvtColor(next_state, cv2.COLOR_BGR2GRAY)
        x_min, x_max = 59, 494
        y_min, y_max = 666, 674
        screen_roi = img[y_min:y_max, x_min:x_max]
        cond1 = np.where(screen_roi[4] > 60, True, False)
        cond2 = np.where(screen_roi[4] < 90, True, False)
        health = np.logical_and(cond1, cond2).sum() / screen_roi.shape[1]
        health *= 100
        # if health < 1:
        #     self.dead_counter += 1
        # else:
        #     self.dead_counter = 0

        print(f"Sekrio_health is {health}")
        return health
    
    def extract_boss_health(self, next_state):
        """
        Extract the boss's health from the game state image.

        Args:
            next_state (numpy.ndarray): The current game state image.

        Returns:
            float: The extracted boss health value as a percentage.
        """
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
        """
        Extract the player's gesture meter from the game state image.

        Args:
            next_state (numpy.ndarray): The current game state image.

        Returns:
            float: The extracted gesture meter value as a percentage.
        """
        if self.use_color_gesture:
            img = next_state
            x_min, x_max = 523, 779
            y_min, y_max = 636, 644
            screen_roi = img[y_min:y_max, x_min:x_max]
            hsv = cv2.cvtColor(screen_roi, cv2.COLOR_BGR2HSV) 
            lower = np.array([15, 100, 100])
            upper = np.array([45, 255, 255])
            mask = cv2.inRange(hsv, lower, upper) 
            cond = np.where(mask[4] > 0, True, False)
            gesture = cond.sum() / screen_roi.shape[1]
            gesture *= 100
        
        
        else:
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
        """
        Extract the boss's gesture meter from the game state image.

        Args:
            next_state (numpy.ndarray): The current game state image.

        Returns:
            float: The extracted boss gesture meter value as a percentage.
        """
        if self.use_color_gesture:
            img = next_state
            x_min, x_max = 427, 869
            y_min, y_max = 32, 38
            screen_roi = img[y_min:y_max, x_min:x_max]
            hsv = cv2.cvtColor(screen_roi, cv2.COLOR_BGR2HSV) 
            lower = np.array([15, 100, 100])
            upper = np.array([45, 255, 255])
            mask = cv2.inRange(hsv, lower, upper) 
            cond = np.where(mask[4] > 0, True, False)
            gesture = cond.sum() / screen_roi.shape[1]
            gesture *= 100
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
    
    def cal_reward(self, new_state, action):
        """
        Calculate the reward based on the new state of the game and the action taken.

        Args:
            new_state (numpy.ndarray): The new state of the game after taking an action.
            action (int): The action that was taken.

        Returns:
            float: The calculated reward.
        """
        reward = 0
        
        self_health = self.extract_self_health(new_state)
        self_gesture = self.extract_self_gesture(new_state)

        boss_health = self.extract_boss_health(new_state)
        boss_gesture = self.extract_boss_gesture(new_state)

        if self_health > 50:
            reward += 0.2
        
        # if self_gesture < 70:
        #     reward += 0.2
        
        reward += 86+ self.boss_health_prev - 2 * boss_health 
        self.boss_health_prev = boss_health 
        reward += boss_gesture 
        
        
        # prevent sekiro to take stupid action
        if self_health > 80 and action == 9:
            reward -= 10
        if self.counter < 3 and action == 9:
            reward -= 10
        return reward

    def check_done(self, new_state):
        """
        Check if the game has reached a terminal state (e.g., the player character is dead).

        Args:
            new_state (numpy.ndarray): The new game state as a RGB image.

        Returns:
            bool: True if the game is over, False otherwise.
        """
        x_min, x_max = 533, 776
        y_min, y_max = 222, 425
        new_state = new_state[y_min:y_max, x_min:x_max]
        gray_state = cv2.cvtColor(new_state, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(gray_state, self.template_path_death, cv2.TM_CCOEFF_NORMED)

        if np.max(res) >= self.threshold:
            self.dead_counter += 1
            print("Sekiro is dead")
            return True
        else:
            return False
        # if self.dead_counter >= 1:
        #     print("Sekiro is dead")
        #     return True
             
        # else:
        #     return False    

    def step(self, action):
        """
        Perform a step in the environment by taking an action and observing the result.

        Args:
            action (int): The action to be performed.

        Returns:
            tuple: Containing:
                - numpy.ndarray: The new state of the game.
                - float: The reward obtained from the new state.
                - bool: Whether the game has reached a terminal state.
        """
        self.take_action(action)
        cv2_img = self.get_state()
        reward = self.cal_reward(cv2_img, action)
        done = self.check_done(cv2_img)
        self.counter += 1
        if done:
            reward -= 2
        print(f"step_reward is: {reward}")
        print(f"step_counter is: {self.counter}")
        print("")

        return cv2_img, reward, done

    def reset(self):
        """
        Reset the game environment to its initial state.

        This method resets various counters, waits for the game to reload,
        and performs necessary actions to start a new game session.
        """
        self.counter = 0
        self.dead_counter = 0
        time.sleep(6)
        HoldKey(j_char)
        time.sleep(0.3)
        ReleaseKey(j_char)
        time.sleep(10)
        self.action_interface.reset_state()
        time.sleep(11)
        HoldKey(l_char)
        time.sleep(0.3)
        ReleaseKey(l_char)

if __name__ == "__main__":
    """
    Main entry point for testing the Sekiro_Env class.

    This section contains commented-out code for running a simple random agent
    in the Sekiro environment and capturing screenshots.
    """
    # cv2.imshow('test_health', next_state)
    # cv2.waitKey(0)
    
    # def model(state):
    #     action = random.randint(0, 9)
    #     return action
    

    transform = transforms.Compose([
            transforms.Resize((224, 224)),
             transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    

    env = Sekiro_Env()

    model = DQN(env.action_space, 'efficientnet').to('cuda')
    model.load_state_dict(torch.load('checkpoints/bc/checkpoint_epoch_6.pth', map_location=torch.device('cuda'))['model_state_dict'])
    model.eval()
    # for i in range(10000000000):
    #     img = env.get_state()
    #     cv2.imwrite(f"test_imgs/test_{i}.png", img)
    win32gui.SetForegroundWindow(env.hwin_sekiro)
    obs = env.get_state()
    
    while True:
        input = transform(Image.fromarray(cv2.cvtColor(obs, cv2.COLOR_BGR2RGB))).to("cuda").unsqueeze(0)
        output = model(input)
        action = output.argmax().item()
        print(action)
        obs, reward, done = env.step(action)
        
        
        keys_pressed_tp = key_check()
        if 'Q' in keys_pressed_tp:
            # exit loop
            print('exiting...')
            
            break


