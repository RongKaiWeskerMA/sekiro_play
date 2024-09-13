import os
import time
import mss
import cv2
import socket
import sys
import struct
import math
import random
import win32api as wapi
import win32api
import win32gui
import win32process
import ctypes
from ctypes import *
from pymem import *
import numpy as np

from .key_input import key_check, mouse_check, mouse_l_click_check, mouse_r_click_check
from .key_output import set_pos, HoldKey, ReleaseKey
from .key_output import left_click, hold_left_click, release_left_click
from .key_output import right_click, hold_right_click, release_right_click
from .key_output import w_char, s_char, a_char, d_char, n_char, q_char, b_char, j_char
from .key_output import ctrl_char, shift_char, space_char, up_key, down_key
from .key_output import r_char, one_char, two_char, three_char, four_char
from .key_output import p_char, e_char, c_char_, t_char, cons_char, ret_char
from .key_output import m_char, u_char, under_char, g_char, esc_char
from .key_output import i_char, v_char, o_char, g_char, k_char, seven_char, x_char, c_char2, y_char
import pyautogui

class action_interface:
    """
    A class to define and manage actions in the game environment.
    This class provides an interface for mapping actions to specific game inputs,
    allowing for easy interaction with the game through predefined actions.
    """

    def __init__(self):
        """
        Initialize the action_interface class.

        Attributes:
        - action_map (dict): A dictionary mapping action indices to their corresponding actions and descriptions.
        """
        self.action_map = self.define_action_map()

    def define_action_map(self):
        """
        Define the action map for the game.

        - Maps integer indices to specific game actions, each represented by a tuple containing
          a description of the action and a lambda function that performs the action.

        Returns:
        - action_map (dict): A dictionary where each key is an integer representing an action,
                             and each value is a tuple (description, function).
        """
        action_map = {
            0: ("move_forward", lambda: self.move_forward()),
            1: ("move_left", lambda: self.move_left()),
            2: ("move_backward", lambda: self.move_backward()),
            3: ("move_right", lambda: self.move_right()),
            4: ("attack", lambda: self.attack()),
            5: ("dodge", lambda: self.dodge()),
            6: ("jump", lambda: self.jump()),
            7: ("parry", lambda: self.parry()),
            8: ("drink", lambda: self.drink()),    
            9: ("static", lambda: self.static()),
        }

        # action_map = {
        #     0: ("move_forward", lambda: self.move_forward()),
        #     1: ("move_left", lambda: self.move_left()),
        #     2: ("move_right", lambda: self.move_right()),
        #     3: ("attack", lambda: self.attack()),
        #     4: ("dodge", lambda: self.dodge()),
        #     5: ("parry", lambda: self.parry()),
        #     6: ("drink", lambda: self.drink()),    
        # }



        return action_map

    def move_forward(self):
        """
        Simulate moving forward in the game by holding the 'W' key for a set duration.
        """
        HoldKey(w_char)
        time.sleep(1)
        ReleaseKey(w_char)

    def move_left(self):
        """
        Simulate moving left in the game by holding the 'A' key for a set duration.
        """
        HoldKey(a_char)
        time.sleep(0.3)
        ReleaseKey(a_char)

    def move_right(self):
        """
        Simulate moving right in the game by holding the 'D' key for a set duration.
        """
        HoldKey(d_char)
        time.sleep(0.3)
        ReleaseKey(d_char)

    def move_backward(self):
        """
        Simulate moving backward in the game by holding the 'S' key for a set duration.
        """
        HoldKey(s_char)
        time.sleep(0.3)
        ReleaseKey(s_char)

    def run(self):
        """
        Simulate running in the game by holding both the 'W' and 'Shift' keys for a set duration.
        """
        HoldKey(w_char)
        HoldKey(shift_char)
        time.sleep(0.7)
        ReleaseKey(w_char)
        ReleaseKey(shift_char)

    def jump(self):
        """
        Simulate jumping in the game by holding the 'Space' key for a set duration.
        """
        HoldKey(space_char)
        time.sleep(0.3)
        ReleaseKey(space_char)

    def attack(self):
        """
        Simulate attacking in the game by pressing the 'J' key twice with a short delay between presses.
        """
        for i in range(2):
            HoldKey(k_char)
            time.sleep(0.05)
            ReleaseKey(k_char)
            time.sleep(0.5)

    def dodge(self):
        """
        Simulate dodging in the game by holding the 'Shift' key for a set duration.
        """
        HoldKey(shift_char)
        time.sleep(0.3)
        ReleaseKey(shift_char)

    def parry(self):
        """
        Simulate parrying in the game by pressing the 'K' key three times with a short delay between presses.
        """
        for i in range(3):
            HoldKey(j_char)
            time.sleep(0.05)
            ReleaseKey(j_char)
            time.sleep(0.1)

    def drink(self):
        """
        Simulate drinking in the game by holding the 'R' key for a set duration.
        """
        HoldKey(r_char)
        time.sleep(0.3)
        ReleaseKey(r_char)
    
    def static(self):
        """
        Simulate static in the game by holding the 'W' key for a set duration.
        """
        pass

    def reset_state(self):
        """
        Reset the game state to a known starting point by performing a series of predefined actions.

        - Simulates moving forward, interacting with an object, navigating menus, and confirming selections.
        """
        # Move forward to a specific point
        HoldKey(w_char)
        time.sleep(0.7)
        ReleaseKey(w_char)

        # Interact with an object by pressing the 'E' key
        HoldKey(e_char)
        time.sleep(0.3)
        ReleaseKey(e_char)
        time.sleep(2)
        
        # Navigate up in a menu by pressing the 'Up' key twice
        for i in range(2):
            HoldKey(up_key)
            time.sleep(0.05)
            ReleaseKey(up_key)
            time.sleep(0.5)

        # Confirm selection by pressing the 'Enter' key
        HoldKey(ret_char)
        time.sleep(0.3)
        ReleaseKey(ret_char)

        # Navigate down in a menu by pressing the 'Down' key twice
        for i in range(2):
            HoldKey(down_key)
            time.sleep(0.05)
            ReleaseKey(down_key)
            time.sleep(0.5)

        # Confirm selections twice by pressing the 'Enter' key
        for i in range(2):
            HoldKey(ret_char)
            time.sleep(0.3)
            ReleaseKey(ret_char)
            time.sleep(1)

if __name__ == "__main__":
    time.sleep(3)
    act = action_interface()
    act.reset_state()

    # Example usage of other actions:
    # HoldKey(e_char)
    # time.sleep(0.3)
    # ReleaseKey(e_char)

    # for i in range(3):
    #     HoldKey(k_char)
    #     time.sleep(0.05)
    #     ReleaseKey(k_char)
    #     time.sleep(0.1)
