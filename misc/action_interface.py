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
from ctypes  import *
from pymem import *
import numpy as np

from key_input import key_check, mouse_check, mouse_l_click_check, mouse_r_click_check
from key_output import set_pos, HoldKey, ReleaseKey
from key_output import left_click, hold_left_click, release_left_click
from key_output import right_click, hold_right_click, release_right_click
from key_output import w_char, s_char, a_char, d_char, n_char, q_char, b_char, j_char
from key_output import ctrl_char, shift_char, space_char, up_key, down_key
from key_output import r_char, one_char, two_char, three_char, four_char
from key_output import p_char, e_char, c_char_, t_char, cons_char, ret_char
from key_output import m_char, u_char, under_char, g_char, esc_char
from key_output import i_char, v_char, o_char, g_char, k_char, seven_char, x_char, c_char2, y_char
import pyautogui


class action_interface():
    def __init__(self):
        self.action_map = self.define_action_map()

    
    def define_action_map(self):
        action_map = {
            0: ("move_forward", lambda: self.move_forward()),
            1: ("move_left", lambda: self.move_left()),
            2: ("move_backward", lambda: self.move_backward()),
            3: ("move_right", lambda: self.move_right()),
            4: ("attack", lambda: self.attack()),
            5: ("dodge", lambda: self.dodge()),
            6: ("run", lambda: self.run()),
            7: ("jump", lambda: self.jump()),
            8: ("parry", lambda: self.parry()),
            9: ("drink", lambda: self.drink()),    
        }

        return action_map
    

    def move_forward(self):
        HoldKey(w_char)
        time.sleep(1)
        ReleaseKey(w_char)

    def move_left(self):
        HoldKey(a_char)
        time.sleep(0.3)
        ReleaseKey(a_char)
    
    def move_right(self):
        HoldKey(d_char)
        time.sleep(0.3)
        ReleaseKey(d_char)
    
    def move_backward(self):
        HoldKey(s_char)
        time.sleep(0.3)
        ReleaseKey(s_char)
    
    def run(self):
        HoldKey(w_char)
        HoldKey(shift_char)
        time.sleep(1)
        ReleaseKey(w_char)
        ReleaseKey(shift_char)
    
    def jump(self):
        HoldKey(space_char)
        time.sleep(0.3)
        ReleaseKey(space_char)
    
    def attack(self):
        for i in range(2):
            HoldKey(j_char)
            time.sleep(0.05)
            ReleaseKey(j_char)
            time.sleep(0.5)

    def dodge(self):
        HoldKey(shift_char)
        time.sleep(0.3)
        ReleaseKey(shift_char)

    def parry(self):
        for i in range(3):
            HoldKey(k_char)
            time.sleep(0.05)
            ReleaseKey(k_char)
            time.sleep(0.1)

    def drink(self):
        HoldKey(r_char)
        time.sleep(0.3)
        ReleaseKey(r_char)
    

    def reset_state(self):
        # reach the fire point
        HoldKey(w_char)
        time.sleep(0.7)
        ReleaseKey(w_char)

        # press e
        HoldKey(e_char)
        time.sleep(0.3)
        ReleaseKey(e_char)
        time.sleep(2)
        
        # press up key
        for i in range(2):
            HoldKey(up_key)
            time.sleep(0.05)
            ReleaseKey(up_key)
            time.sleep(0.5)
        # enter
        HoldKey(ret_char)
        time.sleep(0.3)
        ReleaseKey(ret_char)

        # press down
        for i in range(2):
            HoldKey(down_key)
            time.sleep(0.05)
            ReleaseKey(down_key)
            time.sleep(0.5)

        # enter
        for i in range(2):
            HoldKey(ret_char)
            time.sleep(0.3)
            ReleaseKey(ret_char)
            time.sleep(1)

if __name__ == "__main__":
    time.sleep(3)
    act = action_interface()
    act.reset_state()

    # interface = action_interface()
    # HoldKey(e_char)
    # time.sleep(0.3)
    # ReleaseKey(e_char)

    # for i in range(3):
    #     HoldKey(k_char)
    #     time.sleep(0.05)
    #     ReleaseKey(k_char)
    #     time.sleep(0.1)
   