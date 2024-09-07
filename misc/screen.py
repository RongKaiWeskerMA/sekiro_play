import cv2
import win32gui, win32ui, win32con, win32api
import numpy as np
import torch

def grab_window(hwin, game_resolution=(1024,768), SHOW_IMAGE=False):
        """"
        -- Inputs --

        hwin
        this is the HWND id of the cs go window
        we play in windowed rather than full screen mode
        e.g. https://docs.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-getforegroundwindow

        game_resolution=(1024,768)
        is the windowed resolution of the game
        I think could get away with game_resolution=(640,480)
        and should be quicker to grab from
        but for now, during development, I like to see the game in reasonable
        size

        SHOW_IMAGE
        whether to display the image. probably a bad idea to 
        do that here except for testing
        better to use cv2.imshow('img',img) outside the funcion

        -- Outputs --
        currently this function returns img_small
        img is the raw capture image, in BGR
        img_small is a low res image, with the thought of
        using this as input to a NN
        """

        # we used to try to get the resolution automatically
        # but this didn't seem that reliable for some reason
        # left,top,right,bottom = win32gui.GetWindowRect(hwin)
        # width = right - left 
        # height = bottom - top


        bar_height = 50 # height of header bar
        sekiro_img_dimension = [720, 1280]

        # how much of top and bottom of image to ignore
        # (reasoning we don't need the entire screen)
        # much more efficient not to grab it in the first place
        # rather than crop out later
        # this stage can be a bit of a bottle neck
        offset_height_top = 0 
        offset_height_bottom = 30


        offset_sides = 20 # ignore this many pixels on sides, 
        width = game_resolution[0] - 2*offset_sides
        height = game_resolution[1]-offset_height_top-offset_height_bottom


        hwindc = win32gui.GetWindowDC(hwin)
        srcdc = win32ui.CreateDCFromHandle(hwindc)
        memdc = srcdc.CreateCompatibleDC()
        bmp = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(srcdc, width, height)
        memdc.SelectObject(bmp)

        memdc.BitBlt((0, 0), (width, height), srcdc, (offset_sides, bar_height+offset_height_top), win32con.SRCCOPY)
        
        signedIntsArray = bmp.GetBitmapBits(True)
        img = np.frombuffer(signedIntsArray, dtype='uint8')
        img.shape = (height,width,4)

        srcdc.DeleteDC()
        memdc.DeleteDC()
        win32gui.ReleaseDC(hwin, hwindc)
        win32gui.DeleteObject(bmp.GetHandle())

        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        # img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        

        img_small = cv2.resize(img, sekiro_img_dimension[::-1])


        if SHOW_IMAGE:

            target_width = 800
            scale = target_width / img_small.shape[1] # how much to magnify
            dim = (target_width,int(img_small.shape[0] * scale))
            resized = cv2.resize(img_small, dim, interpolation = cv2.INTER_AREA)
            cv2.imshow('resized',resized) 
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()

        return img_small