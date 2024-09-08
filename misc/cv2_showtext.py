import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# function to display the coordinates of 
# of the points clicked on the image  
def click_event(event, x, y, flags, params): 
  
    # checking for left mouse clicks 
    if event == cv2.EVENT_LBUTTONDOWN: 
  
        # displaying the coordinates 
        # on the Shell 
        print(x, ' ', y) 
  
        # displaying the coordinates 
        # on the image window 
        font = cv2.FONT_HERSHEY_SIMPLEX 
        cv2.putText(img, str(x) + ',' +
                    str(y), (x,y), font, 
                    1, (255, 0, 0), 2) 
        cv2.imshow('image', img) 
  
    # checking for right mouse clicks      
    if event==cv2.EVENT_RBUTTONDOWN: 
  
        # displaying the coordinates 
        # on the Shell 
        print(x, ' ', y) 
  
        # displaying the coordinates 
        # on the image window 
        font = cv2.FONT_HERSHEY_SIMPLEX 
        b = img[y, x, 0] 
        g = img[y, x, 1] 
        r = img[y, x, 2] 
        cv2.putText(img, str(b) + ',' +
                    str(g) + ',' + str(r), 
                    (x,y), font, 1, 
                    (255, 255, 0), 2) 
        cv2.imshow('image', img) 
  
# driver function 
if __name__=="__main__": 
  
    # reading the image 
    img = cv2.imread('assets/self_gesture.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.GaussianBlur(img, (3,3), 0)
    x_min, x_max = 523, 779
    y_min, y_max = 636, 644
    screen_roi = img[y_min:y_max, x_min:x_max]
    # screen_roi = screen_roi[...,2]
    sns.heatmap(screen_roi)
    plt.show()
    cond1 = np.where(screen_roi[4] > 80, True, False)
    cond2 = np.where(screen_roi[4] < 120, True, False)
    health = np.logical_and(cond1, cond2).sum() / screen_roi.shape[1]
    health *= 100
    cv2.imshow('test', screen_roi)
    cv2.waitKey(0)
    # displaying the image 
    cv2.imshow('image', img) 
  
    # setting mouse handler for the image 
    # and calling the click_event() function 
    cv2.setMouseCallback('image', click_event) 
  
    # wait for a key to be pressed to exit 
    cv2.waitKey(0) 
  
    # close the window 
    cv2.destroyAllWindows() 