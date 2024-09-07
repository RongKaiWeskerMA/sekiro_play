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
    img = cv2.imread('assets/check.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x_min, x_max = 66, 494
    y_min, y_max = 666, 674
    screen_roi = img[y_min:y_max, x_min:x_max]
    sns.heatmap(screen_roi)
    plt.show()
    canny = cv2.Canny(cv2.GaussianBlur(screen_roi,(3,3),0), 0, 100)
    value = canny.argmax(axis=-1)
    health = np.median(value) 
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