import cv2
import numpy as np
import pytesseract as tess




image = cv2.imread("test_imgs/check_115.png")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
text = tess.image_to_string(image_rgb, lang="chi_sim")
print(text)
h, w, c = image.shape
boxes = tess.image_to_boxes(image)
for b in boxes.splitlines():
    b = b.split(' ')
    image = cv2.rectangle(image, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)
 
cv2.imshow('text detect', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
