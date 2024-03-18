import numpy as np
import cv2

image = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.namedWindow('Canvas')
global ix, iy, is_drawing
is_drawing = False

def paint(event, x, y, flags, param):
    global ix, iy, is_drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        is_drawing = True
        ix = x
        iy = y
    elif event == cv2.EVENT_MOUSEMOVE:
        if is_drawing == True:
            cv2.line(image, (ix, iy), (x, y), (255, 255, 255), 5)
            ix = x
            iy = y
    elif event == cv2.EVENT_LBUTTONUP:
        is_drawing =False
        cv2.line(image, (ix, iy), (x, y), (255, 255, 255), 5)
        ix = x
        iy = y

    return x, y

cv2.setMouseCallback('Canvas', paint)

while True:
    cv2.imshow('Canvas', 255 - image)
    key = cv2.waitKey(10)
    if key == ord(" "):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

