import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
offset = 30
imgSize =300


while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
    cv2.imshow("image",img)
    key = cv2.waitKey(1)
    if hands:
        hand1 = hands[0]
        
        if len(hands)==1:
            x, y, w, h = hand1['bbox']
            
            imgcrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            imgCropShape = imgcrop.shape
        else:
            hand2 = hands[1]
            x1, y1, w1, h1 = hand1['bbox']
            x2, y2, w2, h2 = hand2['bbox']
        
            x = min(x1,x2)
            y = min(y1,y2)
            boxgapX = min(abs((x1+w1)-x2),abs((x1+w1)-(x2+w2)))
            boxgapY = min(abs((y1+h1)-x2),abs((y1+h1)-(y2+h2)))
            w = w1+w2+boxgapX
            h = h1+h2+boxgapY


            print(x,y,w,h)
            imgcrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            imgCropShape = imgcrop.shape
        
        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgcrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgcrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
        cv2.imshow("ImageCrop", imgcrop)
        cv2.imshow("ImageWhite", imgWhite)
    if key == ord("q"):
        break
cv2.destroyAllWindows()  