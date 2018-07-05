import sys
import numpy as np
import cv2
import get_points
from pytesseract import image_to_string

# Load image
im = cv2.imread('sample2.jpg')

height, width = im.shape[:2]
im = cv2.resize(im,(int(0.3*width), int(0.3*height)), interpolation = cv2.INTER_CUBIC)

# Uncomment to do image thresholding

# gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
# blur = cv2.GaussianBlur(gray,(5,5),0)
# ret,thresh = cv2.threshold(blur,180,255,0)
# thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
# thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
# im = thresh
# cv2.imshow("Selected", im)
# cv2.waitKey(0)

fields_dict = []
while True:
    points = get_points.run(im)
    fields_dict.append(points)
    print("Points = ", points)          # Get the bounding box from user
    temp_img = im
    cv2.rectangle(temp_img,(points[0][0],points[0][1]),(points[0][2],points[0][3]),(255,0,0),2)
    cv2.imshow("Selected", temp_img)
    key = cv2.waitKey(0)
    if key == 27:
        break

print(fields_dict)

for val in fields_dict:
    x1 = val[0][0]
    y1 = val[0][1]
    x2 = val[0][2]
    y2 = val[0][3]

    temp_img = im[y1:y2, x1:x2][:]

    cv2.imshow("Selected", temp_img)
    key = cv2.waitKey(0)
    
    # Image thresholding

    gray = cv2.cvtColor(temp_img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)

    _, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    
    samples =  np.empty((0,100))
    responses = []
    keys = [i for i in range(48,58)]
    
    fields = []
    boxes = []

    # Bounding box detection

    for cnt in contours:
        if cv2.contourArea(cnt)>50:
            [x,y,w,h] = cv2.boundingRect(cnt)
            x = x + x1
            y = y + y1

            # Set limit for size of bounding box
            if  h>15 and w > 5 and w<30:
                l = [x, y, w, h]
                boxes.append(l)            # Stores image coordinates for individual letter
    
                cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),1)
                roi = thresh[y:y+h,x:x+w]
                # roismall = cv2.resize(roi,(10,10))
                cv2.imshow('norm',temp_img)
                key = cv2.waitKey(0)
    
                if key == 27:
                    sys.exit()

    boxes.sort(key=lambda x: x[0])
    fields.append([0 + x1, 0 + y1, boxes[0][0] - x1, boxes[0][3]])
    print("Boxes = ", boxes)
    print("Fields = ", fields)
    temp_field = im[x1:boxes[0][0], y1:y1+boxes[0][3]]
    print("Field is :")                 # PyTesseract function to recognize the field name
    print(image_to_string(temp_field, lang='eng'))
