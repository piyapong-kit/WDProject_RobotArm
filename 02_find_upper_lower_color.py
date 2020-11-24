#Trackbar find the specific Value HSV
import cv2
import numpy as np

def callback(x):
    pass

cv2.namedWindow('Input_Range')

ilowH = 0
ihighH = 360

ilowS = 0
ihighS = 255

ilowV = 0
ihighV = 255

#create trackbars for color change
cv2.createTrackbar('low_HUE','Input_Range',ilowH,360,callback)
cv2.createTrackbar('high_HUE','Input_Range',ihighH,360,callback)

cv2.createTrackbar('low_Saturate','Input_Range',ilowS,255,callback)
cv2.createTrackbar('high_Saturate','Input_Range',ihighS,255,callback)

cv2.createTrackbar('low_Intensity','Input_Range',ilowV,255,callback)
cv2.createTrackbar('high_Intensity','Input_Range',ihighV,255,callback)

file_path = 'Board.jpg'

while(1):
    cap = cv2.imread(file_path,1)
    
    #get Trackbar positions
    ilowH = cv2.getTrackbarPos('low_HUE','Input_Range')
    ihighH = cv2.getTrackbarPos('high_HUE','Input_Range')
    ilowS = cv2.getTrackbarPos('low_Saturate','Input_Range')
    ihighS = cv2.getTrackbarPos('high_Saturate','Input_Range')
    ilowV = cv2.getTrackbarPos('low_Intensity','Input_Range')
    ihighV = cv2.getTrackbarPos('high_Intensity','Input_Range')
    
    hsv = cv2.cvtColor(cap, cv2.COLOR_BGR2HSV)
    cv2.imshow('RGB',cap)
    lower_hsv = np.array([ilowH,ilowS,ilowV])
    higher_hsv = np.array([ihighH,ihighS,ihighV])
    mask = cv2.inRange(hsv,lower_hsv,higher_hsv)
    #cv2.imshow('mask',mask)
    
    cap = cv2.bitwise_and(cap,cap,mask=mask)
    
    #show Thresholded image
    cv2.imshow('cap',cap)
    
    #print(ilowH,ilowS,ilowV)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break
cv2.destroyAllWindows()
