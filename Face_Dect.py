import cv2
import numpy as np

webCam_ID = 0

#USER INFO
id = raw_input('Enter the USER ID (which will help the program to indetify the user) = \n')
sampleNum = 0;

face_cascade = cv2.CascadeClassifier('h_frface.xml')

cam_cv2 = cv2.VideoCapture(1)

while True:
    ret, img = cam_cv2.read()
    gray_color = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    detect_faces = face_cascade.detectMultiScale(gray_color,1.3,5)
    for (x,y,w,h) in detect_faces:
        #Gen a fle of user data with the user id and the sple num
        sampleNum = sampleNum + 1;
        cv2.imwrite('USERINFO/user_' + str(id)+ "_" + str(sampleNum) + ".jpg",gray_color[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.waitKey(100)

    cv2.imshow('img',img)
    cv2.waitKey(1)
    if(sampleNum > 20):
        break;

#closing the window
cam_cv2.release()
cv2.destroyAllWindows()
