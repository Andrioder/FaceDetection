import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('h_frface.xml')

reco = cv2.createLBPHFaceRecognizer()
reco.load('TrainerXML/train_data.xml')
id = 0
font = cv2.cv.InitFont(cv2.cv.CV_FONT_VECTOR0,5,1,0,4)

cam_cv2 = cv2.VideoCapture(1)
cam_cv2.open(1)


#magic here
while True:
    ret, img = cam_cv2.read()
    gray_color = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    detect_faces = face_cascade.detectMultiScale(gray_color,1.3,5)
    for (x,y,w,h) in detect_faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        id, conf = reco.predict(gray_color[y:y+h, x:x+w])
        if (id == 1):
            id = "Vinayak"
        elif(id != 1):
            id = "UKWN"
        
        cv2.cv.PutText(cv2.cv.fromarray(img),str(id),(x,y+h),font,255)

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if(k == 27):
        break;

#closing the window
cam_cv2.release()
cv2.destroyAllWindows()
