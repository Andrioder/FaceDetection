# FaceDetection
Face Detection in python using OpenCV
Steps:
1. First make 2 folders TrainerXML and USERINFO
2. Then run the Face_Dect.py, it will create the training data
3. Then run the train.py, form this the model will start traing on the data
4. THen run the Run_Dect.py


if the webcam is not opening then change the webcam id here form Face_dect.py and Run_Dect.py\
cam_cv2 = cv2.VideoCapture(webcamid) for ex: cam_cv2 = cv2.VideoCapture(0)

and accoring to you change the name here as this model detects the faces according to the id
if (id == 1):
            id = "Vinayak"
        elif(id != 1):
            id = "UKWN"
