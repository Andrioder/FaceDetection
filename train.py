import os
import numpy as np
from PIL import Image
import cv2

detector = cv2.createLBPHFaceRecognizer()
path = 'USERINFO'

def getTheImage(pth):
    image_Paths = [os.path.join(pth,f) for f in os.listdir(pth)]
    user_ID = []
    user_FACES = []
    for image_PTH in image_Paths:
        faceIMG = Image.open(image_PTH).convert('L')
        faceNP = np.array(faceIMG,'uint8')   #cv can only understand in np.array

        #Getting the user id form the path name or the image name
        #[-1] this means our trainer will read the file name frome Rght to Lft
        id = int(os.path.split(image_PTH)[-1].split('_')[1])
        user_FACES.append(faceNP)
        user_ID.append(id)

        #optional: showing the training img
        cv2.imshow('TRAINING',faceNP)
        cv2.waitKey(10)
        return user_ID, user_FACES
    

user_id, user_faces = getTheImage(path)
detector.train(user_faces, np.array(user_id))
detector.save('TrainerXML/train_data.xml')
cv2.destroyAllWindows()
