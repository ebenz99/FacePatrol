import cv2
import os
import numpy as np
from PIL import Image
import pickle

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

basedir = os.getcwd()
imagedir = os.path.join(basedir,'ethanface')


currentID = 0
labelIDs = {}
y_labels = []
x_train = []



for root, dirs, files in os.walk(imagedir):
	for file in files:
		if file.endswith("jpg"):
			path = os.path.join(root,file)
			label = os.path.basename(root).replace(" ","-").lower()
			#print(path)
			#print(label)
			if label in labelIDs:
				pass
			else:
				labelIDs[label] = currentID
				currentID+=1
			id_ = labelIDs[label]
			pil_image = Image.open(path).convert("L")

			size = (550, 550)
			final_image = pil_image.resize(size, Image.ANTIALIAS)
			image_array = np.array(final_image, "uint8")
			#print(image_array)
			print(id_)
			faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

			for (x,y,w,h) in faces:
				roi = image_array[y:y+h, x:x+w]
				x_train.append(roi)
				y_labels.append(id_)

print(y_labels)
print(x_train)
