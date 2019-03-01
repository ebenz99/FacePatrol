#adapted from @jmitchel3 on Github
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
			if label not in labelIDs:
				labelIDs[label] = currentID
				currentID+=1
			id_ = labelIDs[label]
			pil_image = Image.open(path).convert("L")

			size = (550, 550)
			final_image = pil_image.resize(size, Image.ANTIALIAS)
			image_array = np.array(final_image, "uint8")
			#print(image_array)
			#print(image_array)
			faces = face_cascade.detectMultiScale(image_array)
			for arr in faces:
				print(arr)
				for i in range(0,len(arr)//4):
					roi = image_array[arr[(4*i)+1]:arr[(4*i)+1]+arr[(4*i)+3], arr[(4*i)]:arr[(4*i)+1]+arr[(4*i)+2]]
					x_train.append(roi)
					y_labels.append(id_)
#print(x_train)
with open(os.getcwd()+"/pickles/face-labels.pickle", 'wb') as f:
	pickle.dump(labelIDs, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save(os.getcwd()+"/recognizers/face-trainner.yml")
