# Ethan Bensman
# github.com/ebenz99

import numpy as np
import cv2
import subprocess
import sys
import time
import os
from collections import Counter
import matplotlib.pyplot as plt


# Set up variables
cap = cv2.VideoCapture(0)
lockCounter = 0
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(os.getcwd()+"/recognizers/face-trainner.yml")

lockCounterv=0

arr = []


while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()
	
	# Convert frames to black and white image
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	cv2.imshow('frame',gray)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	# Draw rectangle on face, not necessary
	for (x,y,w,h) in faces:   
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)  
		roi_gray = gray[y:y+h, x:x+w]
		r = cv2.equalizeHist(roi_gray)
		id_, conf = recognizer.predict(r)
		print(conf)
		if conf>=58 and conf<=62:
			arr.append(int(conf))
			lockCounter=0
			break
		else:
			lockCounter+=1
		#if int(conf) not in dic:
		#	dic[int(conf)] == 0
		#ic[int(conf)]+=1
		arr.append(int(conf))
	print(lockCounter)
	if lockCounter >= 25:
		break
		subprocess.call('/System/Library/CoreServices/Menu\ Extras/User.menu/Contents/Resources/CGSession -suspend',shell=True)
		sys.exit()
	
	# Display the resulting frame, used for testing
	cv2.imshow('LockAway', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break



# Clean up
cap.release()
cv2.destroyAllWindows()
""" Analytics
x = sorted(Counter(arr).items())
print(x)
labels, values = zip(*x)
indexes = np.arange(len(labels))
width = 1
plt.bar(indexes, values, width)
plt.xticks(indexes + width * 0.5, labels)
plt.show()
"""