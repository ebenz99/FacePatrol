# Hayden Riewe
# github.com/hriewe
# hrcyber.tech

import numpy as np
import cv2
import subprocess
import sys
import time
import os

# Set up variables
cap = cv2.VideoCapture(0)
lockCounter = 0
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(os.getcwd()+"/recognizers/face-trainner.yml")

lockCounterv=0

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
		id_, conf = recognizer.predict(roi_gray)
		print(conf)
		if conf>=50 and conf<=60:
			lockCounter=0
			break
		else:
			lockCounter+=1


	print(lockCounter)
	if lockCounter >= 10:
		exit(1) #for testing purposes
		subprocess.call('/System/Library/CoreServices/Menu\ Extras/User.menu/Contents/Resources/CGSession -suspend',shell=True)
		sys.exit()
	
	# Display the resulting frame, used for testing
	cv2.imshow('LockAway', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break



# Clean up
cap.release()
cv2.destroyAllWindows()