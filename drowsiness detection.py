import cv2
from keras.models import load_model
import os
import numpy as np
from pygame import mixer
import time 


mixer.init()
sound = mixer.Sound('alarm.wav')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
leye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')
reye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  # Changed to frontalface_default.xml
 
model = load_model('models/cnncat2.h5')

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
score = 0
thicc = 2
sound_playing = False

while True:
    ret, frame = cap.read()
    height, width = frame.shape[:2]
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        leyes = leye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(10, 10))
        for (lex, ley, lew, leh) in leyes:
            cv2.rectangle(roi_color, (lex, ley), (lex + lew, ley + leh), (255, 0, 0), 2)
            leye_img = roi_gray[ley:ley+leh, lex:lex+lew]
            leye_img = cv2.resize(leye_img, (24, 24))
            leye_img = leye_img / 255.0
            leye_img = np.reshape(leye_img, (1, 24, 24, 1))
            prediction = np.argmax(model.predict(leye_img), axis=-1)
            if prediction == 1:  # Eye closed
                score -= 1  # Decrease score when eyes are closed
                cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

            else:
                score += 1  # Increase score when eyes are open
                cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)


        reyes = reye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(10, 10))
        for (rex, rey, rew, reh) in reyes:
            cv2.rectangle(roi_color, (rex, rey), (rex + rew, rey + reh), (0, 255, 0), 2)
            reye_img = roi_gray[rey:rey+reh, rex:rex+rew]
            reye_img = cv2.resize(reye_img, (24, 24))
            reye_img = reye_img / 255.0
            reye_img = np.reshape(reye_img, (1, 24, 24, 1))
            prediction = np.argmax(model.predict(reye_img), axis=-1)
            if prediction == 1:  # Eye closed
                score -= 1  # Decrease score when eyes are closed
                cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

            else:
                score += 1  # Increase score when eyes are open
                cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

        mouths = mouth_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=22)
        for (mx, my, mw, mh) in mouths:
            cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (0, 0, 255), 2)
            mouth_img = roi_gray[my:my+mh, mx:mx+mw]
            mouth_img = cv2.resize(mouth_img, (24, 24))
            mouth_img = mouth_img / 255.0
            mouth_img = np.reshape(mouth_img, (1, 24, 24, 1))
            prediction = np.argmax(model.predict(mouth_img), axis=-1)
            if prediction == 1:
                score += 1
                #cv2.putText(roi_color, 'Mouth Open', (mx, my - 10), font, 0.7, (0, 255, 0), 2)
            else:
                score -= 1
                #cv2.putText(frame, "Mouth Closed", (x, y - 10), font, 0.7, (0, 0, 255), 2)

    if score < 0:
        score = 0   
    
    cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    
    if score > 15:
        try:
            sound.play()
            
        except:
            pass
        
        if thicc < 16:
            thicc += 2
        else:
            thicc -= 2
            if thicc < 2:
                thicc = 2
        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), thicc) 
    elif score <= 15:
        sound_playing = False

    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()