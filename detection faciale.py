import cv2
cap = cv2.VideoCapture(0) 	 #Le nombre de device 0
cap.set(3, 640) 			#largeur de la fenêtre de la capture
cap.set(4, 480) 			#hauteur de la fenêtre de la capture
# charger les fichiers XML qui font la détection des visages
face_cascade=cv2.CascadeClassifier('C:\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml') #charger les fichiers d'apprentissage et détection
eye_cascade=cv2.CascadeClassifier('C:\opencv\sources\data\haarcascades\haarcascade_eye.xml')
smileCascade=cv2.CascadeClassifier('C:\opencv\sources\data\haarcascades\haarcascade_smile.xml')
face_id = input('\n enter user id end press <return> ==>  ')
count=0 #nombre de visages

while(True):
    ret, frame = cap.read() # Capture frame
    # charger une capture vidéo à multiples couleurs et détecter les visages et les yeux dans la capture
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.4, minNeighbors=3 ,minSize=(30,30) )
    print(len(faces))
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        # enregistrer les captures dans le fichier dataset
        count += 1
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
    
    smile = smileCascade.detectMultiScale(roi_gray,scaleFactor= 1.16,minNeighbors=35,minSize=(25, 25))
    for (x2, y2, w2, h2) in smile:
        cv2.rectangle(roi_color, (x2, y2), (x2+w2, y2+h2), (255, 0, 0), 2)
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        cv2.imshow('frame',frame)
    

    # Condition pour arrêter la capture
    if cv2.waitKey(1000) & 0xFF == ord('q'):
        break
cam.release()  # arrêter le fonctionnement du caméra (device 0)
cv2.destroyAllWindows() 
    
        	
