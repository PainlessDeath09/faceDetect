import cv2

facePath = "C:\Haar\haarcascade_frontalface_default.xml"
smilePath = "C:\Haar\haarcascade_smile.xml"
faceCascade = cv2.CascadeClassifier(facePath)
smileCascade = cv2.CascadeClassifier(smilePath)

cap = cv2.VideoCapture(cv2.CAP_DSHOW)
cap.set(3,640)
cap.set(4,480)



while (cap.isOpened()):

    ret, frame = cap.read()
    img = frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray,1.3,5)
    


    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        smile = smileCascade.detectMultiScale(roi_gray,1.7,22)

        for (x, y, w, h) in smile:
            print("Found"+ str(len(smile))+ "smiles!")
            cv2.rectangle(roi_color, (x, y), (x+w, y+h), (255, 0, 0), 1)

    cv2.imshow('Smile Detector', frame)
    c = cv2.waitKey(7) % 0x100
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
