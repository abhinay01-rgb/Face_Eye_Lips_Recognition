import cv2 #Install OpenCV

cap = cv2.VideoCapture(0) #To use Webcam
cap.set(3, 640)
cap.set(4, 480)
cap.set(10, 100)
#Haarcascade files for detection of a feature
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
lips_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

while True:
    Aquib, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 6)
    
    # For creating rectangle around the specified feature
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        lips = lips_cascade.detectMultiScale(roi_gray, 1.4, 6)
        for (sx, sy, sw, sh) in lips:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 225), 2)

    cv2.imshow('Webcam', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()