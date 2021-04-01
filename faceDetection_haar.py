import cv2


face_cascade=cv2.CascadeClassifier('Models/HaarCascade/haarcascade_frontalface_alt2.xml')
img =cv2.imread('test.jpg',1)
img=cv2.resize(img,(960,540))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)
faces=face_cascade.detectMultiScale(gray,1.05,3)

for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)


cv2.imshow('Hello',img)
cv2.waitKey(10000)
cv2.destroyAllWindows()
