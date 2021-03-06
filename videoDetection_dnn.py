import cv2
import numpy as np


path1="Models/caffe/deploy.prototxt";
path2="Models/caffe/res10_300x300_ssd_iter_140000.caffemodel";
conf_threshold=float(0.5)

net = cv2.dnn.readNetFromCaffe(path1,path2)


cap=cv2.VideoCapture(0)

while(True):

    _,frame=cap.read()
    
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            text = "{:.2f}%".format(confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
            if(startY -10 >10):
                y=startY-10
            else:
                y=startY+10
        
            cv2.putText(frame, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    cv2.imshow('Live',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
