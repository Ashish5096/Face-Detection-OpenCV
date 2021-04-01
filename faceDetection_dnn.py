import cv2
import numpy as np

DNN ="TF"

if DNN =="TF":
    configFile="Models/caffe/deploy.prototxt";
    modelFile="Models/caffe/res10_300x300_ssd_iter_140000_fp16.caffemodel";
    net = cv2.dnn.readNetFromCaffe(configFile,modelFile)

else:
    configFile="Models/TensorFlow/tflite_graph.pbtxt";
    modelFile= "Models/TensorFlow/tflite_graph.pb";
    net = cv2.dnn.readNetFromTensorflow(configFile,modelFile)

conf_threshold=float(0.5)

frame =cv2.imread("test.jpg",1)
frame=cv2.resize(frame,(960,540))
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

cv2.imshow("Output", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

