import cv2
import numpy as np
from  tracker import *


tracker = EuclideanDistTracker()
# tracker2 = tracker.TrackerMOSSE_create()
# tracker3 = tracker.TrackerCRST_create()
# img = cv2.imread("ryspassport.jpg")
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
object_detector = cv2.createBackgroundSubtractorMOG2(history=150,varThreshold=3000)
classnames=[]
classfile = "coco.names"
with open(classfile,'rt') as f:
    classnames = f.read().strip('\n').split('\n')

configpath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightspath = 'frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weightspath,configpath)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)
total_passed_vehicle = []

region = [(300,300),(300,500),(750,600),(750,400)]

region_id  = set()
while True:
    success,img=cap.read()
    # height,weight,_ = img.shape
    # roi = img[250:500,300,540]

    roi =img[100:600,250:830]
    classids, confs, bbox = net.detect(img,confThreshold=0.6)
    # print(classids,bbox)
    font=cv2.FONT_ITALIC
    if len(classids) !=0:
        for classid, confidence, box in zip(classids.flatten(),confs.flatten(),bbox):
            x,y,w,h = box
            # class_name =classid[0]
            cv2.rectangle(img, box,color=(0,255,0),thickness=2)
            cv2.putText(img,str(classnames[classid-1]),(x, y - 15), cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),2)
            # cv2.putText(img,str (round((confidence*100))), (box[0] + 50, box[1] + 30),
            #             font, 1, (12, 250, 0), 2)
            # detections.append(box)


    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask,254,255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for con in contours:
            area = cv2.contourArea(con)
            if area > 5200:
                # cv2.drawContours(roi, [con],-1,(0,255,0),1)
                x, y, w, h = cv2.boundingRect(con)
                detections.append([x, y, w, h])

    # object tracking
    boxes_ids = tracker.update(detections)
    print(boxes_ids)

    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        total_passed_vehicle.append(id)
        vehile=len(total_passed_vehicle)
        print(total_passed_vehicle)
        print(box_id)


    cv2.putText(img, "vehicles:"+str(len(total_passed_vehicle)), (20,30),cv2.FONT_HERSHEY_SIMPLEX,1 ,(0,255,0),2)
    # cv2.putText(img, "vehicles:" + str(id).upper(), (100, 500), cv2.FONT_HERSHEY_PLAIN, 1, (245, 255, 0), 2)
    # cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
    # inside_region = cv2.pointPolygonTest(np.array(region), (int(bbox[0]), int(bbox[2])), False)
    #
    # if inside_region > 0:
    #     region_id.add()
    # cv2.polylines(img, [np.array(region)], True, (0, 255, 255), 4)
    # vehicle_count = len(region_id)



    cv2.imshow('output',img)
    cv2.imshow("mask",mask)
    cv2.imshow("roi",roi)
    if cv2.waitKey(1) & 0xFF ==ord('z'):
        break

cap.release()
cv2.destroyAllWindows()

# print(height,weight)









    # (box[0]+10,box[1]+30)
    #     cv2.putText(roi,str(id),(x,y - 15),cv2.FONT_HERSHEY_SIMPLEX,1 ,(0,255,0),2)
    #     cv2.rectangle(roi,(x,y),(x+w,y+h),(0,255,0),3)


#         # print(box)
#
# print(detections)


