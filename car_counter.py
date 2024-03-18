import cv2
import cvzone
import math
import time
from ultralytics import YOLO
from sort import *       # * = everything
import numpy as np 

# Use a smaller model for faster inference
model = YOLO("computer_vision/yolo-weights/yolov8n.pt")
total_count=[]
#TRACKER
tracker=Sort(max_age=20, min_hits=3, iou_threshold=0.3)      

#LINE
limits=[400,297,673,297]

#max_age= It represents the maximum age (in terms of frames) that a track is allowed to live without an associated detection.
#max_hits= This parameter sets the maximum number of consecutive misses (frames without an associated detection) a track can have 
#before it is considered for deletion. If the track has more consecutive misses than this value, it may be removed.
#iou_threshold= It stands for Intersection over Union (IoU) threshold. IoU is a measure of how much two bounding boxes overlap.



# Resize the input images to a smaller size
cap = cv2.VideoCapture("computer_vision/object_detection/car_counter/cars.mp4")
width = 640 # Adjust based on your needs
height = 480 # Adjust based on your needs
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench"
              ]

mask=cv2.imread("computer_vision/object_detection/car_counter/mask.png")


while True:
    success, img = cap.read()
    
    #we'll resize the masked image to the size of the original frame and mask it to the img with bitwise_and and now it will detect only
    #the cars of that particular region not anythiing else 
    mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
    imgregion=cv2.bitwise_and(img,mask_resized)
    results = model(imgregion, stream=True)
    
    detections= np.empty((0,5))
    #then if it is a car,truck etc and conf>0.3 then only we'll save it to out detections list line 75

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            print(x1, y1, x2, y2)
        
            w, h = x2 - x1, y2 - y1
        
            conf = math.ceil((box.conf[0] * 100)) / 100
            print(conf)
        
            cls = int(box.cls[0])
        
            if 0 <= cls < len(classNames):  # Check if cls is a valid index
                currentClass = classNames[cls]
            
                if currentClass in ["car", "truck", "bus", "motorbike"] and conf > 0.3:
                # ... rest of your code ...

                
                #cvzone.putTextRect(img,f'{currentClass} {conf}',(max(0,x1),max(35,y1)),scale=0.7,thickness=1,offset=3) 
                #cvzone.cornerRect(img,(x1,y1,w,h),l=9,rt=5)   
                
                    currArr=np.array([x1,y1,x2,y2,conf])
                    detections=np.vstack((detections,currArr))
            
            #we'll constrain our values to get good results 
            #we'll select only the vehicles passing through the main road not the roas on the left side of the video 
            #for this we'll create a masking file in canva by making only the region we want white and rest to black 
            #checkout the file mask.png 
            
            
    results=tracker.update(detections)  
    cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(15, 82, 186),1)  
    for results in results:
        x1,y1,x2,y2,id=results
        x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
        print(results)  
        w,h=x2-x1,y2-y1  
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=8)
        
        cx,cy=x1+w//2,y1+h//2
        cv2.circle(img,(cx,cy),5,(255,255,255),cv2.FILLED)
        if limits[0]<cx<limits[2] and limits[1]-15<cy<limits[3]+15:    #we'll give a margin bcz sometimes a fast car did not touch the line so it
                                                                       #will not count it  so we'll create a region to cmake it count 
            if total_count.count(id)==0:            # we'll find the count of id in there if it is 0 that is it is new so we'll count+=1
                total_count.append(id)
                cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(52, 255, 152),1)    #as it detects it will turn line to green
            
            
    cvzone.putTextRect(img, f'Count: {len(total_count)}', (50,50))
        
    cv2.imshow("Image", img)
    #cv2.imshow("ImageRegion", imgregion)
    cv2.waitKey(1)
