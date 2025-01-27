from ultralytics import YOLO
import cv2 as cv
import math
import cvzone
#from roboflow import Roboflow

capture = cv.VideoCapture(1)

classNames = ['black-bishop', 'black-king', 'black-knight', 'black-pawn', 'black-queen', 'black-rook', 'black-root', 'chess-board', 'white-bishop', 'white-king', 'white-knight', 'white-pawn', 'white-queen', 'white-rook']
# I TRIED TO TEST MY ROBOFLOW MODEL IN HERE BUT YOLO MODEL WHICH I TRAINED WITH MY CHESS DATASET IS WORKING BETTER
# rf = Roboflow(api_key="2pFN3kK0G95yzTjEdY7R")
# project = rf.workspace().project("chess-pieces-vc3td/1")
# model = project.version(v1).model
model = YOLO("model.pt")

while True:
    isTrue, frame = capture.read()
    results = model(frame,stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv.rectangle(frame,(x1,y1),(x2,y2),(255,0,255),3)
            
            # Confidence
            conf = math.ceil((box.conf[0]*100)/100)
            # Class Name
            cls =int(box.cls[0])

            cvzone.putTextRect(frame, f'{classNames[cls]} {conf}',(max(0,x1), max(35,y1)), scale=1,thickness=1)
    cv.imshow('Cam',frame)
    if cv.waitKey(20) & 0xFF==ord('d'):
        break
capture.release()
cv.destroyAllWindows()