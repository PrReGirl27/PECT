#Importing any neccesary packages
import cv2
from ultralytics import YOLO
import time
from playsound3 import playsound
import threading



model = YOLO("pect_model.pt") 
cap = cv2.VideoCapture("testingGuy.mp4") 
SleepingTime = None
lastTimeSinceAudioHasPlayed = 0 
cooldown = 1

def soundPlay():
    playsound("wakeUpBeep.mp3")

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("newTest.mp4", fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    annotated_frame = results[0].plot()
    #variables that update every frame
    sleep = False
    TimeSpentSleep = 0
    #functions
    for box in results[0].boxes:
        id= int(box.cls.item())
        if model.names[id] == "sleep":
            sleep = True
        
    if sleep == True:
        if SleepingTime is None:
            SleepingTime = time.time()
    else:
        SleepingTime = None
    
    if SleepingTime is not None:
        TimeSpentSleep = time.time() - SleepingTime
    else:  
        TimeSpentSleep = 0
        
    cv2.putText(annotated_frame,
                f"Time Sleep: {TimeSpentSleep:.1f}s",
                (20,100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0), 
                3)
    if TimeSpentSleep >= 1:

        cv2.putText(annotated_frame,"Wake Up!!",(20,200),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0), 3)

        if time.time()- lastTimeSinceAudioHasPlayed > cooldown:
            threading.Thread(target=soundPlay, daemon=True).start() 
            lastTimeSinceAudioHasPlayed = time.time()
        
        
        
    cv2.imshow("Video", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
     break
out.release()
cap.release()
cv2.destroyAllWindows()