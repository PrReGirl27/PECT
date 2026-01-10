#Importing any neccesary packages
import cv2
from ultralytics import YOLO
import time
from playsound3 import playsound
import threading



model = YOLO("pect_model.pt") #my Yolo Model
cap = cv2.VideoCapture("testingGuy2.mp4") #which video my model will took at
SleepingTime = None #will be used to track time sleep

#varibles to track when the play the alarm when someone is sleep
lastTimeSinceAudioHasPlayed = 0 
cooldown = 1

#function that plays the alert to wake the persion up
def soundPlay():
    playsound("wakeUpBeep.mp3")

#process to save each run and test. it tells the program how to save it and what to same it as
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("newTest.mp4", fourcc, fps, (width, height))

#while the video is playing
while cap.isOpened():
    #checks to make sure that the frame is avavible to be used
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame) #the model runs inference on each frame of the video and makes predictions/dections
    annotated_frame = results[0].plot() #draws the predictions/detections on the spefice frame for the user to see

    #variables that update every frame
    sleep = False #person isn't sleep, the variable is checked and updated each frame
    TimeSpentSleep = 0 #other essential varible for tracking how much time the persion has spent sleeping
    #functions that are updated each frame

    #checks to see if the detections/predictions have the id of "sleep" in each frame
    for box in results[0].boxes: 
        id= int(box.cls.item())
        if model.names[id] == "sleep": #if the model has the id of "sleep," then the sleep variable is set to true
            sleep = True

    #if person is sleeping...   
    if sleep == True:
        if SleepingTime is None: #if SleepingTime still equals None, set it equal to time.time()
            SleepingTime = time.time() #this time.time() is set at a differnt time than when the program was started
    else: #if not, then set SleepingTime to none again
        SleepingTime = None
    
    #if SleepingTime does not equal None...
    if SleepingTime is not None:
        TimeSpentSleep = time.time() - SleepingTime #Subract SleepingTime from time.time()
    else: #if not, set TimeSpentSleep to zero again
        TimeSpentSleep = 0
    #This timer works because SleepingTime will not equal the same as time.time()
    #time.time() is like a stop watch from when the program first started
    #time.time() inside of SleepingTime start a stop watch from when that variable was set to equal time.time
    #therefore, since SleepingTime is updated each frame, we get the accurate amount of time the person was sleep for


    #text draw on each frame
    cv2.putText(annotated_frame, #where to draw
                f"Time Sleep: {TimeSpentSleep:.1f}s", #draws the TimeSpentSleep timer
                (20,100), #where to draw on the frame
                cv2.FONT_HERSHEY_SIMPLEX, #what font to use
                1, #font scale
                (255, 0, 0), #what color to draw 
                3) #thickness
    if TimeSpentSleep >= 1: #if the person has been sleep for one or more seconds, then..

        cv2.putText(annotated_frame, #where to draw
                    "Wake Up!!", #what to draw
                    (20,200), #where to draw on the frame
                    cv2.FONT_HERSHEY_SIMPLEX, #what font to use
                    1,#font scale 
                    (0, 255, 0), #what color to draw
                    3)#thickness
        
        #if it's been over 1 second...
        if time.time()- lastTimeSinceAudioHasPlayed > cooldown:
            threading.Thread(target=soundPlay, daemon=True).start()  #play sound on a different thread
            lastTimeSinceAudioHasPlayed = time.time() #rest lastTimeSinceAudioHasPlayed
        
        
        
    cv2.imshow("Video", annotated_frame) #pulls up the frame on my screen when the code is run

    if cv2.waitKey(1) & 0xFF == ord('q'): #if "q" is clicked, stop the program
     break

#stops everything  
out.release()
cap.release()
cv2.destroyAllWindows()