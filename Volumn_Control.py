import cv2
import mediapipe
import numpy as np
import  HandTrackingModule as htm
import math
from comtypes import CLSCTX_ALL # type: ignore
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume # type: ignore

wCam=640
hCam=480

cap=cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
#
minVol=volRange[0]
maxVol=volRange[1]
volBar=0
vol=0

detector=htm.handDetector(detectionCon=0.8)

while True:
    success, img= cap.read()
    img = detector.findHands(img)

    lmList=detector.findPosition(img,draw=False)
    if len(lmList) !=0:
        # print(lmList[4], lmList[8])
        x1, y1=lmList[4][1], lmList[4][2]
        x2, y2=lmList[8][1], lmList[8][2]
        cx, cy = (x1+x2)//2, (y1+y2)//2

        cv2.circle(img, (x1, y1), 15, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (cx, cy), 15, (255, 0, 0), cv2.FILLED)
        cv2.line(img,(x1, y1),(x2, y2),(255, 0, 255), 3)

        length=math.hypot(x2-x1,y2-y1)
        # print(length)
        #handRange = 18 to 160
        #VolRange = -63.5 to 0

        vol=np.interp(length,[18,160],[minVol,maxVol])
        volBar=np.interp(length,[18,160],[400,150])
        print(int(length),vol)
        volume.SetMasterVolumeLevel(vol, None)
        if length<40:
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
        if length > 160:
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

    cv2.rectangle(img,(50,150),(85,400),(0,255,0),3)
    cv2.rectangle(img,(50,int(volBar)),(85,400),(0,255,0), cv2.FILLED)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
    