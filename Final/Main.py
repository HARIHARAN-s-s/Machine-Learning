import cv2
import time, math, numpy as np
import HandTrackingModule as htm
import pyautogui, autopy
import alsaaudio  # Linux library to control audio

# Initialize camera and hand detector
wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0
smoothening = 7 

detector = htm.handDetector(maxHands=1, detectionCon=0.85, trackCon=0.8)

# Initialize volume control
mixer = alsaaudio.Mixer('Master')
volRange = (0, 100)  # Correct range for alsaaudio
minVol, maxVol = volRange
hmin, hmax = 50, 200
volBar, volPer = 400, 0
vol = 0
color = (0, 215, 255)

tipIds = [4, 8, 12, 16, 20]
mode, active = '', 0

pyautogui.FAILSAFE = False

def putText(img, mode, loc=(250, 450), color=(0, 255, 255)):
    cv2.putText(img, str(mode), loc, cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, color, 3)

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    fingers = []

    if len(lmList) != 0:
        # Thumb detection logic
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        if fingers == [0, 0, 0, 0, 0] and active == 0:
            mode = 'N'
        elif fingers == [0, 1, 0, 0, 0] and active == 0:
            mode = 'Scroll'
            active = 1
        elif fingers == [1, 1, 0, 0, 0] and active == 0:
            mode = 'Volume'
            active = 1
        elif fingers == [1, 1, 1, 1, 1] and active == 0:
            mode = 'Cursor'
            active = 1

    # Scroll Mode
    if mode == 'Scroll':
        active = 1
        putText(img, mode)
        cv2.rectangle(img, (200, 410), (245, 460), (255, 255, 255), cv2.FILLED)
        if fingers == [0, 1, 0, 0, 0]:
            putText(img, mode='U', loc=(200, 455), color=(0, 255, 0))
            pyautogui.scroll(300)
        elif fingers == [0, 1, 1, 0, 0]:
            putText(img, mode='D', loc=(200, 455), color=(0, 0, 255))
            pyautogui.scroll(-300)
        elif fingers == [0, 0, 0, 0, 0]:
            active = 0
            mode = 'N'

    # Volume Mode
    if mode == 'Volume':
        active = 1
        putText(img, mode)
        if fingers[-1] == 1:  # Pinky up exits Volume Mode
            active = 0
            mode = 'N'
        else:
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]
            length = math.hypot(x2 - x1, y2 - y1)

            # Correct volume calculation
            vol = np.interp(length, [hmin, hmax], [minVol, maxVol])
            volBar = np.interp(vol, [minVol, maxVol], [400, 150])
            volPer = np.interp(vol, [minVol, maxVol], [0, 100])

            # Ensure volume is within range
            volN = max(min(int(vol), maxVol), minVol)
            mixer.setvolume(volN)

            # Display visual indicators
            cv2.rectangle(img, (30, 150), (55, 400), (209, 206, 0), 3)
            cv2.rectangle(img, (30, int(volBar)), (55, 400), (215, 255, 127), cv2.FILLED)
            cv2.putText(img, f'{int(volPer)}%', (25, 430), cv2.FONT_HERSHEY_COMPLEX, 0.9, (209, 206, 0), 3)

    # Cursor Mode
    if mode == 'Cursor':
        active = 1
        putText(img, mode)
        if fingers[1:] == [0, 0, 0, 0]:  # Only index finger detected
            active = 0
            mode = 'N'
        else:
            x1, y1 = lmList[8][1], lmList[8][2]
            w, h = autopy.screen.size()
            X = int(np.interp(x1, [110, 620], [0, w - 1]))
            Y = int(np.interp(y1, [20, 350], [0, h - 1]))
            autopy.mouse.move(X, Y)
            if fingers[0] == 0:  # Thumb down -> Click
                pyautogui.click()

    # FPS Display
    cTime = time.time()
    fps = 1 / (cTime - pTime + 0.01)
    pTime = cTime

    cv2.putText(img, f'FPS:{int(fps)}', (480, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)
    cv2.imshow('Hand LiveFeed', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
