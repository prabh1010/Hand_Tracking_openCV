import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectCon=0.5, trackCon=0.5):   #maxhands can be changed according to requirements
        self.mode = mode
        self.maxHands = maxHands
        self.detectCon = detectCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils  # method in mp to draw lines between each point on hand

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:  # for each hand detected this will run the loop
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)  # not taking the RGB we will be displaying these lines on BGR (original)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):  # will give id and location of the hand
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)  # positon of center , multiplying with the value to find pixel location

                lmList.append([id, cx, cy])          #each position getting stored in list
                if draw and id == 4:                                      #for the 4th point i.e the tip of thumb
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)     #drawing circles on the ID mentioned (4)
        return lmList


def main():
    cTIME = 0
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 2)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
