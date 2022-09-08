import cv2
import mediapipe as mp
import time
class handdetector():
    def __init__(self,mode=False,maxHands=2,model_complexity=1,detectionCon=0.5,trackCon=0.5):
        self.mode=mode
        self.maxHands=maxHands
        self.model_complexity=model_complexity
        self.detectionCon=detectionCon
        self.trackCon=trackCon
        self.mphands=mp.solutions.hands
        self.hands=self.mphands.Hands(self.mode,self.maxHands,self.model_complexity,self.detectionCon,self.trackCon)
        self.mpdraw=mp.solutions.drawing_utils

    def findHands(self,frame,draw=True):
        frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        self.results=self.hands.process(frame_rgb)
        if self.results.multi_hand_landmarks:
            for lndmrks in self.results.multi_hand_landmarks:
                if draw:
                   self.mpdraw.draw_landmarks(frame,lndmrks,self.mphands.HAND_CONNECTIONS) 
        return frame
    
    def findPosition(self,frame,handNo=0,draw=True):
        lmlist=[]
        if self.results.multi_hand_landmarks: 
               myhands=self.results.multi_hand_landmarks[handNo]
               for id,ldm in enumerate(myhands.landmark):
                h,w,c=frame.shape
                x,y,z=ldm.x,ldm.y,ldm.z
                cx,cy=int(w*x),int(h*y)
                lmlist.append([id ,cx,cy])
                if draw:
                 cv2.circle(frame,(cx,cy),10,(0,0,255),cv2.FILLED)
                
        return lmlist

def main():
    ctime=0
    ptime=0
    cam=cv2.VideoCapture(0)
    detector=handdetector()
    while(True):
        ret,frame=cam.read()
       
        frame=detector.findHands(frame)
        lmlist=detector.findPosition(frame)
        if len(lmlist)!= 0:
         print(lmlist[4])
        ctime=time.time()
        fps=1/(ctime-ptime)
        ptime=ctime
        
        cv2.putText(frame,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,2,(0,0,0),3,)  
        cv2.imshow("image",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cam.release()
            cv2.destroyAllWindows()

if __name__=="__main__":
    main()