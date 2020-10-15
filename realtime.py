from keras.models import load_model
import cv2
import numpy as np
REV_CLASS_MAP={
    0:'nomask',1:'mask'
}

def mapper(val):
    return REV_CLASS_MAP[val]


model=load_model('mask_nomask.h5')

cap = cv2.VideoCapture(0)


while True:
    returnVal,frame=cap.read()
    if not returnVal:
        continue
        # if video is not capturing the loop will skip this exicution
    
    cv2.rectangle(frame,(100,100),(500,500),(255,255,200),3)
    img=frame[100:500,100:500]
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img=cv2.resize(img, (288, 288))
    pred=model.predict(np.array([img]))
    code=np.argmax(pred[0])
    classN=mapper(code)
    print(classN)

    font = cv2.FONT_ITALIC
    if classN=='mask':
        cv2.putText(frame,'You Are with Mask',(5, 50), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    elif classN=='nomask':
        cv2.putText(frame,'You Are withot mask',(5, 50), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame,'No person in frame',(5, 50), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    
    cv2.imshow("RealTime",frame)
    k=cv2.waitKey(10)
    
