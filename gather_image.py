import cv2
import os
import sys
import numpy as np

try:
    label=sys.argv[1]
    no_samples=int(sys.argv[2])
    print(no_samples)
except:
    print('Illegal Arguments Pleaes provide Correct Arguments')
    print('Filename.py label no_of_sample')
IMG_SAVE_PATH='image_data'
IMG_CLASS_PATH=os.path.join(IMG_SAVE_PATH,label)


# making directory

try:
    os.mkdir(IMG_SAVE_PATH)
except FileExistsError:
    pass
try:
    os.mkdir(IMG_CLASS_PATH)
except FileExistsError:
    print('All images will be saved in existing folder for the label')


# cpaturing videos as frame

# video capture object
cap=cv2.VideoCapture(0)
count=0
start=False
while True:
    returnValue, frame=cap.read()
    print('Value of count',count)
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if count == no_samples:
        break
        # cap.read() will return true if video captured corrctly so on end will retuirn false
    if not returnValue:
        continue
        # if video is not capturing the loop will skip this exicution
    
    cv2.rectangle(frame,(100,100),(500,500),(255,255,200),3)

    # cv recatngle function (image,startpoint ,end point,color,thickness)


    if start:
        image=frame[100:500,100:500]
        save_path=os.path.join(IMG_CLASS_PATH,'{}.jpg'.format(count+1))
        cv2.imwrite(save_path,image)
        count+=1



    font = cv2.FONT_ITALIC
    cv2.putText(frame,"Collecting {}".format(count),(5, 50), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    # puttext(image upon text show,text,orgin,font,fontscale,color,thickness,typeof LINE_AA)

    

    windowname='Collecting Image -Press s to start collecting and q to quit '
    cv2.imshow(windowname,frame)
    k=cv2.waitKey(10)
    if k==ord('a'):
        start= not start
        # count+=1
    if k==ord('q'):
        break
    
    
    