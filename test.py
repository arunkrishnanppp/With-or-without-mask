from keras.models import load_model
import cv2
import numpy as np
import sys
file_path=sys.argv[1]
print(file_path)
# D:\PYTHON\ML\WITH_MASK\Image_data\mask\3.jpg


rev_class_map={
    0:'nomask',1:'mask'
}

def classMapper(val):
    return rev_class_map[val]


model=load_model('mask_nomask.h5')
print(model)


# prepare an image fo rtesying
img=cv2.imread(file_path)
cv2.imshow('imagei',img)
cv2.waitKey(100)
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img=cv2.resize(img, (288, 288))
pred=model.predict(np.array([img]))
print(pred)
code=np.argmax(pred[0])
classType = classMapper(code)

print("You Are with {}".format(classType))
