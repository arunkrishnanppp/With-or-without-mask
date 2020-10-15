import cv2
import numpy as np
from keras.applications.imagenet_utils import _obtain_input_shape
from keras_squeezenet import SqueezeNet
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.layers import Activation, Dropout, Convolution2D, GlobalAveragePooling2D
from keras.models import Sequential
import tensorflow as tf
import os


image_save_path='image_data'

class_map={
    'nomask':0,
    'mask':1
}
no_class=len(class_map)


def classMapper(val):
    return class_map[val]
def get_model():
    model=Sequential([
        SqueezeNet(input_shape=(288,288,3),include_top=False),
        Dropout(0.5),
        Convolution2D(no_class,(1,1),padding='valid'),
        Activation('relu'),
        GlobalAveragePooling2D(),
        Activation('softmax')
    ])
    return model

# Load image data from filesystem
dataset=[]
for dir in os.listdir(image_save_path):
    path=os.path.join(image_save_path,dir)
    if not os.path.isdir(path):
        continue
    for item in os.listdir(path):
        if item.startswith('.'):
            continue
        img=cv2.imread(os.path.join(path,item))
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img=cv2.resize(img, (288, 288))
        dataset.append([img, dir])
# print(dataset)


data,labels=zip(*dataset)

labels=list(map(classMapper,labels))

print(labels)
# encoding labels using onhat encododing
labels=np_utils.to_categorical(labels)
print(labels)
# getting the model
model=get_model()
model.compile(
    optimizer=Adam(lr=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(np.array(data), np.array(labels), epochs=10)
model.save("rock-paper-scissors-model.h5")




