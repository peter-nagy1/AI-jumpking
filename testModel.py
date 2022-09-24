import numpy as np
import cv2
from mss import mss
from PIL import Image
import os
import time
import tensorflow as tf

def process_img(oImg):
    pImg = cv2.cvtColor(oImg, cv2.COLOR_BGR2GRAY)
    pImg = cv2.Canny(pImg, threshold1=200, threshold2=500)
    return pImg

def formatImg(img):
    img = np.array([img])

    # scale the data
    img = img/255.0
    
    return img

# Load model
model = tf.keras.models.load_model('multilevel_3stages')

for i in range(5):
    print(i+1)
    time.sleep(1)

mon = {'top' : 40, 'left' : 0, 'width' : 960, 'height' : 700}
sct = mss()
while True :
    sct.get_pixels(mon)
    frame = Image.frombytes( 'RGB', (sct.width, sct.height), sct.image )
    frame = np.array(frame)
    frame = frame[ ::2, ::2, : ] # can be used to downgrade the input

    img = process_img(frame)        # Process

    # Predict img
    fpreImg = formatImg(img)

    preImgPred = model.predict(fpreImg)
    preImgPred = np.argmax(preImgPred)
    print(preImgPred)

    if cv2.waitKey ( 1 ) & 0xff == ord( 'q' ) :
            cv2.destroyAllWindows()