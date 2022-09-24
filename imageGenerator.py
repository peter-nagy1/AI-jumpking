import numpy as np
import cv2
from mss import mss
from PIL import Image
import os
import time

def process_img(oImg):
    pImg = cv2.cvtColor(oImg, cv2.COLOR_BGR2GRAY)
    pImg = cv2.Canny(pImg, threshold1=200, threshold2=500)
    return pImg

def main():
    imgs = []

    mon = {'top' : 40, 'left' : 0, 'width' : 960, 'height' : 700}
    sct = mss()
    while True :
        sct.get_pixels(mon)
        frame = Image.frombytes( 'RGB', (sct.width, sct.height), sct.image )
        frame = np.array(frame)
        frame = frame[ ::2, ::2, : ] # can be used to downgrade the input

        img = process_img(frame)        # Process

        imgs.append(img)

        if len(imgs) == 1000:
            np.save("training/level3.npy", imgs)
            break


for i in range(5):
    print(i+1)
    time.sleep(1)

main()