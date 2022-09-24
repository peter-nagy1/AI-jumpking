import numpy as np
import cv2
from mss import mss
from PIL import Image
import os
import time
import imagehash

def process_img(oImg):
    pImg = cv2.cvtColor(oImg, cv2.COLOR_BGR2GRAY)
    pImg = cv2.Canny(pImg, threshold1=200, threshold2=500)
    return pImg

def formatImg(img):
    img = np.array([img])

    # scale the data
    img = img/255.0
    
    return img

def similar_images(img1, img2):

    img1 = Image.fromarray(img1)
    img2 = Image.fromarray(img2)
    hash1 = imagehash.phash(img1)
    hash2 = imagehash.phash(img2)
    return hash1 - hash2


# Load good move
gm = np.load("goodMoves0.npy" , allow_pickle=True)

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

    fpreImg = formatImg(img)

    print(similar_images(img, gm[0][0]))

    if cv2.waitKey ( 1 ) & 0xff == ord( 'q' ) :
            cv2.destroyAllWindows()


