import numpy as np
import cv2
from mss import mss
from PIL import Image
import time
import random
import input
import tensorflow as tf
import imagehash

def process_img(oImg):
    pImg = cv2.cvtColor(oImg, cv2.COLOR_BGR2GRAY)
    pImg = cv2.Canny(pImg, threshold1=300, threshold2=500)
    return pImg

def getImg():
    mon = {'top' : 40, 'left' : 0, 'width' : 960, 'height' : 700}
    sct = mss()
    while True :
        sct.get_pixels(mon)
        frame = Image.frombytes( 'RGB', (sct.width, sct.height), sct.image )
        frame = np.array(frame)
        frame = frame[ ::2, ::2, : ] # can be used to downgrade the input

        img = process_img(frame)        # Process

        return img

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
    return (hash1 - hash2) < 5


# --------------- MAIN ----------------

# Load model
modelstages = tf.keras.models.load_model('multilevel_3stages')
modellvls = tf.keras.models.load_model('2lvls')

# variables
goodMoves = []      # savedData[0][0] = solution for lvl1_1     # So if we're on that level just repeat the solution

# Wait time
for i in range(5):
    print(i+1)
    time.sleep(1)

# Execute moves
for i in range(100):


    # Read image
    preImg = getImg()
    fpreImg = formatImg(preImg)


    # Predictions
    preImgPredlvl = modellvls.predict(fpreImg)
    preImgPredlvl = np.argmax(preImgPredlvl)

    preImgPredStage = modelstages.predict(fpreImg)
    preImgPredStage = np.argmax(preImgPredStage)
    print(preImgPredlvl, preImgPredStage)


    # Do a move of random size if pre img is not existent in savedData
    # Else do the same move as previously saved
    move = 0
    sz = 0

    match = False
    for gmove in goodMoves:
        if similar_images(gmove[0], preImg):
            move = gmove[1]
            sz = gmove[2]
            input.pickMove(move, sz)
            match = True
            print("Used saved move!")
            break
    if not match:
        sz = random.random()
        move = input.pickRandMove(sz)

    # Wait until move animation is done
    time.sleep(1)


    # Read image
    postImg = getImg()
    fpostImg = formatImg(postImg)


    # Predictions
    postImgPredlvl = modellvls.predict(fpostImg)
    postImgPredlvl = np.argmax(postImgPredlvl)

    postImgPredStage = modelstages.predict(fpostImg)
    postImgPredStage = np.argmax(postImgPredStage)
    print(postImgPredlvl, postImgPredStage)

    # Save image and move
    if (postImgPredStage - preImgPredStage > 0 or postImgPredlvl - preImgPredlvl > 0) and not match and move > 1:
        goodMoves.append([preImg, move, sz])
        print("Learned new move!")
 
    # Delete if saved image is not good
    if match and not (postImgPredStage - preImgPredStage > 0 or postImgPredlvl - preImgPredlvl > 0):
        for i in range(len(goodMoves)):
            if similar_images(goodMoves[i][0], preImg):
                goodMoves.pop(i)
                print("Deleted move!")
                break