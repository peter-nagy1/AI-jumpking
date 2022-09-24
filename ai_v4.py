import numpy as np
import cv2
from mss import mss
from PIL import Image
import time
import random
import input
import tensorflow as tf
import imagehash
from collections import deque

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

# Checks if the last 2 elements are the same
def sameLastMove(seq):
    if similar_images(seq[0][0], seq[1][0]) and seq[0][1] == seq[1][1] and seq[0][2] == seq[1][2]:
        print("Move done twice in a row!")
        return True
    return False

# Checks if an element is in the sequence:
def elemInMoves(elem, moves):
    for move in moves:
        if np.array_equal(move[0], elem[0]) and move[1] == elem[1] and move[2] == elem[2]:
            print("Element already existent!")
            return True
    return False

def improved(preLvl, preStage, postLvl, postStage):
    preNum = int(str(preLvl) + str(preStage))
    postNum = int(str(postLvl) + str(postStage))

    return postNum > preNum

def worsened(preLvl, preStage, postLvl, postStage):
    preNum = int(str(preLvl) + str(preStage))
    postNum = int(str(postLvl) + str(postStage))

    return postNum < preNum

# --------------- MAIN ----------------

# Load model
modelstages = tf.keras.models.load_model('multilevel_3stages')
modellvls = tf.keras.models.load_model('2lvls')


# variables
goodMoves = []

# Maintain a sequence of last n moves
seq = deque()
SEQ_LEN = 5

# Load good moves
try:
    goodMoves = np.load("/history/good_moves.npy")
except IOError:
    print("No existing move file found.")

# Wait time
for i in range(5):
    print(i+1)
    time.sleep(1)

# Execute moves
for i in range(500):

    print("Move:", i, "----------")

    # Read image
    preImg = getImg()
    fpreImg = formatImg(preImg)


    # Predictions
    preImgPredlvl = modellvls.predict(fpreImg)
    preImgPredlvl = np.argmax(preImgPredlvl)

    preImgPredStage = modelstages.predict(fpreImg)
    preImgPredStage = np.argmax(preImgPredStage)
    print("prePred:", preImgPredlvl, preImgPredStage)


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

    # Add move to sequence
    seq.appendleft([preImg, move, sz])
    # Adjust sequence
    if len(seq) > SEQ_LEN:
        seq.pop()

    # Wait until move animation is done
    time.sleep(2)


    # Read image
    postImg = getImg()
    fpostImg = formatImg(postImg)


    # Predictions
    postImgPredlvl = modellvls.predict(fpostImg)
    postImgPredlvl = np.argmax(postImgPredlvl)

    postImgPredStage = modelstages.predict(fpostImg)
    postImgPredStage = np.argmax(postImgPredStage)
    print("postPred:", postImgPredlvl, postImgPredStage)

    # Save image and move
    if improved(preImgPredlvl, preImgPredStage, postImgPredlvl, postImgPredStage) and not match and move > 1:
        for elem in seq:
            if not elemInMoves(elem, goodMoves):
                goodMoves.append(elem)
        print("Learned new sequence of moves!")
 
    # Delete if saved image is not good
    if match and (worsened(preImgPredlvl, preImgPredStage, postImgPredlvl, postImgPredStage) or sameLastMove(seq)):
        for i in range(len(goodMoves)):
            if similar_images(goodMoves[i][0], preImg):
                goodMoves.pop(i)
                print("Deleted move!")
                break

# Save good moves in a file
np.save("/history/good_moves.npy", goodMoves)

# TODO Implement 0.1 step size of moves
# TODO Implement bad moves