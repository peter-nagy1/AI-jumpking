import numpy as np
import cv2
from mss import mss
from PIL import Image
import time
import datetime
import random
import input
import tensorflow as tf
import imagehash
from collections import deque
from win32gui import GetWindowText, GetForegroundWindow, FindWindow, MoveWindow, SetForegroundWindow
import sys


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
    return (hash1 - hash2) < 3

# Checks if the last 2 elements are the same
def sameLastMove(seq):
    if len(seq) >= 2:
        if similar_images(seq[0][0], seq[1][0]) and seq[0][1] == seq[1][1] and seq[0][2] == seq[1][2]:
            print("Move done twice in a row!")
            return True
    return False

# Checks if an element is in the sequence:
def elemInMoves(elem, moves):
    for move in moves:
        if np.array_equal(move[0], elem[0]) and move[1] == elem[1] and move[2] == elem[2]:
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
modellvls = tf.keras.models.load_model('3lvls')


# variables
NUM_LVLS_LEARNED = 3
GAME_NAME = "Jump King"

goodMoves = []
badMoves = []
lvls = [0] * NUM_LVLS_LEARNED
options = input.createOptions()

# Maintain a sequence of last n moves
seq = deque()
SEQ_LEN = 3 # 2-5


# Load good moves
try:
    goodMoves = np.load("history/good_moves.npy", allow_pickle=True).tolist()
    badMoves = np.load("history/bad_moves.npy", allow_pickle=True).tolist()
except IOError:
    print("No existing moves file found!")


# Set game window position
try:
    gameWin = FindWindow(None, GAME_NAME)
    MoveWindow(gameWin, -8, 0, 967 + 8, 761, False)
    SetForegroundWindow(gameWin)
except Exception:
    print("Can't find game window!")

# Wait to set window to foreground
time.sleep(1)

print("STARTING TRAINING")

# Execute moves
for i in range(int(sys.argv[1])):

    # Exit if changed window
    if GetWindowText(GetForegroundWindow()) != GAME_NAME:
        print("Exiting!")
        exit()


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
    # Check in good moves
    for gmove in goodMoves:
        if similar_images(gmove[0], preImg):
            move = gmove[1]
            sz = gmove[2]
            input.pickMove(move, sz)
            match = True
            print("Used saved move!")            
            break
    if not match:

        badMove = True
        opt = options.copy()

        # Remove all the options of moves that are bad
        for bmove in badMoves:
            if similar_images(bmove[0], preImg):
                for mvSz in bmove[1:]:
                    opt.remove(mvSz)
                break

        # Pick move from options
        mvSz = random.randint(0, len(opt)-1)
        move = opt[mvSz][0]
        sz = opt[mvSz][1]

        # Execute move
        input.pickMove(move, sz)

    print("Move:", move, sz)

    # Add move to sequence
    seq.appendleft([preImg, move, sz])
    # Adjust sequence
    if len(seq) > SEQ_LEN:
        seq.pop()

    # Wait until move animation is done
    time.sleep(1)


    # Read image
    postImg = getImg()
    fpostImg = formatImg(postImg)


    # Predictions
    postImgPredlvl = modellvls.predict(fpostImg)
    postImgPredlvl = np.argmax(postImgPredlvl)

    # Save number of times this level was visited
    lvls[postImgPredlvl] += 1

    postImgPredStage = modelstages.predict(fpostImg)
    postImgPredStage = np.argmax(postImgPredStage)
    print("postPred:", postImgPredlvl, postImgPredStage)

    # Save good image and move
    if improved(preImgPredlvl, preImgPredStage, postImgPredlvl, postImgPredStage) and not match and move > 1:
        for elem in seq:
            if not elemInMoves(elem, goodMoves):
                goodMoves.append(elem)
        print("Learned sequence of moves!")
 
    # Save bad image and move
    if worsened(preImgPredlvl, preImgPredStage, postImgPredlvl, postImgPredStage):

        found = False
        for bmove in badMoves:
            if similar_images(bmove[0], preImg):
                if (move, sz) not in bmove[1:]:
                    bmove.append((move, sz))
                found = True
                break

        if not found:
            badMoves.append([preImg, (move, sz)])

        # Empty sequence
        seq.clear()
        print("Learned bad move!")

    # Delete if saved image is not good
    if match and (worsened(preImgPredlvl, preImgPredStage, postImgPredlvl, postImgPredStage) or sameLastMove(seq)):
        for i in range(len(goodMoves)):
            if similar_images(goodMoves[i][0], preImg):
                goodMoves.pop(i)
                print("Deleted move!")
                break
    

    
# Save good moves in a file
np.save("history/good_moves.npy", goodMoves)
np.save("history/bad_moves.npy", badMoves)

# Save backup of files
TIME = str(datetime.datetime.now()).replace(" ", "_").replace(":", "-").replace(".", "-", 1)
np.save(f"history/backup/good_moves_v6_{TIME}.npy", goodMoves)
np.save(f"history/backup/bad_moves_v6_{TIME}.npy", badMoves)

# Session Stats
print("Session Stats")
print("Lvl 1:", lvls[0])
print("Lvl 2:", lvls[1])
print("Lvl 3:", lvls[2])
print("Learned", len(goodMoves), "good moves")
print("Learned", len(badMoves), "bad moves")