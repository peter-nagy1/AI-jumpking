import pyautogui as pag
import time
import random

def leftMove(sz):
    pag.keyDown('left')
    time.sleep(sz)
    pag.keyUp('left')

def rightMove(sz):
    pag.keyDown('right')
    time.sleep(sz)
    pag.keyUp('right')

def leftJump(sz):
    pag.keyDown('space')
    pag.keyDown('left')
    time.sleep(sz)
    pag.keyUp('space')
    pag.keyUp('left')

def rightJump(sz):
    pag.keyDown('space')
    pag.keyDown('right')
    time.sleep(sz)
    pag.keyUp('space')
    pag.keyUp('right')

# Not in use
def getRandMove():
    return random.randint(0,3)

# Not in use
def getRandSize():
    num = random.randint(1,10) / 10
    if num == 0.1:
        num = random.randint(1,10) / 100
    return num

def createOptions():
    move = list(range(4))
    sz = list(map(lambda x: x/100, list(range(1, 10)))) + list(map(lambda x: x/10, list(range(1, 11))))
    return [(a,b) for a in move for b in sz]

def pickMove(num, sz):
    if num == 0:
        leftMove(sz)
    elif num == 1:
        rightMove(sz)
    elif num == 2:
        leftJump(sz)
    elif num == 3:
        rightJump(sz)