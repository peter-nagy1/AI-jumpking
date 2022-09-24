import numpy as np
import cv2
from mss import mss
from PIL import Image

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

        cv2.imshow ('frame', img)       # Show
        
        if cv2.waitKey ( 1 ) & 0xff == ord( 'q' ) :
            cv2.destroyAllWindows()


getImg()
