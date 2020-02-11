
import numpy as np
import cv2 as CV
import time as TI

# import collections as CO
from collections import deque 

# buf = deque(maxlen = 320)
buffer_time = 10 # s
fps = 60
buffer_size = int(buffer_time * fps * 1.05)
print (buffer_size)
buf = deque(maxlen = buffer_size )

cap = CV.VideoCapture(0)
# cap.release()
def SetOptimal(cap):
    # approx. 1MB per frame
    cap.set(CV.CAP_PROP_CONVERT_RGB, False)
    cap.set(CV.CAP_PROP_FPS, 60) # only relevant if multiple fps supported per mode
    cap.set(CV.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(CV.CAP_PROP_FRAME_HEIGHT, 720)

def SetMidSpeed(cap):
    # approx. 1MB per frame
    cap.set(CV.CAP_PROP_CONVERT_RGB, False)
    cap.set(CV.CAP_PROP_FPS, fps) # only relevant if multiple fps supported per mode
    cap.set(CV.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(CV.CAP_PROP_FRAME_HEIGHT, 480)

def SetHighSpeed(cap):
    # approx. 250KB per frame
    cap.set(CV.CAP_PROP_CONVERT_RGB, False)
    cap.set(CV.CAP_PROP_FPS, fps) # effectively only 96 fps
    cap.set(CV.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(CV.CAP_PROP_FRAME_HEIGHT, 240)

## official modes oCam:
#  320x240@160
#  640x480@80
# 1280x720@60
# 1280x960@45

# cap.set(CV.CAP_PROP_FRAME_WIDTH, 1280 )
# cap.set(CV.CAP_PROP_FRAME_HEIGHT, 720 )
# cap.set(CV.CAP_PROP_FPS, 30)

SetOptimal(cap)
# SetMidSpeed(cap)
# SetHighSpeed(cap)

# print (cap.get(CV.CAP_PROP_FPS)) # not meaningful

start = TI.time()
now = TI.time()
print ('starting!')
while now-start < 10.:
    # Capture frame-by-frame
    now = TI.time()
    ret, frame = cap.read()
    # print (now - start, now - prev, ret)
    # prev = now
    # CV.imwrite('frames/%010.0f.png' % ((now - start)*1e3), frame, (CV.IMWRITE_PNG_COMPRESSION, 0))

    buf.append((now - start, ret, frame))

    # Display the resulting frame
    # CV.imshow('frame',frame)
    # if CV.waitKey(1) & 0xFF == ord('q'):
    #     break

# When everything done, release the capture
cap.release()
CV.destroyAllWindows()

print ('saving!')
# print (len(buf))
for t, ret, frame in buf:

    # Our operations on the frame come here
    # if len(frame.shape) == 3:
    #     frame = CV.cvtColor(frame, CV.COLOR_BGR2GRAY)

    if ret:
        CV.imwrite('frames/%010.0f.png' % (t*1e3), frame, (CV.IMWRITE_PNG_COMPRESSION, 0)) # compression 9: up to 80% memory saving
