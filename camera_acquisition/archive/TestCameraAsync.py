# https://medium.com/hackernoon/a-simple-introduction-to-pythons-asyncio-595d9c9ecf8c

import numpy as np
import cv2 as CV
import time as TI
import threading as TH

# import collections as CO
from collections import deque as DEque
# import queue as QU 


import asyncio



class CameraBufferer(object):
    def __init__(self):
        self._grabbing = True
        self._recording = False
        self.fps = 60

        # buf = deque(maxlen = 320)
        self.buffer = DEque()#maxlen = int(buffer_time * fps * 1.05) )

        self.cam = CV.VideoCapture(0)

        self.cam.set(CV.CAP_PROP_FRAME_WIDTH, 1280 )
        self.cam.set(CV.CAP_PROP_FRAME_HEIGHT, 720 )
        # self.cam.set(CV.CAP_PROP_FPS, 60)

        self._ready = False
        self._newgrab = asyncio.Condition()

    async def Grabber(self):
        while self._grabbing:

            # print ("\tgrab...")
            self.cam.grab()
            self._ready = True

            # await self._newgrab.acquire()
            # self._newgrab.notify_all()
            # self._newgrab.release()

            await asyncio.sleep((1./self.fps)*1e-3)


    def Ready(self):
        return self._ready


    async def Retriever(self):

        # print ("sleeping.")
        # await asyncio.sleep(2.)

        self.start = TI.time()
        now = TI.time()
        # await cond.acquire()

        print ('starting!')
        while now-self.start < 10.:
            # Capture frame-by-frame
            now = TI.time()
            # print (now-self.start)
            # grab = self.cam.grab()

            # await self._newgrab.acquire()
            # await self._newgrab.wait_for(self.Ready)
            # self._newgrab.release()
            # # print("Thing was found to be true!")
            # # await 

            # print ("\tread...")
            ret, frame = self.cam.read()
            self._ready = False
            self.buffer.append((now - self.start, ret, frame))

            await asyncio.sleep((1./self.fps)*1e-4)

        self._grabbing = False


    def Save(self):
        # print (len(buf))
        for t, ret, frame in self.buffer:

            # Our operations on the frame come here
            # if len(frame.shape) == 3:
            #     frame = CV.cvtColor(frame, CV.COLOR_BGR2GRAY)

            if ret:
                CV.imwrite('frames/%010.0f.png' % (t*1e3), frame, (CV.IMWRITE_PNG_COMPRESSION, 0)) # compression 9: up to 80% memory saving

    def __enter__(self):
        # required for context management ("with")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # exiting when in context manager
        self.Quit()

    def Quit(self):
        # When everything done, release the capture
        self._grabbing = False
        self._recording = False 

        self.cam.release()
        CV.destroyAllWindows()



with CameraBufferer() as cambu:
    # this is the event loop
    loop = asyncio.get_event_loop()

    # schedule both the coroutines to run on the event loop
    loop.run_until_complete(asyncio.gather(cambu.Grabber(), cambu.Retriever()))

    print ('saving!')
    cambu.Save()
