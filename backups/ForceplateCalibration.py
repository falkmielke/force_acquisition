

################################################################################
### Libraries                                                                ###
################################################################################
import IOToolbox as IOT
# import multiprocessing as MPR
import time as TI
import os as OS
import numpy as NP
import pandas as PD
import re as RE
import threading as TH
from collections import deque as DEQue # double ended queue

import matplotlib as MP # plotting
import matplotlib.pyplot as MPP # plot control



IOT.TriggeredForcePlateDAQ( \
                  fp_type = 'AMTI' \
                , device_nr = 0 \
                , pins = {'led': 7} \
                , sampling_rate = self.sampling_rate \
                , scan_frq = self.scan_frq \
                , recording_duration = self.recording_duration \
                )