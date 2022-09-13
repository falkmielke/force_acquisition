


################################################################################
### Libraries                                                                ###
################################################################################
import os as OS
# OS.system("cd ~/acquisition_software")
import IOToolbox as IOT
# import multiprocessing as MPR
import time as TI
import numpy as NP
import pandas as PD
import re as RE
import threading as TH
from collections import deque as DEQue # double ended queue

import matplotlib as MP # plotting
import matplotlib.pyplot as MPP # plot control
MP.use('TkAgg')

# check if there is a data folder
if not OS.path.exists('data'):
    OS.makedirs('data')

################################################################################
### Muted Force Plate                                                        ###
################################################################################
class SilentForcePlateDAQ(IOT.TriggeredForcePlateDAQ):
    def StdOut(self, *args, **kwargs):
        pass




################################################################################
### Force Recorder                                                           ###
################################################################################
class CalibrationRecorder(object):
    def __init__(self, recording_duration, label = '', sampling_rate = 1e3):

        self.sampling_rate = sampling_rate
        self.recording_duration = recording_duration
        self.label = label
        

        self._threads = None


        # initialize first DAQ
        self.daqs = {}
        try:
            daq1 = SilentForcePlateDAQ( \
                                  fp_type = 'kistler' \
                                , device_nr = 0 \
                                , pins = {'led': 7} \
                                , sampling_rate = self.sampling_rate \
                                , recording_duration = self.recording_duration \
                                )
            self.daqs[daq1.label] = daq1
        except Exception as e:
            print ('error in DAQ1 setup:\n\t', e, '\n')



        # prepare auto save
        self.PrepareAutosave()

        ### initialize viewer
        self.device_labels = self.daqs.keys()
        self.q = DEQue()





    def PrepareAutosave(self):
        # auto file saving
        self.datetag = TI.strftime('%Y%m%d')
        self.file_pattern = "data/{date:s}_{label:s}_calibration_rec{nr:03.0f}{suffix:s}.csv"
        self.suffix = ''

        # check how many previous recordings there were
        previous_recordings = list(sorted([file for file in OS.listdir('data') if OS.path.splitext(file)[1] == '.csv']))
        counts = [0]
        #print (len(previous_recordings))
        for file in previous_recordings:
            filename = OS.path.splitext(file)[0]
            found = RE.findall(r"(?<=_rec)\d*", filename) # find file name patterns of the type "*daqXXX_*"
            # print (filename, found)
            if not (len(found) == 0):
                counts.append(int(found[0]))#[:-1]

        self.recording_counter = max(counts) + 1
        #print (self.recording_counter)
        print (f'continuing on recording {self.recording_counter}')




    def SetRecordingDuration(self, new_duration):
        self.recording_duration = new_duration

        for daq in self.daqs.values():
            daq.recording_duration = new_duration
            daq.PrepareAnalogAcquisition()



    def LaunchPlates(self, daq):
        # will start up an input device
        daq.Record()


    def Record(self):
        # will begin recording, waiting for trigger
        print ('\n', '_'*32)

        self._threads = [TH.Thread(target = self.LaunchPlates, args = [daq]) for daq in self.daqs.values()]

        print ('waiting for trigger to record...', end = '\r')
        for trd in self._threads:
            trd.start()

        TI.sleep(0.1)
        for trd in self._threads:
            trd.join()
        self._threads = None
        
        print ('saving...', ' '*32 , end = '\r')
        self.StoreOutput()

        TI.sleep(1.)
        # print (self.q)



    def StoreOutput(self):

        MakeFileName = lambda daq, suffix: self.file_pattern.format( \
                                                              date = self.datetag \
                                                            , label = self.label \
                                                            , daq = daq \
                                                            , nr = self.recording_counter \
                                                            , suffix = suffix \
                                                            )

        for daq, device in self.daqs.items():
            sync, data = device.RetrieveOutput()
            # sync.to_csv(MakeFileName(daq = daq, suffix = 'sync'), sep = ';', index = False)
            data.to_csv(MakeFileName(daq = daq, suffix = ''), sep = ';')

            # send to viewer
            self.q.append([self.recording_counter, daq, data])

            device.Empty()

        
        print('done recording %03.0f! ' % (self.recording_counter), ' '*32)
        self.recording_counter += 1

    def Stop(self):

        for daq, device in self.daqs.items():
            device.analog_input.scan_stop()
            device.Quit()
                
        self.playing = False
        self.fig.close()



    def Loop(self):
        self.playing = True
        while self.playing:

            try:
                self.Record()
            except Exception as e:
                raise e
                self.Stop()            
                break
            except KeyboardInterrupt as ki:
                self.Stop()            
                break


        print('\n') # end last line






################################################################################
### Mission Control                                                          ###
################################################################################
if __name__ == "__main__":


    recording_duration = 6. # s
    fr = CalibrationRecorder( recording_duration = recording_duration \
                            , label = 'kistler' \
                            , sampling_rate = 10.0e3
                            )

    fr.Loop()
