import IOToolbox as IOT
# import multiprocessing as MPR
import time as TI
import os as OS
import numpy as NP
import pandas as PD
import re as RE




class ForceRecorder(object):
    def __init__(self, recording_duration, label = '', sampling_rate = 1e3, scan_frq = 1e6, clock_hz = 2.0e6):

        self.sampling_rate = sampling_rate
        self.scan_frq = scan_frq
        self.clock_hz = clock_hz
        self.recording_duration = recording_duration
        self.label = label


        # # find FT232H
        # try:
        #     self.serial = IOT.FindDevices()[0]
        # except Exception as e:
        #     print ('No breakout found!:\n\t IOT.FindDevices()', e, '\n')

        # initialize first DAQ
        self.daqs = {}
        try:
            daq1 = TriggeredForcePlateDAQ( \
                                  fp_type = 'dualkistler' \
                                , device_nr = 0 \
                                , pins = {'led': 7} \
                                , sampling_rate = self.sampling_rate \
                                , scan_frq = self.scan_frq \
                                , recording_duration = self.recording_duration \
                                )
            self.daqs[daq1.label] = daq1
        except Exception as e:
            print ('error in DAQ1 setup:\n\t', e, '\n')


        # initialize second DAQ
        try:
            daq2 = TriggeredForcePlateDAQ( \
                                  fp_type = 'dualkistler2' \
                                , device_nr = 0 \
                                , pins = {'led': 7} \
                                , sampling_rate = self.sampling_rate \
                                , scan_frq = self.scan_frq \
                                , recording_duration = self.recording_duration \
                                )
            self.daqs[daq2.label] = daq2
        except Exception as e:
            print ('error in DAQ2 setup:\n\t', e, '\n')


        # prepare auto save
        self.PrepareAutosave()


    def PrepareAutosave(self):
        # auto file saving
        self.datetag = TI.strftime('%Y%m%d')
        self.file_pattern = "data/{date:s}_{label:s}_{daq:s}_rec{nr:03.0f}_{suffix:s}.csv"
        self.suffix = ''

        # check how many previous recordings there were
        previous_recordings = [file for file in OS.listdir('data') if OS.path.splitext(file)[1] == '.csv']
        counts = [0]
        for file in previous_recordings:
            filename = OS.path.splitext(file)[0]
            found = RE.findall(r"(?<=_rec)\d*_", filename) # find file name patterns of the type "*daqXXX_*"
            print (filename, found)
            if not (len(found) == 0):
                counts.append(int(found[0][:-1]))

        self.recording_counter = max(counts) + 1



    def SetRecordingDuration(self, recording_duration):
        self.recording_duration = recording_duration

        for daq in self.daqs.values():
            daq.recording_duration = recording_duration
            daq.PrepareAnalogAcquisition()

        # self.sensors.recording_duration = recording_duration



    def LaunchPlates(self, daq):
        daq.Record()


    def Record(self):
        self._threads = [TH.Thread(target = self.LaunchPlates, args = [daq]) for daq in self.daqs.values()]
        for trd in self._threads:
            trd.start()

        TI.sleep(0.1)
        for trd in self._threads:
            trd.join()

        self.StoreOutput()


    def StoreOutput(self):

        print ('saved recording nr. %i' % (self.recording_counter))
        MakeFileName = lambda daq, suffix: self.file_pattern.format( \
                                                              date = self.datetag \
                                                            , label = self.label \
                                                            , daq = daq \
                                                            , nr = self.recording_counter \
                                                            , suffix = suffix \
                                                            )

        for daq, device in self.daqs.items():
            sync, data = device.RetrieveOutput()
            sync.to_csv(MakeFileName(daq = daq, suffix = 'sync'), sep = ';', index = False)
            data.to_csv(MakeFileName(daq = daq, suffix = 'force'), sep = ';')

        
        self.recording_counter += 1




if __name__ == "__main__":
    recording_duration = 6 # s
    fr = ForceRecorder(   recording_duration = recording_duration \
                        , label = 'goa' \
                        , sampling_rate = 1e3 \
                        , scan_frq = 1e6 \
                        # , clock_hz = 2.0e6 \
                        )

