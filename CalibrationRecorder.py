import IOToolbox as IOT
# import multiprocessing as MPR
import time as TI
import os as OS
import numpy as NP
import pandas as PD
import re as RE

MakeDataDict = lambda sync, force: {'sync': sync, 'force': force}#, 'sensor': sensor}



def GetBalanceScore(combined_data, perturbation_times):
    main_sync = combined_data['sync'].loc[NP.logical_and( \
                                                    NP.logical_not(combined_data['sync']['reference'].values) \
                                                    , NP.logical_not(combined_data['sync']['baseline'].values) \
                                                    ), :]

    start = main_sync.loc[main_sync['type'].values == 'start_trigger', 'time'].values[0]

    scores = []
    for dt in perturbation_times:
        test_timepoint = start + dt
        score = IOT.BalanceScore( \
                          forcetrace = combined_data['force'] \
                        , test_timepoint = test_timepoint \
                        , interval = [-1., 3.] \
                    )
        scores.append(score)

    print ( 'balance score: \t%.5f' % (NP.mean(scores)) )



class BalanceRecorder():
    def __init__(self, recording_duration, baseline_duration = 0, perturbation_times = [], label = ''):

        self.scan_frq = 1e6
        self.clock_hz = 2.0e6
        self.recording_duration = recording_duration
        self.baseline_duration = baseline_duration
        self.prerecord_baseline = self.baseline_duration > 0
        self.perturbation_times = perturbation_times
        self.label = label

        self.store = []


        # find FT232H

        # try:
        #     self.serial = IOT.FindDevices()[0]
        # except Exception as e:
        #     print ('No breakout found!:\n\t IOT.FindDevices()', e, '\n')

        # initialize force plate
        # TDOT: try:except for constructors to catch problems and verbose
        try:
            self.forceplate = IOT.DAQForcePlate( \
                                  fp_type = 'kistler' \
                                , pins = {'trigger': 5, 'led': 7} \
                                # , baseline_duration = self.baseline_duration \
                                , recording_duration = self.recording_duration \
                                , sampling_rate = 1e3 \
                                , scan_frq = self.scan_frq \
                                )
        except Exception as e:
            print ('error in DAQ setup:\n\t', e, '\n')


        # link the trigger
        self.is_triggered = (self.forceplate.is_triggered or self.forceplate.is_multitriggered)
        self.trigger = self.forceplate.trigger

        # # get sensors
        # try:
        #     self.ft_breakout = IOT.FT232H( \
        #                           serial = self.serial \
        #                         , multiplexer_address = 0x70 \
        #                         , clock_hz = self.clock_hz \
        #                         )
        # except Exception as e:
        #     print ('error in FT232H setup:\n\t', e, '\n')

        # try:
        #     self.sensors = IOT.BufferedMultiNXP( \
        #                           self.ft_breakout \
        #                         , recording_duration = self.recording_duration \
        #                         , multiplexer_channels = [0,1,2] \
        #                         , sampling_rate = self.scan_frq \
        #                         , clock_hz = self.clock_hz \
        #                         )
        # except Exception as e:
        #     print ('error in sensor setup:\n\t', e, '\n')

        # prepare auto save
        self.PrepareAutosave()


    def PrepareAutosave(self):
        # auto file saving
        self.datetag = TI.strftime('%Y%m%d')
        self.file_pattern = "data/{date:s}_{label:s}_cal{nr:03.0f}_{suffix:s}.csv"
        self.suffix = ''

        # check how many previous recordings there were
        previous_recordings = [file for file in OS.listdir('data') if OS.path.splitext(file)[1] == '.csv']
        counts = [0]
        for file in previous_recordings:
            filename = OS.path.splitext(file)[0]
            found = RE.findall(r"(?<=_cal)\d*_", filename) # find file name patterns of the type "*calXXX_*"
            print (filename, found)
            if not (len(found) == 0):
                counts.append(int(found[0][:-1]))

        self.recording_counter = max(counts) + 1



    def SetRecordingDuration(self, recording_duration):
        self.recording_duration = recording_duration

        self.forceplate.recording_duration = recording_duration
        self.forceplate.PrepareAnalogAcquisition()

        # self.sensors.recording_duration = recording_duration



    def Record(self):

        # self.sensors.Start()
        self.forceplate.Record()
        while not self.forceplate.DAQIsIdle():
            TI.sleep(1/self.scan_frq)
        # self.sensors.Stop()



    # def ReferenceRecording(self):

    #     # set recording duration to baseline duration
    #     standard_duration = self.recording_duration
    #     self.SetRecordingDuration(1.)

    #     # Record
    #     self.Record()

    #     # store data
    #     self.store.append(self.RetrieveOutput(baseline = False, reference = True, verbose = False))

    #     # revert recording duration
    #     self.SetRecordingDuration(standard_duration)

    #     return




    # def BaselineRecording(self):

    #     if self.baseline_duration <= 0: 
    #         raise Exception('please set a baseline duration > 0!')
    #         return

    #     # set recording duration to baseline duration
    #     standard_duration = self.recording_duration
    #     self.SetRecordingDuration(self.baseline_duration)

    #     # Record
    #     self.Record()

    #     # store data
    #     self.store.append(self.RetrieveOutput(baseline = True, reference = False, verbose = False))

    #     # revert recording duration
    #     self.SetRecordingDuration(standard_duration)

    #     return



    def TriggeredRecording(self):
        if self.is_triggered:
            triggered_bit = None
            while triggered_bit is None:
                print ('awaiting trigger.')
                # wait for a trigger
                try:
                    triggered_bit = self.trigger.Await(scan_frq = self.scan_frq)
                except KeyboardInterrupt as ki:
                    raise ki
                
                # triggering failed
                if triggered_bit is None:
                    return

                # take a baseline recording
                if triggered_bit == self.forceplate.pins.get('baseline', -1):
                    # print ('baseline recording...')
                    # self.BaselineRecording()
                    # print ('done!')

                    # exit if only baseline is recorded
                    if not self.prerecord_baseline:
                        return

                    # optionally repeat triggering
                    triggered_bit = None

                elif triggered_bit == self.forceplate.pins.get('reference', -1):
                    # print ('reference recording...')
                    # self.ReferenceRecording()
                    # print ('done!')
                    triggered_bit = None

                else:
                    # a recording bit has been hit, take real recording
                    break


            print ('triggered on DAQ channel %i. recording... ' % (triggered_bit))#, end='') 

        TI.sleep(0.1)
        self.Record()
        print ('done!')
        self.store.append(self.RetrieveOutput(baseline = False, reference = False, verbose = True))


        # print ('Awaiting reference recording (empty force plate)...')
        # TI.sleep(0.1)
        # triggered_bit = self.trigger.Await(scan_frq = self.scan_frq)
        # if triggered_bit:
        #     print ('recording...')
        #     self.ReferenceRecording()
        # print ('all done!')


        # store and done!
        self.StoreOutput()



    def RetrieveOutput(self, baseline = None, reference = None, verbose = False):
        force_times, force_data = self.forceplate.RetrieveOutput(verbose = verbose)
        # sensor_data = self.sensors.RetrieveOutput()

        # trigger times
        sync_data = {'time': [], 'type': []}
        for tp, tt in force_times.items():
            sync_data['time'].append(tt)
            sync_data['type'].append( tp + "_trigger" )

        sync_data = PD.DataFrame.from_dict(sync_data)

        # prepare control columns
        for df in [sync_data]: # sensor_data
            df['reference'] = False
            df['baseline'] = False

        # # tag baseline
        # if baseline is not None:
        #     for df in [sync_data, force_data, sensor_data]:
        #         df['baseline'] = baseline
        # if reference is not None:
        #     for df in [sync_data, force_data, sensor_data]:
        #         df['reference'] = reference

        return MakeDataDict(sync = sync_data, force = force_data)#, sensor = sensor_data)



    def StoreOutput(self):
        # data = recorder.RetrieveOutput(baseline = False)
        combined_data = {}
        for data in self.store:
            for source, df in data.items():
                if combined_data.get(source, None) is None:
                    combined_data[source] = df
                else:
                    combined_data[source] = PD.concat([combined_data[source], df], axis = 0, sort=False)


        print ('saved recording nr. %i' % (self.recording_counter))
        MakeFileName = lambda suffix: self.file_pattern.format( \
                                                              date = self.datetag \
                                                            , label = self.label \
                                                            , nr = self.recording_counter \
                                                            , suffix = suffix \
                                                            )

        combined_data['sync'].to_csv(MakeFileName('sync'), sep = ';', index = False)
        combined_data['force'].to_csv(MakeFileName('force'), sep = ';')
        # combined_data['sensor'].to_csv(MakeFileName('sensor'), sep = ';')

        # try:
        #     GetBalanceScore(combined_data, self.perturbation_times)
        # except Exception as e:
        #     print ('no balance score available!:\n\t', e, '\n')
        
        # empty chain
        self.store = []
        self.recording_counter += 1


if __name__ == "__main__":


    recorder = BalanceRecorder( \
                                  recording_duration = 6. \
                                , baseline_duration = 0. \
                                # , perturbation_times = [40., 50., 60., 70.] \
                                , label = 'kistler4' \
                                )
    OS.system('clear')

    while True:
        print ('\n'*3, '_'*32)
        try:
            recorder.TriggeredRecording()
        except KeyboardInterrupt:
            break
