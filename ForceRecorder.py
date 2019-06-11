


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


################################################################################
### Muted Force Plate                                                        ###
################################################################################
class SilentForcePlateDAQ(IOT.TriggeredForcePlateDAQ):
    def StdOut(self, *args, **kwargs):
        pass



################################################################################
### Sensor Wrapper                                                           ###
################################################################################
class Sensor(object):

    def __init__(self, recording_duration = 1., clock_hz = 1.0e6, sr = 10000, trigger_pin = 5):

        self.recording_duration = recording_duration
        self.clock_hz = clock_hz
        self.sr = sr
        self.trigger_pin = trigger_pin

        # find FT232H
        try:
            self.serial = IOT.FindDevices()[0]
        except Exception as e:
            print ('No breakout found!:\n\t IOT.FindDevices()', e, '\n')
        
        # get sensor
        try:
            self.ft_breakout = IOT.FT232H( \
                                  serial = self.serial \
                                , clock_hz = self.clock_hz \
                                )
        except Exception as e:
            print ('error in FT232H setup:\n\t', e, '\n')




        self.device = IOT.NXP(self.ft_breakout, clock_hz = self.clock_hz) # 0,1,2
        self.signal = IOT.DeviceBufferLoader( device = self.device, generating_rate = self.sr ) #, max_length = 512 for post trigger

        self.ft_breakout.setup(self.trigger_pin, IOT.IN)

        self.Empty()


    def Record(self, rising = False):

        status = not rising
        while True:
            new = self.ft_breakout.input(self.trigger_pin)
            if status == new:
                TI.sleep(1/self.sr)
            else:
                break


        self.sync.append([TI.time(), 0])
        self.signal.Start()
        
        TI.sleep(self.recording_duration)
        self.signal.Stop()
        self.sync.append([TI.time(), -1])
        


    def RetrieveOutput(self):
        time, data = self.signal.RetrieveOutput()
        data = PD.DataFrame(data, index = time, columns = self.device.config['columns'])

        time_out = PD.DataFrame(NP.stack(self.sync, axis = 0), columns = ['time', 'current_scan_count'])
        return time_out, data



    def PrepareAnalogAcquisition(self):
        pass

    def Empty(self):
        # remove previous data
        self.sync = []
        self.data = None

    def Quit(self):
        self.ft_breakout.close()



################################################################################
### Force Recorder                                                           ###
################################################################################
class ForceRecorder(object):
    def __init__(self, recording_duration, label = '', sampling_rate = 1e3, scan_frq = 1e6, clock_hz = 1.0e6, viewer = None):

        self.sampling_rate = sampling_rate
        self.scan_frq = scan_frq
        self.recording_duration = recording_duration
        self.label = label
        
        self.viewer = viewer


        self._threads = None


        # initialize first DAQ
        self.daqs = {}
        try:
            daq1 = SilentForcePlateDAQ( \
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
            daq2 = SilentForcePlateDAQ( \
                                  fp_type = 'dualkistler2' \
                                , device_nr = 1 \
                                , pins = {'led': 7} \
                                , sampling_rate = self.sampling_rate \
                                , scan_frq = self.scan_frq \
                                , recording_duration = self.recording_duration \
                                )
            self.daqs[daq2.label] = daq2
        except Exception as e:
            print ('error in DAQ2 setup:\n\t', e, '\n')

        # initialize NXP sensor
        try:
            self.daqs['nxp'] = Sensor(recording_duration = self.recording_duration, clock_hz = clock_hz)

        except Exception as e:
            print ('error in sensor setup:\n\t', e, '\n')


        # prepare auto save
        self.PrepareAutosave()

        ### initialize viewer
        self.device_labels = self.daqs.keys()
        self.q = DEQue()

        self.PreparePlot()




    # def Exit(self, event):
    #     if event.key in ['q', 'e', 'escape', '<space>']:
    #         self.playing = False


    def RestoreAxes(self):
        for daq, ax in self.ax_dict.items():
            ax.clear()
            ax.set_xlim(0., self.recording_duration)

            ax.set_ylim(0., len(all_data_columns[daq])+1.)
            ax.set_xlabel('time (s)')

            ax.set_yticks(NP.arange(len(all_data_columns[daq]))+1)
            ax.set_yticklabels(all_data_columns[daq][::-1])

            for split in plot_splits[daq]:
                ax.axhline(split, color = 'k')


    def PreparePlot(self):

        self.fig, axes = MPP.subplots(1, len(self.device_labels))

        self.fig.subplots_adjust( \
                              top    = 0.99 \
                            , right  = 0.99 \
                            , bottom = 0.06 \
                            , left   = 0.06 \
                            , wspace = 0.10 # column spacing \
                            , hspace = 0.10 # row spacing \
                            )

        self.ax_dict = {devlab: axes[nr] for nr, devlab in enumerate(self.device_labels)}

        self.RestoreAxes()
        # self.ax.yaxis.tick_right()
        # self.ax.yaxis.set_label_position("right")
        # self.ax.set_title('press "E" to exit.')

        # draw all empty
        MPP.show(False)
        MPP.draw()
        self.fig.canvas.draw()

        # self.fig.canvas.mpl_connect('key_press_event', self.Exit)

        # cache the background
        self.plot_backgrounds = { key: self.fig.canvas.copy_from_bbox(ax.bbox) \
                                    for key, ax in self.ax_dict.items() \
                                }

        # prepare lines
        self.handles = {}
        for devlab in self.device_labels:
            for col in all_data_columns[devlab]:
                self.handles[col] = self.ax_dict[devlab].plot([0], [0], linestyle = '-')[0]



    def UpdatePlot(self):

            rec_nr, daq, data = self.q.popleft()

            # restore background
            # self.fig.canvas.restore_region(self.plot_backgrounds[daq])
            self.RestoreAxes()

            # self.fig.suptitle("recording %i" % (rec_nr))

            # adjust and redraw data
            y_ticks = []
            for col in data.columns:


                t = NP.linspace(0, self.recording_duration, data.shape[0], endpoint = False)
                y = NP.array(data[col].values, dtype = float)
                
                # normalize
                y -= y[0]
                y /= (NP.max(y)-NP.min(y)+1e-3)/2

                # shift
                offset = len(all_data_columns[daq])-all_data_columns[daq].index(col)
                y += offset

                self.handles[col].set_data(t, y) # plot one less to avoid flickering
                self.ax_dict[daq].draw_artist(self.handles[col])




            self.ax_dict[daq].set_title("recording %i" % (rec_nr))
            self.fig.canvas.blit(self.ax_dict[daq].bbox)




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
            # print (filename, found)
            if not (len(found) == 0):
                counts.append(int(found[0][:-1]))

        self.recording_counter = max(counts) + 1



    def SetRecordingDuration(self, new_duration):
        self.recording_duration = new_duration

        for daq in self.daqs.values():
            daq.recording_duration = new_duration
            daq.PrepareAnalogAcquisition()



    def LaunchPlates(self, daq):
        # will start up all the input devices
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
            sync.to_csv(MakeFileName(daq = daq, suffix = 'sync'), sep = ';', index = False)
            data.to_csv(MakeFileName(daq = daq, suffix = 'force'), sep = ';')

            # send to viewer
            self.q.append([self.recording_counter, daq, data])

            device.Empty()

        
        print('done recording %i! ' % (self.recording_counter), ' '*32)
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


            # TI.sleep(1)
            MPP.pause(1.e0)
            while (len(self.q) > 0):
                self.UpdatePlot()






################################################################################
### Data Viewer                                                              ###
################################################################################
all_data_columns = { \
                      'nxp': ['a_x', 'a_y', 'a_z', 'g_x', 'g_y', 'g_z', 'm_x', 'm_y', 'm_z'] \
                    , 'blue': list(sorted(IOT.forceplate_settings['dualkistler']['channel_order'])) \
                    , 'green': list(sorted(IOT.forceplate_settings['dualkistler2']['channel_order'])) \
                    }
plot_splits = { \
                      'nxp': [3.5, 6.5] \
                    , 'blue': [8.5] \
                    , 'green': [8.5] \
                    }



################################################################################
### Mission Control                                                          ###
################################################################################
if __name__ == "__main__":


    recording_duration = 10 # s
    fr = ForceRecorder(   recording_duration = recording_duration \
                        , label = 'goa' \
                        , sampling_rate = 1e3 \
                        , scan_frq = 1e6 \
                        , clock_hz = 1.8e6 \
                        # , viewer = DataViewer(device_labels = ['nxp', 'blue', 'green']) \
                        )

    fr.Loop()
