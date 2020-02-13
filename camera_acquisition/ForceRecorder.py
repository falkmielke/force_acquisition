


################################################################################
### Libraries                                                                ###
################################################################################
# import multiprocessing as MPR
import time as TI
import os as OS
import numpy as NP
import pandas as PD
import re as RE
import threading as TH
import atexit as EXIT # commands to shut down processes
from collections import deque as DEQue # double ended queue

import matplotlib as MP # plotting
import matplotlib.pyplot as MPP # plot control

import DAQToolbox as IOT


################################################################################
### Muted output                                                        ###
################################################################################

def Silent(self, *args, **kwargs):
        pass

################################################################################
### Force Recorder                                                           ###
################################################################################
class ForceRecorder(object):
    def __init__(self, recording_duration, label = '', fp_type = 'joystick', post_trigger = False, sampling_rate = 1e3, scan_frq = 1e6, clock_hz = 1.0e6, viewer = None):

        self.post_trigger = post_trigger
        self.sampling_rate = sampling_rate
        self.scan_frq = scan_frq
        self.recording_duration = recording_duration
        self.label = label
        
        self.viewer = viewer


        self._threads = None


        EXIT.register(self.Quit)


        # initialize first DAQ
        self.daqs = {}
        try:
            if self.post_trigger:
                daq1 = IOT.PostTriggerDAQ( \
                                  fp_type = fp_type \
                                , device_nr = 0 \
                                , pins = {'led': 7, 'triggers': [5]} \
                                , rising = False \
                                , sampling_rate = self.sampling_rate \
                                , scan_frq = self.scan_frq \
                                , recording_duration = self.recording_duration \
                                )
            else:
                # uses the trigger port on the MCC DAQ for highest accuracy
                daq1 = IOT.TriggeredForcePlateDAQ( \
                                  fp_type = fp_type \
                                , device_nr = 0 \
                                , pins = {'led': 7} \
                                , sampling_rate = self.sampling_rate \
                                , scan_frq = self.scan_frq \
                                , recording_duration = self.recording_duration \
                                )


            # mute by overwriting StdOut
            daq1.StdOut = Silent

            self.daqs[daq1.label] = daq1
        except Exception as e:
            print ('error in DAQ setup:\n\t', e, '\n')

        # initialize camera
        try:
            cam1 = IOT.Camera(recording_duration = self.recording_duration, cam_nr = 0, label = 'cam1', daq = daq1)
            cam1.StdOut = Silent
            self.daqs['cam'] = cam1

        except Exception as e:
            print ('error in camera setup:\n\t', e, '\n')


        # prepare auto save
        self.PrepareAutosave()

        ### initialize viewer
        self.device_labels = [key for key in self.daqs.keys() if key not in ['cam']]
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

        if len(self.device_labels) == 1:
            self.ax_dict = {self.device_labels[0]: axes}
        else:
            self.ax_dict = {devlab: axes[nr] for nr, devlab in enumerate(self.device_labels)}

        self.RestoreAxes()
        # self.ax.yaxis.tick_right()
        # self.ax.yaxis.set_label_position("right")
        # self.ax.set_title('press "E" to exit.')

        # draw all empty
        # MPP.show()
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
        self.file_pattern = "recordings/{date:s}_{label:s}_{daq:s}_rec{nr:03.0f}_{suffix:s}{extension:s}"
        self.suffix = ''

        # check how many previous recordings there were
        previous_recordings = [file for file in OS.listdir('recordings') if OS.path.splitext(file)[1] == '.csv']
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
            daq.PrepareAcquisition()



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
                                                            , extension = '' if daq == 'cam' else '.csv' \
                                                            )

        for daq, device in self.daqs.items():
            sync, data = device.RetrieveOutput()

            if daq == 'cam':
                NP.savez(MakeFileName(daq = daq, suffix = device.label), time = sync, images = data)
                # NP.savez_compressed(MakeFileName(daq = daq, suffix = 'video'), time = sync, images = data)
            else:
                sync.to_csv(MakeFileName(daq = daq, suffix = 'sync'), sep = ';', index = False)
                data.to_csv(MakeFileName(daq = daq, suffix = 'force'), sep = ';')

                # send to viewer
                self.q.append([self.recording_counter, daq, data])

            device.Empty()

        
        print('done recording %i! ' % (self.recording_counter), ' '*32)
        self.recording_counter += 1

        # restart camera
        self.daqs['cam'].RestartCamera()


    def Stop(self):

        for daq, device in self.daqs.items():
            device.AbortRecording()
                
        self.playing = False
        MPP.close()



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


#______________________________________________________________________
# Destructor
#______________________________________________________________________
    def __enter__(self):
        # required for context management ("with")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # exiting when in context manager
        self.Quit()

    def Quit(self):
        print()
        for daq in self.daqs.values():
            try:
                daq.Quit()
            except Exception as e:
                print (e) 




################################################################################
### Data Viewer                                                              ###
################################################################################
# all_data_columns = { \
#                       'nxp': ['a_x', 'a_y', 'a_z', 'g_x', 'g_y', 'g_z', 'm_x', 'm_y', 'm_z'] \
#                     , 'blue': list(sorted(IOT.forceplate_settings['dualkistler']['channel_order'])) \
#                     , 'green': list(sorted(IOT.forceplate_settings['dualkistler2']['channel_order'])) \
#                     }
fp_type = 'joystick'
all_data_columns = { \
                      'green': list(sorted(IOT.forceplate_settings[fp_type]['channel_order'])) \
                      , 'cam': [] \
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


    recording_duration = 10. # s
    with ForceRecorder(   recording_duration = recording_duration \
                        , label = 'rat' \
                        , fp_type = fp_type \
                        , post_trigger = False \
                        , sampling_rate = 1e3 \
                        , scan_frq = 1e6 \
                        ) as forcerec:

        forcerec.Loop()
