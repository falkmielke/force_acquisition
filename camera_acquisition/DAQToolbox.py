
# Universiteit Antwerpen
# Functional Morphology
# Falk Mielke
# 2019/06/06


################################################################################
### Libraries                                                                ###
################################################################################
import os as OS
import sys as SYS
import select as SEL # for timed user input
import re as RE # regular expressions
import time as TI # time, for process pause and date
import atexit as EXIT # commands to shut down processes
import subprocess as SP
import threading as TH # threading for trigger
import queue as QU # data buffer
from collections import deque as Deque # double ended queue
import numpy as NP # numerics
import pandas as PD # data storage
import scipy.signal as SIG
import math as MATH
# import matplotlib as MP # plotting
# MP.use('TkAgg')
# import matplotlib.pyplot as MPP # plot control


import uldaq as UL # MCC DAQ negotiation
import cv2 as CV

import logging
logger = logging.getLogger(__name__)

### MCC DAQ drivers and library
# follow readme in https://github.com/mccdaq/uldaq
    # download   $ wget https://github.com/mccdaq/uldaq/releases/download/v1.1.2/libuldaq-1.1.2.tar.bz2
    # extract    $ tar -xvjf libuldaq-1.1.2.tar.bz2 && cd libuldaq-1.1.2
    # build      $ ./configure && make -j4 && sudo make install -j4
    # if "make" fails, you might need to do: ln -s /usr/bin/autom-1.16 /usr/bin/aclocal-1.14 && ln -s /usr/bin/automake-1.16 /usr/bin/automake-1.14
# pip install uldaq

################################################################################
### Global Specifications                                                    ###
################################################################################
coordinates = ['x', 'y', 'z']



################################################################################
### Plotting                                                                 ###
################################################################################
the_font = {  \
        # It's really sans-serif, but using it doesn't override \sffamily, so we tell Matplotlib
        # to use the "serif" font because then Matplotlib won't ask for any special families.
         # 'family': 'serif' \
        # , 'serif': 'Iwona' \
        'family': 'sans-serif'
        , 'sans-serif': 'DejaVu Sans'
        , 'size': 10#*1.27 \
    }


def PreparePlot():
    # select some default rc parameters
    # MP.rcParams['text.usetex'] = True
    MPP.rc('font',**the_font)
    # Tell Matplotlib how to ask TeX for this font.
    # MP.texmanager.TexManager.font_info['iwona'] = ('iwona', r'\usepackage[light,math]{iwona}')

    MP.rcParams['text.latex.preamble'] = [\
                  r'\usepackage{upgreek}'
                , r'\usepackage{cmbright}'
                , r'\usepackage{sansmath}'
                ]

    MP.rcParams['pdf.fonttype'] = 42 # will make output TrueType (whatever that means)



def PolishAx(ax):
# axis cosmetics
    ax.get_xaxis().set_tick_params(which='both', direction='out')
    ax.get_yaxis().set_tick_params(which='both', direction='out')
    ax.tick_params(top = False)
    ax.tick_params(right = False)
    # ax.tick_params(left=False)

    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['right'].set_visible(False)



################################################################################
### MCC USB1608G DAQ                                                         ###
################################################################################

instrument_labels = { '01DF5B18': 'blue' \
                    , '01DF5AFB': 'green'
                    }

status_dict = { \
                UL.ScanStatus.IDLE: 'idle' \
                , UL.ScanStatus.RUNNING: 'running' \
              }

class MCCDAQ(object):
    # generic device functions

    # for output pin values
    LOW = 0
    HIGH = 255    

    # analog input recording mode
    recording_mode = UL.ScanOption.BLOCKIO
    # recording_mode = UL.ScanOption.CONTINUOUS

    alive = False

#______________________________________________________________________
# constructor
#______________________________________________________________________

    def __init__(self, device_nr = 0):

        self.daq_device = None
        self.times = {} # storage of timing info

        # for mcc daq configuration
        self.range_index = 0

        self.AssembleAndConnect(descriptor_index = device_nr)

        self.label = str(self)

        EXIT.register(self.Quit)


    def AssembleAndConnect(self, descriptor_index = 0):
        # connect to the DAQ device
        try:
            
            interface_type = UL.InterfaceType.USB
            # Get descriptors for all of the available DAQ devices.
            devices = UL.get_daq_device_inventory(interface_type)
            number_of_devices = len(devices)
            if number_of_devices == 0:
                raise Exception('Error: No DAQ devices found')

            # print('Found', number_of_devices, 'DAQ device(s):')
            # for i in range(number_of_devices):
            #     print('  ', devices[i].product_name, ' (', devices[i].unique_id, ')', sep='')

            # Create the DAQ device object associated with the specified descriptor index.
            self.daq_device = UL.DaqDevice(devices[descriptor_index])

        ### digital input
            port_types_index = 0

            # Get the DioDevice object and verify that it is valid.
            self.digital_io = self.daq_device.get_dio_device()
            if self.digital_io is None:
                raise Exception('Error: The DAQ device does not support digital input')


            # Get the port types for the device(AUXPORT, FIRSTPORTA, ...)
            dio_info = self.digital_io.get_info()
            port_types = dio_info.get_port_types()

            if port_types_index >= len(port_types):
                port_types_index = len(port_types) - 1

            self.port = port_types[port_types_index]

        ### analog input
            # Get the AiDevice object and verify that it is valid.
            self.analog_input = self.daq_device.get_ai_device()
            if self.analog_input is None:
                raise Exception('Error: The DAQ device does not support analog input')

            # Verify that the specified device supports hardware pacing for analog input.
            self.ai_info = self.analog_input.get_info()
            if not self.ai_info.has_pacer():
                raise Exception('\nError: The specified DAQ device does not support hardware paced analog input')


        ### connect
            # Establish a connection to the DAQ device.
            # print (dir(descriptor))
            self.daq_device.connect()



        except Exception as e:
            print('constructor fail\n', e)

        print('\nConnected to', str(self), '.')


# for digital I/O
    def SetPins(self):
        # implemented by sub class
        raise TypeError('Digital pin setup not implemented!')
        pass


#______________________________________________________________________
# DAQ status
#______________________________________________________________________
    def GetDAQStatus(self):
        return self.analog_input.get_scan_status()

    def DAQIsRunning(self):
        return self.analog_input.get_scan_status()[0] is UL.ScanStatus.RUNNING

    def DAQIsIdle(self):
        return self.analog_input.get_scan_status()[0] is UL.ScanStatus.IDLE




#______________________________________________________________________
# recording preparation
#______________________________________________________________________
    def CountChannels(self, n_channels, channel_labels):
        ### channel settings
        if (n_channels is None) and (channel_labels is None):
            raise Exception('Error: please provide either channel_labels or n_channels.')

        if channel_labels is None:
            # label by number of channels per default
            self.channel_labels = [ 'a_%i' % (nr) for nr in range(n_channels) ]
        else:
            # count channels from the labels
            self.channel_labels = channel_labels
            n_channels = len(channel_labels)


        self.low_channel = 0 # record from channel...
        self.high_channel = n_channels-1 # ... to (incl) channel

        self.n_channels = n_channels



    def PrepareAcquisition(self):
        # generate settings for a recording
        # note that a single prep is usually enough for successive recordings of the same kind

        try:
            # recording_duration = 3 # s
            samples_per_channel = int(self.recording_duration*self.sampling_rate)

            # The default input mode is SINGLE_ENDED.
            input_mode = UL.AiInputMode.SINGLE_ENDED
            # If SINGLE_ENDED input mode is not supported, set to DIFFERENTIAL.
            if self.ai_info.get_num_chans_by_mode(UL.AiInputMode.SINGLE_ENDED) <= 0:
                input_mode = UL.AiInputMode.DIFFERENTIAL

            # print (dir(UL.AInScanFlag))
            flags = UL.AInScanFlag.DEFAULT

            # Get the number of channels and validate the high channel number.
            number_of_channels = self.ai_info.get_num_chans_by_mode(input_mode)
            if self.high_channel >= number_of_channels:
                self.high_channel = number_of_channels - 1
            channel_count = self.high_channel - self.low_channel + 1
            # self.StdOut (input_mode, channel_count)

            # Get a list of supported ranges and validate the range index.
            ranges = self.ai_info.get_ranges(input_mode)
            if self.range_index >= len(ranges):
                self.range_index = len(ranges) - 1


            trigger_types = self.ai_info.get_trigger_types()
            # [<TriggerType.POS_EDGE: 1>, <TriggerType.NEG_EDGE: 2>, <TriggerType.HIGH: 4>, <TriggerType.LOW: 8>]
            self.analog_input.set_trigger(trigger_types[1], 0, 0, 0, 0)


            # Allocate a buffer to receive the data.
            self.buffer = UL.create_float_buffer(channel_count, samples_per_channel)

            recording_mode = self.recording_mode

            # store settings (keywords for self.analog_input.a_in_scan)
            self.recording_settings = dict( \
                                  low_channel = self.low_channel \
                                , high_channel = self.high_channel \
                                , input_mode = input_mode \
                                , analog_range = ranges[self.range_index] \
                                , samples_per_channel = samples_per_channel \
                                , rate = self.sampling_rate \
                                , options = recording_mode \
                                , flags = flags \
                                , data = self.buffer \
                                )

        except Exception as e:
            print('\n', e)



#______________________________________________________________________
# I/O
#______________________________________________________________________
    def NOOP(self):
        # defined by subclasses
        pass

    def __str__(self):
        descriptor = self.daq_device.get_descriptor()
        return descriptor.dev_string + ' "' + instrument_labels[descriptor.unique_id] + '"'

#______________________________________________________________________
# Destructor
#______________________________________________________________________
    def __enter__(self):
        # required for context management ("with")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # exiting when in context manager
        if not self.alive:
            return

        self.Quit()


    def Quit(self):
        # safely exit
        if not self.alive:
            SYS.exit()
            return

        if self.daq_device:
            # Stop the acquisition if it is still running.
            if not self.DAQIsIdle():
                self.analog_input.scan_stop()
            if self.daq_device.is_connected():
                self.daq_device.disconnect()
            self.daq_device.release()
        print('safely exited %s.' % (self.label))
        self.alive = False

        # SYS.exit()


################################################################################
### MCC DAQ single digital input                                             ###
################################################################################
class DAQ_DigitalInput(MCCDAQ):
    # generic device functions

#______________________________________________________________________
# Construction
#______________________________________________________________________

    def __init__(self, digital_pin, scan_frq, *args, **kwargs):

        # variable preparation
        self.digital_pin = digital_pin
        self.scan_frq = scan_frq
        super(DAQ_DigitalInput, self).__init__(*args, **kwargs)

        self.SetPins()

        self.alive = True


    def SetPins(self):
        # set pin to input
        self.digital_io.d_config_bit(self.port, self.digital_pin, UL.DigitalDirection.INPUT)



#______________________________________________________________________
# I/O
#______________________________________________________________________

    def Read(self):
        # read out the trigger bit
        return self.digital_io.d_bit_in(self.port, self.digital_pin)


    def Record(self):

        previous = self.Read()

        # loop until bit changes
        t0 = TI.time()
        while True:
            TI.sleep(1/self.scan_frq)

            # check trigger
            current = self.Read()
            if current != previous:
                print (TI.time(), current)
            previous = current

            if TI.time() > (t0 + 10):
                break



#______________________________________________________________________
# Test
#______________________________________________________________________
def TestMCCPinIn(pin_nr = None):
    if pin_nr is None:
        return

    daq = DAQ_DigitalInput(digital_pin = pin_nr, scan_frq = 1e6)
    daq.Record()




################################################################################
### MCC DAQ single digital output                                            ###
################################################################################
class DAQ_DigitalOutput(MCCDAQ):
    # generic device functions

#______________________________________________________________________
# Construction
#______________________________________________________________________
    def __init__(self, digital_pin, *args, **kwargs):

        # variable preparation
        self.digital_pin = digital_pin
        super(DAQ_DigitalOutput, self).__init__(*args, **kwargs)

        self.SetPins()

        self.alive = True


    def SetPins(self):
        # set pin to output
        self.digital_io.d_config_bit(self.port, self.digital_pin, UL.DigitalDirection.OUTPUT)



#______________________________________________________________________
# I/O
#______________________________________________________________________

    def LED(self, value_bool):
        self.digital_io.d_bit_out(self.port, self.digital_pin, self.HIGH if value_bool else self.LOW)

    def Flash(self, times = 1, duration = 1):
        self.LED(False)
        for _ in range(times):
            self.LED(True)
            TI.sleep(duration)
            self.LED(False)
            TI.sleep(duration)




#______________________________________________________________________
# test
#______________________________________________________________________
def TestMCCPinOut(pin_nr = None):
    if pin_nr is None:
        return

    daq = DAQ_DigitalOutput(digital_pin = pin_nr)
    daq.Flash(times = 3, duration = 1)



################################################################################
### MCC DAQ Analog Recording                                                 ###
################################################################################
class AnalogInput(MCCDAQ):
    # generic device functions

#______________________________________________________________________
# Construction
#______________________________________________________________________

    def __init__(self, sampling_rate = 1e1, recording_duration = 1. \
                , n_channels = None, channel_labels = None \
                , scan_frq = 1e3 \
                , *args, **kwargs \
                ):

        self.sampling_rate = sampling_rate
        self.recording_duration = recording_duration
        self.scan_frq = scan_frq
        self._recording = False

        # setup for analog input
        self.CountChannels(n_channels, channel_labels)
        super(AnalogInput, self).__init__(*args, **kwargs)
        self.PrepareAcquisition()

        self.alive = True


#______________________________________________________________________
# I/O
#______________________________________________________________________

    def Record(self, wait_time = None):
        self.times['start'] = TI.time()
        self._recording = True
        self.rate = self.analog_input.a_in_scan(**self.recording_settings)

        if wait_time is None:
            self.Wait()
        else:
            TI.sleep( wait_time )
        self._recording = False


    def Wait(self, verbose = True):
        # sleeps until recording is finished.

        # counter = 0
        while True:
            TI.sleep(1/self.scan_frq)
            # if (not self.trigger.waiting):
            #     self.Stop()

            # # for precise daq buffer sync
            # counter += 1
            # if (counter % 100) == 0:
            #     self.StoreAIPointer()

            # also stop when daq was lost
            if self.DAQIsIdle():
                self.times['stop'] = TI.time()
                break

        if verbose:
            print('done recording')# %i.' % (self.recording_counter))


#______________________________________________________________________
# Data reporting
#______________________________________________________________________
    def RetrieveOutput(self, verbose = False):
        if verbose:
            print ('\t', self.rate, 'Hz on DAQ')
        # take data from the buffer
        data = NP.array(self.buffer).reshape([-1,len(self.channel_labels)])
        data = PD.DataFrame(data, columns = self.channel_labels)
        data.index = NP.arange(data.shape[0]) / self.rate + self.times['start'] 
        data.index.name = 'time'

        return self.times, data




################################################################################
### Force Plate Settings                                                     ###
################################################################################
def AssembleForceplateSettings():
    forceplate_settings = {'amti': {}, 'joystick': {}, 'kistler': {}, 'dualkistler': {}, 'dualkistler2': {}, 'test': {}}
    ### AMTI
    forceplate_settings['amti']['measures'] \
                    = ['forces', 'moments']

    forceplate_settings['amti']['data_columns'] \
                    = { \
                          'forces': ['F_x', 'F_y', 'F_z'] \
                        , 'moments': ['M_x', 'M_y', 'M_z'] \
                      }

    forceplate_settings['amti']['channel_order'] \
                    = ['F_x', 'F_y', 'F_z', 'M_x', 'M_y', 'M_z'] # channel order on the MCC board

    forceplate_settings['amti']['v_range'] \
                    = 10. # V
    forceplate_settings['amti']['colors'] \
                    = {   'F_x': (0.2,0.2,0.8), 'F_y': (0.2,0.8,0.2), 'F_z': (0.8,0.2,0.2) \
                        , 'M_x': (0.2,0.2,0.8), 'M_y': (0.2,0.8,0.2), 'M_z': (0.8,0.2,0.2) \
                      }


    ### Kistler
    forceplate_settings['kistler']['measures'] \
                    = ['Fxy', 'Fz']
    forceplate_settings['kistler']['data_columns'] \
                    = { \
                          'Fxy': ['Fx12', 'Fx34', 'Fy14', 'Fy23'] \
                        , 'Fz': ['Fz1', 'Fz2', 'Fz3', 'Fz4'] \
                      }

    forceplate_settings['kistler']['channel_order'] \
                    = ['Fx12', 'Fx34', 'Fy14', 'Fy23', 'Fz1', 'Fz2', 'Fz3', 'Fz4'] # channel order on the MCC board

    forceplate_settings['kistler']['v_range'] \
                    = 10. # V
    forceplate_settings['kistler']['colors'] \
                    = {   'Fx12': (0.4,0.2,0.8), 'Fx34': (0.2,0.4,0.8), 'Fy14': (0.2,0.8,0.4), 'Fy23': (0.4,0.8,0.2) \
                        , 'Fz1': (0.8,0.2,0.4), 'Fz2': (0.8,0.4,0.2), 'Fz3': (0.8,0.4,0.4), 'Fz4': (0.8,0.2,0.2) \
                      }
    # kistler_calib_gain100_1000


    ### Dual Kistler
    forceplate_settings['dualkistler']['measures'] \
                    = ['Fxy', 'Fz']
    forceplate_settings['dualkistler']['data_columns'] \
                    = { \
                          'Fxy':  ['AFx12', 'AFx34', 'AFy14', 'AFy23'] \
                                + ['BFx12', 'BFx34', 'BFy14', 'BFy23'] \
                        , 'Fz':   ['AFz1', 'AFz2', 'AFz3', 'AFz4'] \
                                + ['BFz1', 'BFz2', 'BFz3', 'BFz4'] \
                      }

    forceplate_settings['dualkistler']['channel_order'] \
                    = [ 'AFx12', 'AFx34', 'AFy14', 'AFy23' \
                      , 'BFx12', 'BFx34', 'BFy14', 'BFy23' \
                      , 'AFz1', 'AFz2', 'AFz3', 'AFz4' \
                      , 'BFz1', 'BFz2', 'BFz3', 'BFz4' \
                      ] # channel order on the MCC board

    forceplate_settings['dualkistler']['v_range'] \
                    = 10. # V
    forceplate_settings['dualkistler']['colors'] \
                    = {   'AFx12': (0.4, 0.2, 0.8), 'AFx34': (0.2, 0.4, 0.8), 'AFy14': (0.2, 0.8, 0.4), 'AFy23': (0.4, 0.8, 0.2) \
                        , 'BFx12': (0.4, 0.2, 0.8), 'BFx34': (0.2, 0.4, 0.8), 'BFy14': (0.2, 0.8, 0.4), 'BFy23': (0.4, 0.8, 0.2) \
                        , 'AFz1':  (0.8, 0.2, 0.4), 'AFz2':  (0.8, 0.4, 0.2), 'AFz3':  (0.8, 0.4, 0.4), 'AFz4':  (0.8, 0.2, 0.2) \
                        , 'BFz1':  (0.8, 0.2, 0.4), 'BFz2':  (0.8, 0.4, 0.2), 'BFz3':  (0.8, 0.4, 0.4), 'BFz4':  (0.8, 0.2, 0.2) \
                      }
    # kistler_calib_gain100_1000

 ### Dual Kistler
    forceplate_settings['dualkistler2']['measures'] \
                    = ['Fxy', 'Fz']
    forceplate_settings['dualkistler2']['data_columns'] \
                    = { \
                          'Fxy':  ['CFx12', 'CFx34', 'CFy14', 'CFy23'] \
                                + ['DFx12', 'DFx34', 'DFy14', 'DFy23'] \
                        , 'Fz':   ['CFz1', 'CFz2', 'CFz3', 'CFz4'] \
                                + ['DFz1', 'DFz2', 'DFz3', 'DFz4'] \
                      }

    forceplate_settings['dualkistler2']['channel_order'] \
                    = [ 'CFx12', 'CFx34', 'CFy14', 'CFy23' \
                      , 'DFx12', 'DFx34', 'DFy14', 'DFy23' \
                      , 'CFz1', 'CFz2', 'CFz3', 'CFz4' \
                      , 'DFz1', 'DFz2', 'DFz3', 'DFz4' \
                      ] # channel order on the MCC board

    forceplate_settings['dualkistler2']['v_range'] \
                    = 10. # V
    forceplate_settings['dualkistler2']['colors'] \
                    = {   'CFx12': (0.4, 0.2, 0.8), 'CFx34': (0.2, 0.4, 0.8), 'CFy14': (0.2, 0.8, 0.4), 'CFy23': (0.4, 0.8, 0.2) \
                        , 'DFx12': (0.4, 0.2, 0.8), 'DFx34': (0.2, 0.4, 0.8), 'DFy14': (0.2, 0.8, 0.4), 'DFy23': (0.4, 0.8, 0.2) \
                        , 'CFz1':  (0.8, 0.2, 0.4), 'CFz2':  (0.8, 0.4, 0.2), 'CFz3':  (0.8, 0.4, 0.4), 'CFz4':  (0.8, 0.2, 0.2) \
                        , 'DFz1':  (0.8, 0.2, 0.4), 'DFz2':  (0.8, 0.4, 0.2), 'DFz3':  (0.8, 0.4, 0.4), 'DFz4':  (0.8, 0.2, 0.2) \
                      }
    # kistler_calib_gain100_1000



    ### Joystick
    forceplate_settings['joystick']['measures'] \
                    = ['forces', 'moments']
    forceplate_settings['joystick']['data_columns'] \
                    = { \
                          'forces': ['x'] \
                        , 'moments': ['y'] \
                      }

    forceplate_settings['joystick']['channel_order'] \
                    = ['x', 'y'] # channel order on the MCC board

    forceplate_settings['joystick']['v_range'] \
                    = 3.3 # V
    forceplate_settings['joystick']['colors'] \
                    = {'x': (0.2,0.2,0.8), 'y': (0.2,0.8,0.2), 'z': (0.8,0.2,0.2)}


    forceplate_settings['test']['channel_order'] \
                    = ['v'] # channel order on the MCC board

    forceplate_settings['test']['v_range'] \
                    = 10. # V
    forceplate_settings['test']['colors'] \
                    = {'v': (0.2,0.2,0.2)}


    return forceplate_settings

forceplate_settings = AssembleForceplateSettings()




################################################################################
### MCC USB1608G DAQ, continuous acquisition                                 ###
################################################################################
class Oscilloscope(AnalogInput):
    # a MCC DAQ device, wired for analog input
    recording_mode = UL.ScanOption.CONTINUOUS

    def __init__(self, *args, **kwargs):
        super(Oscilloscope, self).__init__(*args, **kwargs)
        self.PreparePlot()



    def Exit(self, event):
        if event.key in ['q', 'e', 'escape', '<space>']:
            self._playing = False

    def PreparePlot(self):

        self.fig, self.ax = MPP.subplots(1, 1)

        # self.ax.set_ylim(-.5, .5)
        self.ax.set_ylim(-10.5, 10.5)
        self.ax.yaxis.tick_right()
        self.ax.yaxis.set_label_position("right")
        self.ax.set_title('press "E" to exit.')

        # draw all empty
        MPP.show(False)
        MPP.draw()
        self.fig.canvas.draw()

        self.fig.canvas.mpl_connect('key_press_event', self.Exit)

        # cache the background
        self.plot_background = self.fig.canvas.copy_from_bbox(self.ax.bbox)

        # prepare lines
        self.channel_handles = [self.ax.plot([0], [0], linestyle = '-')[0] for _ in range(len(self.channel_labels))]


    def Show(self, window = 2):
        # window: view time in seconds

        self.ax.set_xlim(-window, 0)

        self.recording_duration = window
        self.PrepareAcquisition()
        self.Record()
        # while (TI.time() - t0) <= 2.5:
        #     TI.sleep(0.25)
        #     print (self.GetStatus(), NP.round(TI.time() - t0, 2), 's')

        # print ('blub')
        self._playing = True
        t0 = TI.time()
        dt = TI.time() - t0
        while self._playing:
            dt = TI.time() - t0
            data = NP.array(self.buffer).reshape([-1,len(self.channel_labels)])
            timer = dt % window
            data = NP.roll(data, -int(timer * self.sampling_rate)-1, axis = 0)
            time = NP.arange(data.shape[0]) / self.sampling_rate 
            time -= NP.max(time)

            # restore background
            self.fig.canvas.restore_region(self.plot_background)

            # adjust and redraw data
            for ch, line in enumerate(self.channel_handles):
                line.set_data(time[:-1], data[:-1, ch]) # plot one less to avoid flickering
                self.ax.draw_artist(line)

            # fill in the axes rectangle
            self.fig.canvas.blit(self.ax.bbox)

            # TI.sleep(0.1)
            MPP.pause(1e-3)

            # if dt > 5.:
            #     self._playing = False

        # stop after a time
        self.analog_input.scan_stop()


    def Record(self):
        self.times['start'] = TI.time()
        self.rate = self.analog_input.a_in_scan(**self.recording_settings)


    def Close(self):
        MPP.close(self.fig)
        super(Oscilloscope, self).Close()



    def __exit__(self, *args, **kwargs):
        MPP.close(self.fig)
        super(Oscilloscope, self).__exit__(*args, **kwargs)


################################################################################
### Trigger on MCC DAQ Digital I/O                                           ###
################################################################################
class DIOTrigger(object):
    # a trigger, able to retrieve multiple input pins

    def __init__(self, digital_io, port, pin_list = [0], rising = True, scan_frq = 1e3):

        self.digital_io = digital_io
        self.port = port
        self.pin_list = pin_list
        self.rising = rising
        self.scan_frq = scan_frq
        self.Clean()

    def Clean(self):
        self.was_triggered = False
        self.triggered_bit = None
        self.trigger_time = None

    def Read(self):
        # read out the trigger bits
        return NP.array([self.digital_io.d_bit_in(self.port, pin_nr) for pin_nr in self.pin_list], dtype = bool)


    def Await(self):
        # wait until the trigger bit encounters a rising/falling edge.

        ### triggering loop
        self.Clean()
        self._waiting = True # enables external cancellation

        # store initial status, then update it constantly
        previous = self.Read()
        try:
            # first, wait for baseline condition (all FALSE on rising / all TRUE on falling edge)
            while NP.any(previous == self.rising):
                current = self.Read()
                if NP.any(NP.logical_xor(current, previous)):
                    previous = current
                    continue
                TI.sleep(1/self.scan_frq)

            # loop until bit changes
            while self._waiting:
                current = self.Read()
                t = TI.time()
                if NP.any(NP.logical_xor(current, previous)):
                    # trigger received
                    self.was_triggered = True
                    self.triggered_bit = self.pin_list[NP.argmax(NP.logical_xor(current, previous))]
                    self._waiting = False
                    self.trigger_time = t
                    break
                previous = current
                TI.sleep(1/self.scan_frq)

        except KeyboardInterrupt as ki:
            raise ki

        # return self.triggered_bit

################################################################################
### LED Indicator on MCC DAQ                                                 ###
################################################################################
class LED(object):
    # a connection to a digital input on a daq device

    HIGH = 255
    LOW = 0

    def __init__(self, digital_io, port, pin_nr = 0):

        self.digital_io = digital_io
        self.port = port
        self.pin_nr = pin_nr


    def Switch(self, value_bool):
        self.digital_io.d_bit_out(self.port, self.pin_nr, self.HIGH if value_bool else self.LOW)



################################################################################
### MCC USB1608G Force Plate                                                 ###
################################################################################

class TriggeredForcePlateDAQ(AnalogInput):

    # record on external trigger
    recording_mode = UL.ScanOption.EXTTRIGGER

#______________________________________________________________________
# Construction
#______________________________________________________________________
    def __init__(self, fp_type, pins \
                , *args, **kwargs):
        self.fp_type = fp_type

        # indicator and trigger pins
        self.pins = pins
        self.is_triggered = False
        self.has_indicator = False
        if (self.pins is not None) and (type(self.pins) is dict):
            self.has_indicator = self.pins.get('led', None) is not None

        # stores actual data
        self.Empty()


        ## initialize DAQ
        kwargs['channel_labels'] = forceplate_settings[self.fp_type]['channel_order']
        super(TriggeredForcePlateDAQ, self).__init__(*args, **kwargs)

        self.SetPins()

        ## connect LED
        if self.has_indicator:
            self.led = LED(   digital_io = self.digital_io \
                            , port = self.port \
                            , pin_nr = self.pins['led']\
                            )
        
        # store label
        self.label = instrument_labels[self.daq_device.get_descriptor().unique_id]


    def Empty(self):
        # remove previous data
        self.sync = []
        self.data = None
        

    def SetPins(self):
        # set pin to input
        if self.has_indicator:
            # set pin to output
            self.digital_io.d_config_bit(self.port, self.pins['led'], UL.DigitalDirection.OUTPUT)


#______________________________________________________________________
# Control
#______________________________________________________________________

    def Indicate(self, value):
        # flash an LED
        if not self.has_indicator:
            return

        self.led.Switch(value)



    def StdOut(self, *args, **kwargs):
        print(*args, **kwargs)


#______________________________________________________________________
# I/O
#______________________________________________________________________
    def Record(self):
        self.StdOut('waiting for trigger... ' , end = '\r')
        self.TriggeredRecording()
        self.StdOut('done! ', ' '*20)



    def TriggeredRecording(self):

        # start recording in the background (will wait for trigger)
        self._armed = True
        self.rate = self.analog_input.a_in_scan(**self.recording_settings)

        self.sync.append([TI.time(), -1] )

        # wait until force plate records
        while self.GetDAQStatus()[1].current_scan_count == 0:
            TI.sleep(1/self.scan_frq)

        self._recording = True
        # store start time
        # self.sync.append([TI.time(), -self.GetDAQStatus()[1].current_total_count] )
        # turn LED on
        self.Indicate(True)

        # wait until recording has ended
        self.StdOut('recording... ', ' '*20, end = '\r')
        counter = 0
        while not self.DAQIsIdle():
            if (counter % self.sampling_rate) == 0:
                self.sync.append([TI.time(), self.GetDAQStatus()[1].current_scan_count] )

            TI.sleep(1/self.scan_frq)
            counter += 1

        # store stop time
        self.sync.append([TI.time(), -1] )
        self._recording = False
        self._armed = False
        # turn LED off
        self.Indicate(False)


#______________________________________________________________________
# Data reporting
#______________________________________________________________________
    def RetrieveOutput(self):
        data = NP.array(self.buffer).reshape([-1,len(self.channel_labels)])
        data = PD.DataFrame(data, columns = self.channel_labels)
        data.index = NP.arange(data.shape[0]) / self.rate 
        data.index.name = 'time'

        time_out = PD.DataFrame(NP.stack(self.sync, axis = 0), columns = ['time', 'current_scan_count'])
        # print (time_out)

        return time_out, data



################################################################################
### Force Plate Data                                                         ###
################################################################################
def PlotJoystick(data):

    fig = MPP.figure()
    PreparePlot()

    xy_ax = fig.add_subplot(1,1,1)

    for param in forceplate_settings['joystick']['channel_order']:
        # print (param, data.loc[:, param].values)
        xy_ax.plot(data.index.values, data.loc[:, param].values, color = forceplate_settings['joystick']['colors'][param])

    PolishAx(xy_ax)

    xy_ax.set_xlim([NP.min(data.index.values), NP.max(data.index.values)])
    # xy_ax.set_ylim([-forceplate_settings['joystick']['v_range'], forceplate_settings['joystick']['v_range']])
    xy_ax.set_xlabel('time (s)')
    xy_ax.set_ylabel('voltage (V)')
    MPP.show()


def PlotKistlerForces(data):

    fig = MPP.figure()
    PreparePlot()

    xy_ax = fig.add_subplot(2,1,1)
    z_ax = fig.add_subplot(2,1,2, sharex = xy_ax, sharey = xy_ax)


    for param in forceplate_settings['kistler']['data_columns']['Fxy']:
        # print (param, data.loc[:, param].values)
        xy_ax.plot(data.index.values, data.loc[:, param].values, color = forceplate_settings['kistler']['colors'][param])
    for param in forceplate_settings['kistler']['data_columns']['Fz']:
        z_ax.plot(data.index.values, data.loc[:, param].values, color = forceplate_settings['kistler']['colors'][param])

    PolishAx(xy_ax)
    PolishAx(z_ax)

    xy_ax.set_xlim([NP.min(data.index.values), NP.max(data.index.values)])
    xy_ax.set_ylim([-forceplate_settings['kistler']['v_range'], forceplate_settings['kistler']['v_range']])
    z_ax.set_xlabel('time (s)')
    xy_ax.set_ylabel('voltage (V)')
    z_ax.set_ylabel('voltage (V)')
    MPP.show()



def TestDAQAnalog():
    with AnalogInput( \
                  sampling_rate = 1e3 \
                , recording_duration = 2. \
                , channel_labels = forceplate_settings['joystick']['channel_order'] \
                , scan_frq = 1e6 \
                ) \
        as ai:

        ai.Record()
        times, data = ai.RetrieveOutput()
        # print (data)
    PlotJoystick(data)




def TestOscilloscope():
    with Oscilloscope( \
                  sampling_rate = 1e3 \
                , channel_labels = forceplate_settings['test']['channel_order'] \
                , scan_frq = 1e6 \
                ) \
        as osci:

        osci.Show(window = 16)
        



def TestForcePlate():
    with DAQForcePlate( \
                  fp_type = 'kistler' \
                , pins = {'trigger': 5, 'baseline': 4, 'reference': 3, 'led': 7} \
                , baseline_duration = 1. \
                , recording_duration = 1. \
                , sampling_rate = 1e3 \
                , scan_frq = 1e6 \
                ) \
        as fp: 
        try:
            fp.TriggeredRecording()
        except Exception as e:
            print (e, '\n')

        times, data = fp.CombinedOutput()
        print (times, data)        

        PlotKistlerForces(data)



################################################################################
### Multi DAQ Testing                                                        ###
################################################################################
def PlotDualKistlerForces(data):

    fig = MPP.figure()
    PreparePlot()

    xy_ax = fig.add_subplot(2,1,1)
    z_ax = fig.add_subplot(2,1,2, sharex = xy_ax, sharey = xy_ax)


    for param in forceplate_settings['dualkistler']['data_columns']['Fxy']:
        # print (param, data.loc[:, param].values)
        xy_ax.plot(data.index.values, data.loc[:, param].values, color = forceplate_settings['dualkistler']['colors'][param])
    for param in forceplate_settings['dualkistler']['data_columns']['Fz']:
        z_ax.plot(data.index.values, data.loc[:, param].values, color = forceplate_settings['dualkistler']['colors'][param])

    PolishAx(xy_ax)
    PolishAx(z_ax)

    xy_ax.set_xlim([NP.min(data.index.values), NP.max(data.index.values)])
    xy_ax.set_ylim([-forceplate_settings['dualkistler']['v_range'], forceplate_settings['dualkistler']['v_range']])
    z_ax.set_xlabel('time (s)')
    xy_ax.set_ylabel('voltage (V)')
    z_ax.set_ylabel('voltage (V)')
    MPP.show()



def TestMultiDAQ():
    with TriggeredForcePlateDAQ( \
                  fp_type = 'dualkistler' \
                , device_nr = 1 \
                , pins = {'led': 7} \
                , sampling_rate = 1e3 \
                , scan_frq = 1e6 \
                , recording_duration = 6. \
                ) \
        as fp: 
        try:
            fp.Record()
        except Exception as e:
            print (e, '\n')

        times, data = fp.store[0]
        # print (times, data)        

        PlotDualKistlerForces(data)

    
    # pin_nr = 7
    # daq1_blue = DAQ_DigitalOutput(digital_pin = pin_nr, device_nr = 0)
    # daq2_green = DAQ_DigitalOutput(digital_pin = pin_nr, device_nr = 1)
    
    # daq1_blue.Flash(times = 3, duration = 1)
    # daq2_green.Flash(times = 3, duration = 1)

    # TestGPIO(7) # LED on pin 7
    # TestGPIOInput(5) # trigger in on pin 5
    # trigger out on pin 3



################################################################################
### MCC USB1608G Force Plate                                                 ###
################################################################################

class PostTriggerDAQ(AnalogInput):

    # record on external trigger
    recording_mode = UL.ScanOption.CONTINUOUS

#______________________________________________________________________
# Construction
#______________________________________________________________________
    def __init__(self, fp_type, pins, rising = True \
                , *args, **kwargs):
        self.fp_type = fp_type

        # indicator and trigger pins
        self.pins = pins
        self.is_triggered = False
        self.has_indicator = False
        if (self.pins is not None) and (type(self.pins) is dict):
            self.has_indicator = self.pins.get('led', None) is not None

            self.has_triggerpins = self.pins.get('triggers', None) is not None

        if not self.has_triggerpins:
            raise IOError("post trigger recordings require a trigger pin!")

        # stores actual data
        self.Empty()


        ## initialize DAQ
        kwargs['channel_labels'] = forceplate_settings[self.fp_type]['channel_order']
        super(PostTriggerDAQ, self).__init__(*args, **kwargs)

        self.SetPins()

        ## connect LED
        if self.has_indicator:
            self.led = LED(   digital_io = self.digital_io \
                            , port = self.port \
                            , pin_nr = self.pins['led']\
                            )
        
        ## connect trigger
        if self.has_triggerpins:
            self.trigger = DIOTrigger( \
                              digital_io = self.digital_io \
                            , port = self.port \
                            , pin_list = self.pins['triggers'] \
                            , rising = rising \
                            , scan_frq = self.scan_frq \
                            )

        # store label
        self.label = instrument_labels[self.daq_device.get_descriptor().unique_id]


    def Empty(self):
        # remove previous data
        self.sync = []
        self.data = None
        

    def SetPins(self):
        # set led pin to output
        if self.has_indicator:
            self.digital_io.d_config_bit(self.port, self.pins['led'], UL.DigitalDirection.OUTPUT)

        # set trigger pin to input
        if self.has_triggerpins:
            for pin in self.pins['triggers']:
                self.digital_io.d_config_bit(self.port, pin, UL.DigitalDirection.INPUT)


#______________________________________________________________________
# Control
#______________________________________________________________________

    def Indicate(self, value):
        # flash an LED
        if not self.has_indicator:
            return

        self.led.Switch(value)


    def AwaitTrigger(self):
        # initiate trigger thread
        self._waiting = True
        self._trig = TH.Thread(target = self.trigger.Await)
        self._trig.daemon = True
        self._trig.start()
        

    def AbortRecording(self):
        # unexpectedly abort a recording
        self._end_time = TI.time()
        try:
            self.analog_input.scan_stop()
        except UL.ul_exception.ULException as ule:
            pass # ignore if the device was closed before

        self.trigger._waiting = False
        self._trig.join()
        self.Indicate(False)


    def StdOut(self, *args, **kwargs):
        print(*args, **kwargs)


    def Quit(self, *args, **kwargs):
        self.AbortRecording()
        super(PostTriggerDAQ, self).Quit(*args, **kwargs)
    

#______________________________________________________________________
# Recording Procedure
#______________________________________________________________________
    def Record(self):
        # self.StdOut('waiting for recording... ' , end = '\r')
        self.TriggeredRecording()
        self.StdOut('done! ', ' '*20)


    def Start(self):
        # start recording in the background (will wait for trigger)
        self._armed = True
        self._recording = True
        self.StdOut('recording... waiting for trigger.', ' '*20, end = '\r')
        self._start_time = TI.time()
        self.rate = self.analog_input.a_in_scan(**self.recording_settings)

        # self.sync.append([TI.time(), -1] )
        # turn LED on
        self.Indicate(True)
        self.AwaitTrigger()


    def Stop(self):
        # stop daq
        self.analog_input.scan_stop()
        # stop trigger
        self.trigger._waiting = False

        # store stop time
        self._stop_sample = self.GetDAQStatus()[1].current_scan_count
        # print (self._stop_sample)

        self._end_time = self.trigger.trigger_time
        self.sync.append([self._end_time, self._stop_sample] )

        # adjust status
        self._recording = False
        self._armed = False

        # turn LED off
        self.Indicate(False)

        # await trigger finish
        self._trig.join()



    def TriggeredRecording(self):
        # start
        self.Start()

        # wait until recording has ended
        counter = 0
        while not self.trigger.was_triggered:
            if (counter % self.sampling_rate) == 0:
                self.sync.append([TI.time(), self.GetDAQStatus()[1].current_scan_count] )

            TI.sleep(1/self.scan_frq)
            counter += 1

        # stop
        self.Stop()



#______________________________________________________________________
# Data reporting
#______________________________________________________________________
    def RetrieveOutput(self):
        # dt = self._end_time - self._start_time
        data = NP.array(self.buffer).reshape([-1,len(self.channel_labels)])
        # timer = dt % self.recording_duration
        # data = NP.roll(data, -int(timer * self.sampling_rate)-1, axis = 0)
        data = NP.roll(data, -(self._stop_sample % data.shape[0]), axis = 0)
        time = NP.arange(data.shape[0]) / self.sampling_rate 
        time -= NP.max(time)

        data = PD.DataFrame(data, columns = self.channel_labels)
        data.index = time 
        data.index.name = 'time'

        sync = PD.DataFrame(NP.stack(self.sync, axis = 0), columns = ['time', 'current_scan_count'])
        # print (sync)

        return sync, data


def TestPostTriggerRecording():
    with PostTriggerDAQ( \
                          fp_type = 'joystick' \
                        , pins = {'led': 7, 'triggers': [5]} \
                        , rising = False \
                        , sampling_rate = 1e3 \
                        , recording_duration = 2. \
                        , channel_labels = forceplate_settings['joystick']['channel_order'] \
                        , scan_frq = 1e6 \
                        ) as daq:

        daq.Record()
        times, data = daq.RetrieveOutput()
        print (times, data)

    PlotJoystick(data)

################################################################################
### Camera via OpenCV                                                        ###
################################################################################
class Camera(object):
    # using the odroid oCam for quick video recording

    def __init__(self, recording_duration = 1., cam_nr = 0, label = 'camera', daq = None):
        self.recording_duration = recording_duration # s
        self.cam_nr = cam_nr
        self.label = label
        self.daq = daq
        self._recording = False
        EXIT.register(self.Quit)

        self.cam = CV.VideoCapture(index = self.cam_nr)
        self.StdOut('Camera connected.')

        self.PrepareAcquisition()

    def RestartCamera(self):
        self.cam.release()
        self.cam = CV.VideoCapture(index = self.cam_nr)
        self.PrepareAcquisition()


    def PrepareAcquisition(self):
        self.fps = 80

        # self.cam.set(CV.CAP_PROP_CONVERT_RGB, False)
        # self.cam.set(CV.CAP_PROP_FPS, self.fps) # only relevant if multiple fps supported per mode
        # self.cam.set(CV.CAP_PROP_FRAME_WIDTH, 1280)
        # self.cam.set(CV.CAP_PROP_FRAME_HEIGHT, 720)

        self.cam.set(CV.CAP_PROP_CONVERT_RGB, False)
        self.cam.set(CV.CAP_PROP_FPS, self.fps) # only relevant if multiple fps supported per mode
        self.cam.set(CV.CAP_PROP_FRAME_WIDTH, 640)
        self.cam.set(CV.CAP_PROP_FRAME_HEIGHT, 480)

        # clean buffer
        self.Empty()


    def StdOut(self, *args, **kwargs):
        print(*args, **kwargs)


    def Empty(self):
        # remove previous data
        buffer_size = int(self.recording_duration * self.fps * 1.05)
        # print (buffer_size)
        self.buffer = Deque( maxlen = buffer_size )

        for _ in range(10):
            self.cam.grab()
        

    def DAQIsArmed(self):
        if self.daq is None:
            return (TI.time() - self.start) <= self.recording_duration
        else:
            return self.daq._armed

    def DAQIsRecording(self):
        if self.daq is None:
            return True
        else:
            return self.daq._recording


    def Record(self):
        # This should happen in a thread!
        
        TI.sleep(0.001) # to make sure daq is armed first
        self.Empty()

        self.start = TI.time()
        now = TI.time()

        self._recording = True
        self.StdOut('starting!')
        while self.DAQIsArmed():
            if self.DAQIsRecording():
                now = TI.time()
                ret, frame = self.cam.read()

                if ret:
                    self.buffer.append((now, frame))

            TI.sleep(0.001)

        self._recording = False
        self.StdOut('done recording %.1f seconds!' % (now-self.start))


    def SampleRecording(self):
        self._thread = TH.Thread(target = self.Record)
        self._thread.start()

        TI.sleep(0.1)
        self._thread.join()
        self._thread = None
        

    def AbortRecording(self):
        recdur = self.recording_duration
        self.recording_duration = -1.
        TI.sleep(0.1)
        self.recording_duration = recdur


    def RetrieveOutput(self):        
        sync = []
        data = []
        for t, frame in self.buffer:
            sync.append(t)
            data.append(frame)

        sync = NP.array(sync)
        if len(data) > 1:
            data = NP.stack(data, axis = 2)

        return sync, data


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
        # safely exit
        if self._recording:
            self.AbortRecording()

        self.cam.release()
        CV.destroyAllWindows()
        # SYS.exit()
        print("safely exited %s." % (self.label))


def TestCamera():
    with Camera(recording_duration = 1., cam_nr = 0) as cam1:

        print ("camera started. Recording!")
        cam1.SampleRecording()

        sync, data = cam1.RetrieveOutput()
        NP.savez('test', time = sync, images = data)


    print (sync.shape, data.shape)

    print (sync)
    print (data)



################################################################################
### Mission Control                                                          ###
################################################################################
if __name__ == "__main__":
    pass
    # print ([so for so in UL.ScanOption])

    # TestCamera()

    # TestPostTriggerRecording()

    ### MCC USB1608G DAQ function
    # TestMCCPinIn(pin_nr = 3)
    # TestMCCPinOut(pin_nr = 7)

    ### Force Plates
    # TestDAQAnalog()
    # TestOscilloscope()
    # TestForcePlate()

    # TestQuickAnalysis()

    # TestMultiDAQ()

