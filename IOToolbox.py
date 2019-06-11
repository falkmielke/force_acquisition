
# Universiteit Antwerpen
# Functional Morphology
# Falk Mielke
# 2019/06/06

# contains parts as ADAPTATION of https://github.com/adafruit/Adafruit_Python_GPIO
# used to connect I2C via the FT232H interface



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
from collections import deque as DEQue # double ended queue
import numpy as NP # numerics
import pandas as PD # data storage
import scipy.signal as SIG
import math as MATH
import matplotlib as MP # plotting
import matplotlib.pyplot as MPP # plot control


import ftdi1 as FTDI
import uldaq as UL # MCC DAQ negotiation


import logging
logger = logging.getLogger(__name__)

### MCC DAQ drivers and library
# follow readme in https://github.com/mccdaq/uldaq
    # download   $ wget https://github.com/mccdaq/uldaq/releases/download/v1.1.1/libuldaq-1.1.1.tar.bz2
    # extract    $ tar -xvjf libuldaq-1.1.1.tar.bz2 && cd libuldaq-1.1.1
    # build      $ ./configure && make -j4 && sudo make install -j4
# pip install uldaq

### FTDI FT232H breakout drivers and library
## driver
# see https://www.ftdichip.com/Drivers/D2XX.htm
# wget https://www.ftdichip.com/Drivers/D2XX/Linux/libftd2xx-x86_64-1.4.8.gz
# tar -xvzf libftd2xx-x86_64-1.4.8.gz
# mv release ftd2xx
# cd ftd2xx/build/
# cp libftd2xx.* /usr/local/lib
# chmod 0755 /usr/local/lib/libftd2xx.so.1.4.8
# ln -sf /usr/local/lib/libftd2xx.so.1.4.8 /usr/local/lib/libftd2xx.so


## software
# wget https://www.intra2net.com/en/developer/libftdi/download/libftdi1-1.4.tar.bz2
# tar -xvf libftdi1-1.4.tar.bz2
# cd libftdi1-1.4
# mkdir build && cd build

# cmake -DCMAKE_INSTALL_PREFIX="/usr" ../
# make
# make install
#

## also:
# pip install pyftdi pylibftdi

################################################################################
### Global Specifications                                                    ###
################################################################################
coordinates = ['x', 'y', 'z']

FT232H_VID = 0x0403   # Default FTDI FT232H vendor ID
FT232H_PID = 0x6014   # Default FTDI FT232H product ID

_REPEAT_DELAY = 4



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
    MP.rcParams['text.usetex'] = True
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
### Base GPIO Functionality                                                  ###
################################################################################

RISING      = 1
FALLING     = 2
BOTH        = 3

PUD_OFF  = 0
PUD_DOWN = 1
PUD_UP   = 2


class BaseGPIO(object):
    """Base class for implementing simple digital IO for a platform.
    Implementors are expected to subclass from this and provide an implementation
    of the setup, output, and input functions."""

    def setup(self, pin, mode, pull_up_down=PUD_OFF):
        """Set the input or output mode for a specified pin.  Mode should be
        either OUT or IN."""
        raise NotImplementedError

    def output(self, pin, value):
        """Set the specified pin the provided high/low value.  Value should be
        either HIGH/LOW or a boolean (true = high)."""
        raise NotImplementedError

    def input(self, pin):
        """Read the specified pin and return HIGH/true if the pin is pulled high,
        or LOW/false if pulled low."""
        raise NotImplementedError

    def set_high(self, pin):
        """Set the specified pin HIGH."""
        self.output(pin, HIGH)

    def set_low(self, pin):
        """Set the specified pin LOW."""
        self.output(pin, LOW)

    def is_high(self, pin):
        """Return true if the specified pin is pulled high."""
        return self.input(pin) == HIGH

    def is_low(self, pin):
        """Return true if the specified pin is pulled low."""
        return self.input(pin) == LOW


# Basic implementation of multiple pin methods just loops through pins and
# processes each one individually. This is not optimal, but derived classes can
# provide a more optimal implementation that deals with groups of pins
# simultaneously.
# See MCP230xx or PCF8574 classes for examples of optimized implementations.

    def output_pins(self, pins):
        """Set multiple pins high or low at once.  Pins should be a dict of pin
        name to pin value (HIGH/True for 1, LOW/False for 0).  All provided pins
        will be set to the given values.
        """
        # General implementation just loops through pins and writes them out
        # manually.  This is not optimized, but subclasses can choose to implement
        # a more optimal batch output implementation.  See the MCP230xx class for
        # example of optimized implementation.
        for pin, value in iter(pins.items()):
            self.output(pin, value)

    def setup_pins(self, pins):
        """Setup multiple pins as inputs or outputs at once.  Pins should be a
        dict of pin name to pin type (IN or OUT).
        """
        # General implementation that can be optimized by derived classes.
        for pin, value in iter(pins.items()):
            self.setup(pin, value)

    def input_pins(self, pins):
        """Read multiple pins specified in the given list and return list of pin values
        HIGH/True if the pin is pulled high, or LOW/False if pulled low.
        """
        # General implementation that can be optimized by derived classes.
        return [self.input(pin) for pin in pins]


    def add_event_detect(self, pin, edge):
        """Enable edge detection events for a particular GPIO channel.  Pin 
        should be type IN.  Edge must be RISING, FALLING or BOTH.
        """
        raise NotImplementedError
   
    def remove_event_detect(self, pin):
        """Remove edge detection for a particular GPIO channel.  Pin should be
        type IN.
        """
        raise NotImplementedError
  
    def add_event_callback(self, pin, callback):
        """Add a callback for an event already defined using add_event_detect().
        Pin should be type IN.
        """
        raise NotImplementedError

    def event_detected(self, pin):
        """Returns True if an edge has occured on a given   You need to 
        enable edge detection using add_event_detect() first.   Pin should be 
        type IN.
        """
        raise NotImplementedError

    def wait_for_edge(self, pin, edge):
        """Wait for an edge.   Pin should be type IN.  Edge must be RISING, 
        FALLING or BOTH."""
        raise NotImplementedError

    def cleanup(self, pin=None):
        """Clean up GPIO event detection for specific pin, or all pins if none 
        is specified.
        """
        raise NotImplementedError


# helper functions useful to derived classes

    def _validate_pin(self, pin):
        # Raise an exception if pin is outside the range of allowed values.
        if pin < 0 or pin >= self.NUM_GPIO:
            raise ValueError('Invalid GPIO value, must be between 0 and {0}.'.format(self.NUM_GPIO))

    def _bit2(self, src, bit, val):
        bit = 1 << bit
        return (src | bit) if val else (src & ~bit)



################################################################################
### FT232H Initialization                                                    ###
################################################################################
def _check_running_as_root():
    # NOTE: Checking for root with user ID 0 isn't very portable, perhaps
    # there's a better alternative?
    if OS.geteuid() != 0:
        raise RuntimeError('Expected to be run by root user! Try running with sudo.')


def disable_FTDI_driver():
    """Disable the FTDI drivers for the current platform.  This is necessary
    because they will conflict with libftdi and accessing the FT232H.  Note you
    can enable the FTDI drivers again by calling enable_FTDI_driver.
    """
    SP.call('modprobe -r -q ftdi_sio', shell=True)
    SP.call('modprobe -r -q usbserial', shell=True)
    # Note there is no need to disable FTDI drivers on Windows!

def enable_FTDI_driver():
    """Re-enable the FTDI drivers for the current platform."""
    # SP.check_call('modprobe -q ftdi_sio', shell=True)
    # SP.check_call('modprobe -q usbserial', shell=True)
    pass

def ManageDrivers():
    disable_FTDI_driver()
    EXIT.register(enable_FTDI_driver)




def FindDevices(vid = FT232H_VID, pid = FT232H_PID):
    """Return a list of all FT232H device serial numbers connected to the
    machine.  You can use these serial numbers to open a specific FT232H device
    by passing it to the FT232H initializer's serial parameter.
    """
    try:
        # Create a libftdi context.
        ctx = None
        ctx = FTDI.new()
        # Enumerate FTDI devices.
        device_list = None
        count, device_list = FTDI.usb_find_all(ctx, vid, pid)
        if count < 0:
            raise RuntimeError('ftdi_usb_find_all returned error {0}: {1}'.format(count, FTDI.get_error_string(self._ctx)))
        # Walk through list of devices and assemble list of serial numbers.
        devices = []
        while device_list is not None:
            # Get USB device strings and add serial to list of devices.
            ret, manufacturer, description, serial = FTDI.usb_get_strings(ctx, device_list.dev, 256, 256, 256)
            if serial is not None:
                devices.append(serial)
            device_list = device_list.next
        return devices
    finally:
        # Make sure to clean up list and context when done.
        if device_list is not None:
            FTDI.list_free(device_list)
        if ctx is not None:
            FTDI.free(ctx)


def FindAllI2CSlaves(master):

    print ('Scanning all I2C bus addresses...')
    # Enumerate all I2C addresses.
    for address in range(127):
        # Skip I2C addresses which are reserved.
        if address <= 7 or address >= 120:
            continue
        # Create I2C object.
        i2c = I2CDevice(master, address)
        # Check if a device responds to this address.
        if i2c.ping():
            print ('Found I2C device at address 0x{0:02X}'.format(address))
    print ('Done!')




################################################################################
### FT232H Control                                                           ###
################################################################################
class FT232H(BaseGPIO):

    def __init__(self, vid = FT232H_VID, pid = FT232H_PID, serial = None, multiplexer_address = None, clock_hz = 1e6):
        """Create a FT232H object.  Will search for the first available FT232H
        device with the specified USB vendor ID and product ID (defaults to
        FT232H default VID & PID).  Can also specify an optional serial number
        string to open an explicit FT232H device given its serial number.  See
        the FT232H.enumerate_device_serials() function to see how to list all
        connected device serial numbers.
        """
        # Initialize FTDI device connection.
        ManageDrivers()

        self._ctx = FTDI.new()
        if self._ctx == 0:
            raise RuntimeError('ftdi_new failed! Is libftdi1 installed?')
        # Register handler to close and cleanup FTDI context on program exit.
        EXIT.register(self.close)
        if serial is None:
            # Open USB connection for specified VID and PID if no serial is specified.
            self._check(FTDI.usb_open, vid, pid)
        else:
            # Open USB connection for VID, PID, serial.
            self._check(FTDI.usb_open_string, 's:{0}:{1}:{2}'.format(vid, pid, serial))
        # Reset device.
        self._check(FTDI.usb_reset)
        # Disable flow control. Commented out because it is unclear if this is necessary.
        #self._check(FTDI.setflowctrl, FTDI.SIO_DISABLE_FLOW_CTRL)
        # Change read & write buffers to maximum size, 65535 bytes.
        self._check(FTDI.read_data_set_chunksize, 65535)
        self._check(FTDI.write_data_set_chunksize, 65535)
        # Clear pending read data & write buffers.
        self._check(FTDI.usb_purge_buffers)
        # Enable MPSSE and syncronize communication with device.
        self._mpsse_enable()
        self._mpsse_sync()
        # Initialize all GPIO as inputs.
        self._write(b'\x80\x00\x00\x82\x00\x00')
        self._direction = 0x0000
        self._level = 0x0000

        # connecting a TCA9548A Multiplexer
        self.pins_in_use = []

        if multiplexer_address is not None:
            self.multiplexer = TCA9548A(self, multiplexer_address, clock_hz = clock_hz)


    def close(self):
        """Close the FTDI device.  Will be automatically called when the program ends."""
        if self._ctx is not None:
            FTDI.free(self._ctx)
        self._ctx = None

    def _write(self, string):
        """Helper function to call write_data on the provided FTDI device and
        verify it succeeds.
        """
        # Get modem status. Useful to enable for debugging.
        #ret, status = FTDI.poll_modem_status(self._ctx)
        #if ret == 0:
        #   logger.debug('Modem status {0:02X}'.format(status))
        #else:
        #   logger.debug('Modem status error {0}'.format(ret))
        length = len(string)
        # try:
        #     ret = FTDI.write_data(self._ctx, string, length)
        # except TypeError:
        # print (string)
        ret = FTDI.write_data(self._ctx, string); #compatible with libFtdi 1.3
        # Log the string that was written in a python hex string format using a very
        # ugly one-liner list comprehension for brevity.
        #logger.debug('Wrote {0}'.format(''.join(['\\x{0:02X}'.format(ord(x)) for x in string])))
        if ret < 0:
            raise RuntimeError('ftdi_write_data failed with error {0}: {1}'.format(ret, FTDI.get_error_string(self._ctx)))
        if ret != length:
            raise RuntimeError('ftdi_write_data expected to write {0} bytes but actually wrote {1}!'.format(length, ret))

    def _check(self, command, *args):
        """Helper function to call the provided command on the FTDI device and
        verify the response matches the expected value.
        """
        ret = command(self._ctx, *args)
        logger.debug('Called ftdi_{0} and got response {1}.'.format(command.__name__, ret))
        if ret != 0:
            raise RuntimeError('ftdi_{0} failed with error {1}: {2}'.format(command.__name__, ret, FTDI.get_error_string(self._ctx)))

    def _poll_read(self, expected, timeout_s=5.0):
        """Helper function to continuously poll reads on the FTDI device until an
        expected number of bytes are returned.  Will throw a timeout error if no
        data is received within the specified number of timeout seconds.  Returns
        the read data as a string if successful, otherwise raises an execption.
        """
        start = TI.time()
        # Start with an empty response buffer.
        response = bytearray(expected)
        # print ('response', response, expected)
        index = 0
        # Loop calling read until the response buffer is full or a timeout occurs.
        while TI.time() - start <= timeout_s:
            ret, data = FTDI.read_data(self._ctx, expected - index)
            # print (ret, data)


            # py2:
            # ('response', bytearray(b'\x00\x00'), 2)
            # (0, '\x00\xd7')
            # (0, '\x00\xd7')
            # (2, '\xfa\xab')
            # py3:
            # response bytearray(b'\x00\x00') 2
            # 0 b'\x90='
            # 0 b'\x90='
            # 2 b'\xfa\xab'

            # Fail if there was an error reading data.
            if ret < 0:
                raise RuntimeError('ftdi_read_data failed with error code {0}.'.format(ret))
            # Add returned data to the buffer.
            response[index:index+ret] = data[:ret]
            # index += ret
            # Buffer is full, return the result data.
            # if index >= expected:
            #     return str(response)
            ## Py3: (FM)
            if ret >= expected:
                return response
            TI.sleep(0.01)
        raise RuntimeError('Timeout while polling ftdi_read_data for {0} bytes!'.format(expected))

    def _mpsse_enable(self):
        """Enable MPSSE mode on the FTDI device."""
        # Reset MPSSE by sending mask = 0 and mode = 0
        self._check(FTDI.set_bitmode, 0, 0)
        # Enable MPSSE by sending mask = 0 and mode = 2
        self._check(FTDI.set_bitmode, 0, 2)

    def _mpsse_sync(self, max_retries=10):
        """Synchronize buffers with MPSSE by sending bad opcode and reading expected
        error response.  Should be called once after enabling MPSSE."""
        # Send a bad/unknown command (0xAB), then read buffer until bad command
        # response is found.
        self._write(b'\xAB')
        # Keep reading until bad command response (0xFA 0xAB) is returned.
        # Fail if too many read attempts are made to prevent sticking in a loop.
        tries = 0
        sync = False
        while not sync:
            data = self._poll_read(2)
            if data == b'\xfa\xab':
                sync = True
            tries += 1
            if tries >= max_retries:
                raise RuntimeError('Could not synchronize with FT232H!')

    def mpsse_set_clock(self, clock_hz, adaptive=False, three_phase=False):
        """Set the clock speed of the MPSSE engine.  Can be any value from 450hz
        to 30mhz and will pick that speed or the closest speed below it.
        """
        # Disable clock divisor by 5 to enable faster speeds on FT232H.
        self._write(b'\x8A')
        # Turn on/off adaptive clocking.
        if adaptive:
            self._write(b'\x96')
        else:
            self._write(b'\x97')
        # Turn on/off three phase clock (needed for I2C).
        # Also adjust the frequency for three-phase clocking as specified in section 2.2.4
        # of this document:
        #   http://www.ftdichip.com/Support/Documents/AppNotes/AN_255_USB%20to%20I2C%20Example%20using%20the%20FT232H%20and%20FT201X%20devices.pdf
        if three_phase:
            self._write(b'\x8C')
        else:
            self._write(b'\x8D')
        # Compute divisor for requested clock.
        # Use equation from section 3.8.1 of:
        #  http://www.ftdichip.com/Support/Documents/AppNotes/AN_108_Command_Processor_for_MPSSE_and_MCU_Host_Bus_Emulation_Modes.pdf
        # Note equation is using 60mhz master clock instead of 12mhz.
        divisor = int(MATH.ceil((30000000.0-float(clock_hz))/float(clock_hz))) & 0xFFFF
        if three_phase:
            divisor = int(divisor*(2.0/3.0))
        logger.debug('Setting clockspeed with divisor value {0}'.format(divisor))
        # Send command to set divisor from low and high byte values.

        # my_str = "hello world"
        # my_str_as_bytes = str.encode(my_str)
        # print(my_str_as_bytes, type(my_str_as_bytes) )
        # my_decoded_str = my_str_as_bytes.decode()
        # print(my_decoded_str, type(my_decoded_str) )
        # val = (0x86, divisor & 0xFF, (divisor >> 8) & 0xFF)
        # ba = bytearray(val)
        # help(ba)
        # string = bytes(ba)
        # print(val, ba, string)
        # print (str(bytearray((0x86, divisor & 0xFF, (divisor >> 8) & 0xFF))))
        # print (bytes(bytearray((0x86, divisor & 0xFF, (divisor >> 8) & 0xFF))))

        self._write(bytes(bytearray((0x86, divisor & 0xFF, (divisor >> 8) & 0xFF))))

    def mpsse_read_gpio(self):
        """Read both GPIO bus states and return a 16 bit value with their state.
        D0-D7 are the lower 8 bits and C0-C7 are the upper 8 bits.
        """
        # Send command to read low byte and high byte.
        self._write(b'\x81\x83')
        # Wait for 2 byte response.
        data = self._poll_read(2)
        # Assemble response into 16 bit value.
        low_byte = data[0]
        high_byte = data[1]
        logger.debug('Read MPSSE GPIO low byte = {0:02X} and high byte = {1:02X}'.format(low_byte, high_byte))
        return (high_byte << 8) | low_byte

    def mpsse_gpio(self):
        """Return command to update the MPSSE GPIO state to the current direction
        and level.
        """
        level_low  = (self._level & 0xFF)
        level_high = ((self._level >> 8) & 0xFF)
        dir_low  = (self._direction & 0xFF)
        dir_high = ((self._direction >> 8) & 0xFF)
        return bytes(bytearray((0x80, level_low, dir_low, 0x82, level_high, dir_high)))

    def mpsse_write_gpio(self):
        """Write the current MPSSE GPIO state to the FT232H chip."""
        self._write(self.mpsse_gpio())

    def get_i2c_device(self, address, **kwargs):
        """Return an I2CDevice instance using this FT232H object and the provided
        I2C address.  Meant to be passed as the i2c_provider parameter to objects
        which use the Adafruit_Python_GPIO library for I2C.
        """
        return I2CDevice(self, address, **kwargs)

    # GPIO functions below:

    def _setup_pin(self, pin, mode):
        if pin < 0 or pin > 15:
            raise ValueError('Pin must be between 0 and 15 (inclusive).')
        if mode not in (IN, OUT):
            raise ValueError('Mode must be IN or OUT.')
        if mode == IN:
            # Set the direction and level of the pin to 0.
            self._direction &= ~(1 << pin) & 0xFFFF
            self._level     &= ~(1 << pin) & 0xFFFF
        else:
            # Set the direction of the pin to 1.
            self._direction |= (1 << pin) & 0xFFFF

    def setup(self, pin, mode):
        """Set the input or output mode for a specified pin.  Mode should be
        either OUT or IN."""
        self._setup_pin(pin, mode)
        self.mpsse_write_gpio()

    def setup_pins(self, pins, values={}, write=True):
        """Setup multiple pins as inputs or outputs at once.  Pins should be a
        dict of pin name to pin mode (IN or OUT).  Optional starting values of
        pins can be provided in the values dict (with pin name to pin value).
        """
        # General implementation that can be improved by subclasses.
        for pin, mode in iter(pins.items()):
            self._setup_pin(pin, mode)
        for pin, value in iter(values.items()):
            self._output_pin(pin, value)
        if write:
            self.mpsse_write_gpio()

    def _output_pin(self, pin, value):
        if value:
            self._level |= (1 << pin) & 0xFFFF
        else:
            self._level &= ~(1 << pin) & 0xFFFF

    def output(self, pin, value):
        """Set the specified pin the provided high/low value.  Value should be
        either HIGH/LOW or a boolean (true = high)."""
        if pin < 0 or pin > 15:
            raise ValueError('Pin must be between 0 and 15 (inclusive).')
        self._output_pin(pin, value)
        self.mpsse_write_gpio()

    def output_pins(self, pins, write=True):
        """Set multiple pins high or low at once.  Pins should be a dict of pin
        name to pin value (HIGH/True for 1, LOW/False for 0).  All provided pins
        will be set to the given values.
        """
        for pin, value in iter(pins.items()):
            self._output_pin(pin, value)
        if write:
            self.mpsse_write_gpio()

    def input(self, pin):
        """Read the specified pin and return HIGH/true if the pin is pulled high,
        or LOW/false if pulled low."""
        return self.input_pins([pin])[0]

    def input_pins(self, pins):
        """Read multiple pins specified in the given list and return list of pin values
        HIGH/True if the pin is pulled high, or LOW/False if pulled low."""
        if [pin for pin in pins if pin < 0 or pin > 15]:
            raise ValueError('Pin must be between 0 and 15 (inclusive).')
        _pins = self.mpsse_read_gpio()
        return [((_pins >> pin) & 0x0001) == 1 for pin in pins]


    def LED(self, pin, turn_on = False):
        # turn an LED on or off
        if pin not in self.pins_in_use:
            self.setup(pin, OUT) 

        self.output(pin, HIGH if turn_on else LOW)






################################################################################
### GPIO Function                                                            ###
################################################################################

OUT     = 0
IN      = 1
HIGH    = True
LOW     = False


def TestGPIO(pin_nr = 7):
    ft1 = FT232H(serial = FindDevices()[0])
    # pin_nr = 7 # D7
    # pin_nr = 8 # C0
    ft1.setup(pin_nr, OUT)

    TI.sleep(1)
    for _ in range(2):
        ft1.output(pin_nr, HIGH)
        # Sleep for 1 second.
        TI.sleep(1)
        # Set pin D7 to a low level so the LED turns off.
        ft1.output(pin_nr, LOW)

        TI.sleep(1)


    ft1.close()


def TestGPIOInput(pin_nr = 3):
    ft1 = FT232H(serial = FindDevices()[0])
    ft1.setup(pin_nr, IN)

    TI.sleep(0.01)
    t0 = TI.time()
    status = False
    print (ft1.input(pin_nr))
    while True:
        new = ft1.input(pin_nr)
        if status == new:
            TI.sleep(0.000001)
            if TI.time() - t0 > 10:
                break
        else:
            status = new
            print (status, TI.time()-t0)
        


    ft1.close()


################################################################################
### Trigger Logger = Trogger                                                 ###
################################################################################

class Ligger(TH.Thread):
    # trigger listener

    def __init__(self, output_file, header = None):
        super(Ligger, self).__init__()
        self.data_queue = QU.Queue()
        self.output_file = output_file

        if header is not None:
            self.PrintFiles(header)

        # self.counter = 0

    def PrintFiles(self, result):
        # self.counter += 1
        # print ("writing", self.counter)
        # print (">", result)
        print(result, file = self.output_file)

    def run(self):

        self.running = True
        while self.running:
            if not self.data_queue.empty():
                self.PrintFiles(self.data_queue.get())
                continue
            TI.sleep(1e-8)
        # print ("stopped ligger")

    def Stop(self):
        while not self.data_queue.empty():
            self.PrintFiles(self.data_queue.get())
        # print ("stopped.")
        self.running = False


    def join(self, *args, **kwargs):
        # print (self.data_queue)
        # self.data_queue.join()
        super(Ligger, self).join(*args, **kwargs)
        # print ('all done')


class Shouter(TH.Thread):
    # A Logged Producer Process
    def __init__(self, label, data_queue, autostart = False):
        super(Shouter, self).__init__()
        # self.in_queue = in_queue
        self.data_queue = data_queue
        self.label = label

        self.setDaemon(True)

        if autostart:
            self.start()

    def run(self):
        self.running = True
        while self.running:
            self.Process()
            TI.sleep(1e-8)
        # print ("stopped %s" % str(self.label))

    def Stop(self):
        self.running = False

    def Process(self):
        # Do the processing job here
        dice = RND.uniform(0.,1.,1)
        if dice > 0.999:
            self.data_queue.put( "%i;%.3f;%f" % (self.label, dice, TI.time()) )


class LoggedPin(Shouter):
    def __init__(self, ft_breakout, pin_nr, *args, **kwargs):
        self.ft_breakout = ft_breakout
        self.pin_nr = pin_nr
        # kwargs['label'] = str(pin_nr)
        self.ft_breakout.setup(self.pin_nr, IN)
        self.status = self.ft_breakout.input(self.pin_nr)

        super(LoggedPin, self).__init__(*args, **kwargs)


    def Process(self):
        # Do the processing job here
        t = TI.time()
        new_status = self.ft_breakout.input(self.pin_nr)
        if new_status != self.status:
            self.status = new_status
            self.data_queue.put( "%i;%s;%f" % (self.pin_nr, str(self.status), t) )


class Trogger(object):
    # A Trigger-Logger

    def __init__(self, pins = [], serial = None, storage_file = None):

        if serial is None:
            self.ft_breakout = FT232H(serial = FindDevices()[0])
        else:
            self.ft_breakout = FT232H(serial = serial)

        if storage_file is None:
            storage_file = "%s_log.txt" % (TI.strftime('%Y%m%d'))

        # spawn threads to print
        self.archivar = Ligger( open(storage_file, 'a') \
                              , header = '#________\npin;changes_to;time_%s' % (TI.strftime('%Y%m%d')) \
                              )
        self.archivar.setDaemon(True)
        self.archivar.start()

        # spawn threads to process

        self.logged_pins = [LoggedPin(self.ft_breakout, pin, data_queue = self.archivar.data_queue, autostart = True, label = str(pin)) for pin in pins]

        for lpin in self.logged_pins:
            lpin.data_queue.put( "%i;%s;%f" % (lpin.pin_nr, str(lpin.status), TI.time()) )

        EXIT.register(self.Quit)

    def Log(self):
        # TI.sleep(2)
        while True:
            TI.sleep(1e-2)


    def Quit(self):
        for lpin in self.logged_pins:
            lpin.Stop()
            lpin.join()

        self.archivar.Stop()
        self.archivar.join()

        self.ft_breakout.close()
        print ('safely exited.')


    def __enter__(self):
        # required for context management ("with")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # exiting when in context manager
        self.Quit()



def TestTrogger(pins = []):

    with Trogger(pins) as trog:
        try:
            trog.Log()
        except KeyboardInterrupt as ki:
            trog.Quit()







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



    def PrepareAnalogAcquisition(self):
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
    # def Quit(self, confirm = True):
    #     # safely exit
    #     if not self.alive:
    #         SYS.exit()
    #         return

    #     if confirm:
    #         if not TK.messagebox.askyesno("Quit","really quit?"):
    #             return

    #     if self.daq_device:
    #         # Stop the acquisition if it is still running.
    #         if not self.DAQIsIdle():
    #             self.analog_input.scan_stop()
    #         if self.daq_device.is_connected():
    #             self.daq_device.disconnect()
    #         self.daq_device.release()
    #     print('safely exited %s.' % (str(self)))
    #     self.alive = False

    #     SYS.exit()



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

        SYS.exit()


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

        # setup for analog input
        self.CountChannels(n_channels, channel_labels)
        super(AnalogInput, self).__init__(*args, **kwargs)
        self.PrepareAnalogAcquisition()

        self.alive = True


#______________________________________________________________________
# I/O
#______________________________________________________________________

    def Record(self, wait_time = None):
        self.times['start'] = TI.time()
        self.rate = self.analog_input.a_in_scan(**self.recording_settings)

        if wait_time is None:
            self.Wait()
        else:
            TI.sleep( wait_time )


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
    forceplate_settings = {'amti': {}, 'joystick': {}, 'kistler': {}, 'dualkistler': {}, 'dualkistler2': {}}
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
            self.playing = False

    def PreparePlot(self):

        self.fig, self.ax = MPP.subplots(1, 1)

        self.ax.set_ylim(-5., 5.)
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
        self.PrepareAnalogAcquisition()
        self.Record()
        # while (TI.time() - t0) <= 2.5:
        #     TI.sleep(0.25)
        #     print (self.GetStatus(), NP.round(TI.time() - t0, 2), 's')

        # print ('blub')
        self.playing = True
        t0 = TI.time()
        dt = TI.time() - t0
        while self.playing:
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
            #     self.playing = False

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
class MultiTrigger(object):
    # a connection to a digital input on a daq device

    def __init__(self, digital_io, port, pin_nr = [0], rising = True):

        self.digital_io = digital_io
        self.port = port
        self.pin_nr = pin_nr
        self.rising = rising



    def Read(self):
        # read out the trigger bit
        return NP.array([self.digital_io.d_bit_in(self.port, pin) for pin in self.pin_nr], dtype = bool)


    def Await(self, scan_frq = 1e3):
        # wait until the trigger bit encounters a rising/falling edge.

        ### triggering loop
        # store previous status
        previous = self.Read()
        triggered_bit = None
        try:
            # first, wait for baseline condition (FALSE on rising, TRUE on falling edge)
            if NP.any(previous == self.rising):
                # wait for a single switch
                while True:
                    TI.sleep(1/scan_frq)
                    current = self.Read()
                    if NP.any(NP.logical_xor(current, previous)):
                        previous = current
                        break
                    previous = current

            # loop until bit changes
            while True:
                TI.sleep(1/scan_frq)
                current = self.Read()
                if NP.any(NP.logical_xor(current, previous)):
                    triggered_bit = self.pin_nr[NP.argmax(NP.logical_xor(current, previous))]
                    break
                previous = current

        except KeyboardInterrupt as ki:
            raise ki

        # trigger was successful
        return triggered_bit

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
### MCC USB1608G Force Plate for Sound of Science                            ###
################################################################################
class DAQForcePlateSoS(AnalogInput):

#______________________________________________________________________
# Construction
#______________________________________________________________________
    def __init__(self, fp_type, pins, baseline_duration = 0 \
                , *args, **kwargs):
        self.fp_type = fp_type

        # indicator and trigger pins
        self.pins = pins
        self.is_triggered = False
        self.has_indicator = False
        if (self.pins is not None) and (type(self.pins) is dict):
            self.has_indicator = self.pins.get('led', None) is not None
            self.is_triggered = self.pins.get('trigger', None) is not None
            self.is_multitriggered = (self.pins.get('baseline', None) is not None) \
                                    or (self.pins.get('reference', None) is not None)

        # check whether baseline is recorded before actual recording
        if baseline_duration > 0:
            self.prerecord_baseline = True
        self.baseline_duration = baseline_duration

        # stores actual data
        self.times = {}
        self.data = None

        # stores the data of baseline qnd reference (empty) recordings
        self.store = []

        ## initialize DAQ
        kwargs['channel_labels'] = forceplate_settings[self.fp_type]['channel_order']
        super(DAQForcePlateSoS, self).__init__(*args, **kwargs)

        self.SetPins()

        ## connect LED
        if self.has_indicator:
            self.led = LED(   digital_io = self.digital_io \
                            , port = self.port \
                            , pin_nr = self.pins['led']\
                            )
        

        ## connect trigger or multiple triggers
        if self.is_multitriggered:
            triggers = [self.pins['trigger']]
            for trig in ['baseline', 'reference']:
                pin = self.pins.get(trig, None)
                if pin is not None:
                    triggers.append(pin)
            print (triggers)
            self.pins['triggers'] = triggers

            self.trigger = MultiTrigger( \
                                      digital_io = self.digital_io \
                                    , port = self.port \
                                    , pin_nr = triggers \
                                    , rising = True \
                                  )
        elif self.is_triggered:
            self.trigger = MultiTrigger( \
                                      digital_io = self.digital_io \
                                    , port = self.port \
                                    , pin_nr = [self.pins['trigger']] \
                                    , rising = True \
                                  )


    def SetPins(self):
        # set pin to input
        if self.has_indicator:
            # set pin to output
            self.digital_io.d_config_bit(self.port, self.pins['led'], UL.DigitalDirection.OUTPUT)
        if self.is_triggered:
            self.digital_io.d_config_bit(self.port, self.pins['trigger'], UL.DigitalDirection.INPUT)


#______________________________________________________________________
# Control
#______________________________________________________________________

    def Indicate(self, value):
        # flash an LED
        if not self.has_indicator:
            return

        self.led.Switch(value)


    def AwaitTrigger(self):
        return self.trigger.Await(scan_frq = self.scan_frq)
    

#______________________________________________________________________
# I/O
#______________________________________________________________________
    def TriggeredRecording(self):

        if self.is_triggered or self.is_multitriggered:
            triggered_bit = None
            while triggered_bit is None:
                # wait for a trigger

                print ('waiting for triggers on channels %s.' % (str(self.pins.get('triggers', self.pins['trigger']))))
                triggered_bit = self.AwaitTrigger()

                # triggering failed
                if triggered_bit is None:
                    return

                # take a baseline recording
                if triggered_bit == self.pins.get('baseline', -1):
                    self.BaselineRecording()

                    # exit if only baseline is recorded
                    if not self.prerecord_baseline:
                        return

                    # optionally repeat triggering
                    triggered_bit = None

                elif triggered_bit == self.pins.get('reference', -1):
                    print ('reference recording...', end = '')
                    self.ReferenceRecording()
                    triggered_bit = None

                else:
                    # a recording bit has been hit, take real recording
                    break


            print ('triggered on DAQ channel %i. ' % (triggered_bit), end = '') 
        print ('recording...' , end = '')
        self.Record()
        print ('done! ')

        # store data
        times, data = self.RetrieveOutput(verbose = False)
        self.store.append([times, data])

        print ('Awaiting reference recording (empty force plate)...', end = '')
        triggered_bit = self.AwaitTrigger()
        if triggered_bit:
            self.ReferenceRecording()
        print ('all done!')


    def RetrieveOutput(self, reference = False, baseline = False, *args, **kwargs):
        times, data = super(DAQForcePlateSoS, self).RetrieveOutput()
        times = times.copy()
        data = data.copy()

        data['reference'] = reference
        data['baseline'] = baseline

        return times, data

    def CombinedOutput(self):

        sync_data = {'time': [], 'type': []}
        force_data = []
        for rec_nr, (times, data) in enumerate(self.store):
            for tp, tt in times.items():
                sync_data['time'].append(tt)
                sync_data['type'].append( tp + "_trigger%i" % (rec_nr) )

            data['nr'] = rec_nr
            force_data.append(data)
        sync_data = PD.DataFrame.from_dict(sync_data)
        force_data = PD.concat(force_data, axis = 0)

        return sync_data, force_data

    def Record(self):

        self.Indicate(True)

        self.times['start'] = TI.time()
        self.rate = self.analog_input.a_in_scan(**self.recording_settings)

        while not self.DAQIsIdle():
            TI.sleep(1/self.scan_frq)

        self.times['stop'] = TI.time()
        self.Indicate(False)



    def ReferenceRecording(self):

        # set recording duration to baseline duration
        standard_duration = self.recording_duration
        self.recording_duration = 1. # s
        self.PrepareAnalogAcquisition()
        good_old_times = self.times.copy()
        good_old_data = None
        if self.data is not None:
            good_old_data = self.data.copy()
        self.times = {}
        self.data = None


        # Record
        self.Indicate(True)

        self.times['start'] = TI.time()
        self.rate = self.analog_input.a_in_scan(**self.recording_settings)

        while not self.DAQIsIdle():
            TI.sleep(1/self.scan_frq)

        self.times['stop'] = TI.time()
        self.Indicate(False)

        # store data
        times, data = self.RetrieveOutput(reference = True, verbose = False)
        self.store.append([times, data])

        # revert recording duration
        self.recording_duration = standard_duration
        self.PrepareAnalogAcquisition()
        self.times = good_old_times
        self.data = good_old_data

        print ('done!')
        return



    def BaselineRecording(self):

        if self.baseline_duration <= 0: 
            raise Exception('please set a baseline duration > 0!')
            return

        print ('baseline recording...', end = '')
        # set recording duration to baseline duration
        standard_duration = self.recording_duration
        self.recording_duration = self.baseline_duration # s

        self.PrepareAnalogAcquisition()

        # Record
        self.Indicate(True)

        self.times['start'] = TI.time()
        self.rate = self.analog_input.a_in_scan(**self.recording_settings)

        while not self.DAQIsIdle():
            TI.sleep(1/self.scan_frq)

        self.times['stop'] = TI.time()
        self.Indicate(False)

        # store data
        times, data = self.RetrieveOutput(baseline = True, verbose = False)
        self.store.append([times, data])

        # revert recording duration
        self.recording_duration = standard_duration
        self.PrepareAnalogAcquisition()
        self.times = {}
        self.data = None

        print ('done!')
        return




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

        # # store data
        # times, data = self.RetrieveOutput()
        # self.store.append([times, data])



    def TriggeredRecording(self):

        # start recording in the background (will wait for trigger)
        self.rate = self.analog_input.a_in_scan(**self.recording_settings)

        self.sync.append([TI.time(), -1] )

        # wait until force plate records
        while self.GetDAQStatus()[1].current_scan_count == 0:
            TI.sleep(1/self.scan_frq)

        # store start time
        # self.sync.append([TI.time(), -self.GetDAQStatus()[1].current_total_count] )
        # turn LED on
        self.Indicate(True)

        # wait until recording has ended
        self.StdOut('recording... ', ' '*20, end = '\r')
        counter = 0
        while not self.DAQIsIdle():
            if (counter % 1000) == 0:
                self.sync.append([TI.time(), self.GetDAQStatus()[1].current_scan_count] )

            TI.sleep(1/self.scan_frq)
            counter += 1

        # store stop time
        self.sync.append([TI.time(), -1] )
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
### I2C Devices                                                              ###
################################################################################
class I2CDevice(object):
    """Class for communicating with an I2C device using the smbus library.
    Allows reading and writing 8-bit, 16-bit, and byte array values to registers
    on the device."""
    # Note that most of the functions in this code are adapted from this app note:
    #  http://www.ftdichip.com/Support/Documents/AppNotes/AN_255_USB%20to%20I2C%20Example%20using%20the%20FT232H%20and%20FT201X%20devices.pdf
    def __init__(self, ft232h, address, multiplexer_channel = None, clock_hz = 1e6):
        """Create an instance of the I2C device at the specified address on the
        specified I2C bus number."""
        self._address = address
        self._ft232h = ft232h
        # Enable clock with three phases for I2C.
        self._ft232h.mpsse_set_clock(clock_hz, three_phase=True)
        # Enable drive-zero mode to drive outputs low on 0 and tri-state on 1.
        # This matches the protocol for I2C communication so multiple devices can
        # share the I2C bus.
        self._ft232h._write(b'\x9E\x07\x00')
        self._idle()


        # connect multiplexer
        self.multiplex = False
        if multiplexer_channel is not None:
            self.multiplex = True
            self.multiplexer_channel = multiplexer_channel
        
        self.Route()



    def Route(self):
        # distribute signal via multiplexer
        if self.multiplex:
            self.master.multiplexer.Select(self.multiplexer_channel)



    # simplified read/write (FM)
    def __getitem__(self, register):
        return ReadUnsigned(self, register)

    # default functions
    def _idle(self):
        """Put I2C lines into idle state."""
        # Put the I2C lines into an idle state with SCL and SDA high.
        self._ft232h.setup_pins({0: OUT, 1: OUT, 2: IN},
                                {0: HIGH, 1: HIGH})

    def _transaction_start(self):
        """Start I2C transaction."""
        # Clear command buffer and expected response bytes.
        self._command = []
        self._expected = 0

    def _transaction_end(self):
        """End I2C transaction and get response bytes, including ACKs."""
        # Ask to return response bytes immediately.
        # print (self._command[:2])
        self._command.append(b'\x87')
        # print(bytearray(self._command[:2]))
        # Send the entire command to the MPSSE.
        self._ft232h._write(b"".join(self._command))
        # Read response bytes and return them.
        return bytearray(self._ft232h._poll_read(self._expected))

    def _i2c_start(self):
        """Send I2C start signal. Must be called within a transaction start/end.
        """
        # Set SCL high and SDA low, repeat 4 times to stay in this state for a
        # short period of TI.
        self._ft232h.output_pins({0: HIGH, 1: LOW}, write=False)
        self._command.append(self._ft232h.mpsse_gpio() * _REPEAT_DELAY)
        # Now drop SCL to low (again repeat 4 times for short delay).
        self._ft232h.output_pins({0: LOW, 1: LOW}, write=False)
        self._command.append(self._ft232h.mpsse_gpio() * _REPEAT_DELAY)

    def _i2c_idle(self):
        """Set I2C signals to idle state with SCL and SDA at a high value. Must
        be called within a transaction start/end.
        """
        self._ft232h.output_pins({0: HIGH, 1: HIGH}, write=False)
        self._command.append(self._ft232h.mpsse_gpio() * _REPEAT_DELAY)

    def _i2c_stop(self):
        """Send I2C stop signal. Must be called within a transaction start/end.
        """
        # Set SCL low and SDA low for a short period.
        self._ft232h.output_pins({0: LOW, 1: LOW}, write=False)
        self._command.append(self._ft232h.mpsse_gpio() * _REPEAT_DELAY)
        # Set SCL high and SDA low for a short period.
        self._ft232h.output_pins({0: HIGH, 1: LOW}, write=False)
        self._command.append(self._ft232h.mpsse_gpio() * _REPEAT_DELAY)
        # Finally set SCL high and SDA high for a short period.
        self._ft232h.output_pins({0: HIGH, 1: HIGH}, write=False)
        self._command.append(self._ft232h.mpsse_gpio() * _REPEAT_DELAY)

    def _i2c_read_bytes(self, length=1):
        """Read the specified number of bytes from the I2C bus.  Length is the
        number of bytes to read (must be 1 or more).
        """
        for i in range(length-1):
            # Read a byte and send ACK.
            self._command.append(b'\x20\x00\x00\x13\x00\x00')
            # Make sure pins are back in idle state with clock low and data high.
            self._ft232h.output_pins({0: LOW, 1: HIGH}, write=False)
            self._command.append(self._ft232h.mpsse_gpio())
        # Read last byte and send NAK.
        self._command.append(b'\x20\x00\x00\x13\x00\xFF')
        # Make sure pins are back in idle state with clock low and data high.
        self._ft232h.output_pins({0: LOW, 1: HIGH}, write=False)
        self._command.append(self._ft232h.mpsse_gpio())
        # Increase expected number of bytes.
        self._expected += length

    def _i2c_write_bytes(self, data):
        # print ('_i2c_write_bytes', data)
        """Write the specified number of bytes to the chip."""
        for byte in data:
            # Write byte.
            self._command.append(bytes(bytearray((0x11, 0x00, 0x00, byte))))

            # Make sure pins are back in idle state with clock low and data high.
            self._ft232h.output_pins({0: LOW, 1: HIGH}, write=False)
            self._command.append(self._ft232h.mpsse_gpio() * _REPEAT_DELAY)
            # Read bit for ACK/NAK.
            self._command.append(bytes([0x22, 0x00]))
        # Increase expected response bytes.
        self._expected += len(data)

    def _address_byte(self, read=True):
        """Return the address byte with the specified R/W bit set.  If read is
        True the R/W bit will be 1, otherwise the R/W bit will be 0.
        """
        if read:
            return (self._address << 1) | 0x01
        else:
            return self._address << 1

    def _verify_acks(self, response):
        """Check all the specified bytes have the ACK bit set.  Throws a
        RuntimeError exception if not all the ACKs are set.
        """
        # for byte in response:
        #     print (byte, byte & 0x01)
        for byte in response:
            if byte & 0x01 != 0x00:
                raise IOError('Failed to find expected I2C ACK!')

    def ping(self):
        """Attempt to detect if a device at this address is present on the I2C
        bus.  Will send out the device's address for writing and verify an ACK
        is received.  Returns true if the ACK is received, and false if not.
        """
        self._idle()
        self._transaction_start()
        self._i2c_start()
        self._i2c_write_bytes([self._address_byte(False)])
        self._i2c_stop()
        response = self._transaction_end()
        if len(response) != 1:
            raise RuntimeError('Expected 1 response byte but received {0} byte(s).'.format(len(response)))
        return ((response[0] & 0x01) == 0x00)

    def writeRaw8(self, value):
        """Write an 8-bit value on the bus (without register)."""
        value = value & 0xFF
        self._idle()
        self._transaction_start()
        self._i2c_start()
        self._i2c_write_bytes([self._address_byte(False), value])
        self._i2c_stop()
        response = self._transaction_end()
        self._verify_acks(response)

    def write8(self, register, value):
        """Write an 8-bit value to the specified register."""
        value = value & 0xFF
        self._idle()
        self._transaction_start()
        self._i2c_start()
        self._i2c_write_bytes([self._address_byte(False), register, value])
        self._i2c_stop()
        response = self._transaction_end()
        # self._verify_acks(response)

        verify = False
        while not verify:
            try:
                self._verify_acks(response)
                verify = True
            except IOError:
                # print (err)
                verify = False


    def write16(self, register, value, little_endian=True):
        """Write a 16-bit value to the specified register."""
        value = value & 0xFFFF
        value_low  = value & 0xFF
        value_high = (value >> 8) & 0xFF
        if not little_endian:
            value_low, value_high = value_high, value_low
        self._idle()
        self._transaction_start()
        self._i2c_start()
        self._i2c_write_bytes([self._address_byte(False), register, value_low,
                                value_high])
        self._i2c_stop()
        response = self._transaction_end()
        self._verify_acks(response)

    def writeList(self, register, data):
        """Write bytes to the specified register."""
        self._idle()
        self._transaction_start()
        self._i2c_start()
        self._i2c_write_bytes([self._address_byte(False), register] + data)
        self._i2c_stop()
        response = self._transaction_end()
        self._verify_acks(response)

    def readList(self, register, length):
        """Read a length number of bytes from the specified register.  Results
        will be returned as a bytearray."""
        # print ("begin to read list.")
        # print (self._expected)
        # print (self._command)

        if length <= 0:
            raise ValueError("Length must be at least 1 byte.")
        self._idle()
        self._transaction_start()
        self._i2c_start()
        self._i2c_write_bytes([self._address_byte(True), register])
        self._i2c_stop()
        self._i2c_idle()
        self._i2c_start()
        self._i2c_read_bytes(length)
        self._i2c_stop()
        response = self._transaction_end()
        # print (length, response[-length:])
        self._verify_acks(response[-length:])
        return response[-length:]

    def readRaw8(self):
        """Read an 8-bit value on the bus (without register)."""
        self._idle()
        self._transaction_start()
        self._i2c_start()
        self._i2c_write_bytes([self._address_byte(False)])
        self._i2c_stop()
        self._i2c_idle()
        self._i2c_start()
        self._i2c_write_bytes([self._address_byte(True)])
        self._i2c_read_bytes(1)
        self._i2c_stop()
        response = self._transaction_end()
        self._verify_acks(response[:-1])
        return response[-1]

    def readU8(self, register):
        """Read an unsigned byte from the specified register."""
        self._idle()
        self._transaction_start()
        self._i2c_start()
        self._i2c_write_bytes([self._address_byte(False), register])
        self._i2c_stop()
        self._i2c_idle()
        self._i2c_start()
        self._i2c_write_bytes([self._address_byte(True)])
        self._i2c_read_bytes(1)
        self._i2c_stop()
        response = self._transaction_end()
        try:
            self._verify_acks(response[:-1])
        except (IOError):
            return None
        return response[-1]

    def InsistU8(self, register):
        """Read an unsigned byte from the specified register."""
        """... again and again if ACKs crash"""
        response = None
        while response is None:
            try:
                response = self.readU8(register)
            except (IOError):
                print ('acks failed')
                pass
        return response


    def readS8(self, register):
        """Read a signed byte from the specified register."""
        result = self.readU8(register)
        if result > 127:
            result -= 256
        return result

    def readU16(self, register, little_endian=True):
        """Read an unsigned 16-bit value from the specified register, with the
        specified endianness (default little endian, or least significant byte
        first)."""
        self._idle()
        self._transaction_start()
        self._i2c_start()
        self._i2c_write_bytes([self._address_byte(False), register])
        self._i2c_stop()
        self._i2c_idle()
        self._i2c_start()
        self._i2c_write_bytes([self._address_byte(True)])
        self._i2c_read_bytes(2)
        self._i2c_stop()
        response = self._transaction_end()
        self._verify_acks(response[:-2])
        if little_endian:
            return (response[-1] << 8) | response[-2]
        else:
            return (response[-2] << 8) | response[-1]

    def readS16(self, register, little_endian=True):
        """Read a signed 16-bit value from the specified register, with the
        specified endianness (default little endian, or least significant byte
        first)."""
        result = self.readU16(register, little_endian)
        if result > 32767:
            result -= 65536
        return result

    def readU16LE(self, register):
        """Read an unsigned 16-bit value from the specified register, in little
        endian byte order."""
        return self.readU16(register, little_endian=True)

    def readU16BE(self, register):
        """Read an unsigned 16-bit value from the specified register, in big
        endian byte order."""
        return self.readU16(register, little_endian=False)

    def readS16LE(self, register):
        """Read a signed 16-bit value from the specified register, in little
        endian byte order."""
        return self.readS16(register, little_endian=True)

    def readS16BE(self, register):
        """Read a signed 16-bit value from the specified register, in big
        endian byte order."""
        return self.readS16(register, little_endian=False)



################################################################################
### Bit Ragging                                                              ###
################################################################################
def ReadUnsigned(device, register):
    data = (device.InsistU8(register) << 8) + device.InsistU8(register+1)
    if (data & (1 << 16 - 1)):
        data = data - (1<<16)

    data /= 32768.
    return data

def ReadSigned(device, register):
    data = (device.readS8(register) << 8) + device.readS8(register+1)
    data /= 32768.
    return data


# produce a byte block
def FloatsToBytearray(values):
    byte = NP.array(values*127, dtype = NP.int8)
    byte[byte < 0] + 256
    return byte.tobytes()

def ByteArrayToInts(byte_array):
    values = NP.frombuffer(byte_array, dtype = NP.int8)
    values[values > 127] - 256
    return values


def ByteReadoutConversion(read):
    n_measures = len(read) // 2
    values = []
    for m in range(n_measures):
        # x = read[2*m] | (read[2*m+1] << 8)
        x = read[2*m+1] | (read[2*m] << 8)
        if(x & (1 << 16 - 1)):
            x = x - (1<<16)
        values.append(x)

    return values


def ByteArrayToCoordinates(byte_array):
    values = ByteReadoutConversion(byte_array)
    # values = NP.frombuffer(byte_array, dtype = NP.int8)
    # values[values > 127] - 256
    return values



################################################################################
### Multiplexer                                                              ###
################################################################################
class TCA9548A(I2CDevice):
    def __init__(self, master, address = None, clock_hz = 1e6):

        self.master = master
        self.defaults = {}
        self.defaults['address'] = 0x70

        if address is None:
            self.address = self.defaults['address']
        else:
            self.address = address

        print ('setting up TCA9548A multiplexer, address', format(self.address,'x'))
        super(TCA9548A, self).__init__(self.master, self.address, clock_hz = clock_hz)


    def Select(self, channel):
        self.writeRaw8(1 << channel)




################################################################################
### Buffered I2C Devices                                                     ###
################################################################################
class DeviceBufferLoader(DEQue):

    def __init__(self, device, generating_rate = 10, max_length = None):
        self.t_start = TI.time()
        self.generating_rate = generating_rate # Hz
        self.running = False
        self._fill = None # thread
        self.max_length = max_length

        super(DeviceBufferLoader, self).__init__(maxlen = self.max_length)

        self.device = device


    def Loop(self):
        self.device.master.LED(7, True)
        while self.running: 
            data = self.device.GetData()
            self.append( [TI.time(), data] )
            TI.sleep(1/self.generating_rate)

        self.device.master.LED(7, False)



    def Start(self):
        self.running = True
        self._fill = TH.Thread(target = self.Loop)
        self._fill.daemon = True
        self._fill.start()


    def Stop( self ):
        self.running = False
        if self._fill:
            self._fill.join()
        self._fill = None



    def RetrieveOutput(self):
        # get the output
        time = []
        data_out = []
        while len(self) > 0:
            t, read = self.popleft()
            time.append(t)
            data_out.append( ByteArrayToCoordinates(read) )

        data_out = NP.array(data_out) * self.device.config.get('unit_scale', 1)

        return NP.array(time), data_out






################################################################################
### NXP Precision 9DOF                                                       ###
################################################################################
# https://www.adafruit.com/product/3463
# https://learn.adafruit.com/nxp-precision-9dof-breakout/overview

# FXOS8700 3-Axis accelerometer and magnetometer, and the FXAS21002 3-axis gyroscope
# https://github.com/adafruit/Adafruit_FXOS8700
# https://github.com/adafruit/Adafruit_FXAS21002C    




#______________________________________________________________________
# accelerometer and magnetometer
#______________________________________________________________________
fxos8700_registers = {}
fxos8700_config = {}
fxos8700_defaults = {}

fxos8700_defaults['address'] = 0x1F
fxos8700_config['ID'] = 0xC7

fxos8700_config['scale'] = {}
# fxos8700_config['scale']['accel'] = {'2g': 0.061, '4g': 0.122, '8g': 0.244, '16g': 0.732} # milli-g
fxos8700_config['scale']['mag'] = {'2g': 0.000244, '4g': 0.000488, '8g': 0.000976, 'ut': 0.1} # dps
# fxos8700_config['scale']['temp'] = 8 # +/- 8 bit = 1deg Celsius


fxos8700_registers['identity'] = 0x0D

fxos8700_config['sensitivity'] = {'accel': {}, 'mag': {}}
fxos8700_config['sensitivity']['accel']['2g'] = 0x00
fxos8700_config['sensitivity']['accel']['4g'] = 0x01
fxos8700_config['sensitivity']['accel']['8g'] = 0x02


fxos8700_defaults['sensitivity'] = '8g'

fxos8700_registers['control'] = {}
fxos8700_registers['control']['mag'] = [0x5B, 0x5C, 0x5D]  
fxos8700_registers['control']['accel'] = [0x2A, 0x2B, 0x2C, 0x2D, 0x2E] 

fxos8700_registers['status'] = {'accel': 0x00, 'mag': 0x32}

fxos8700_registers['data_cfg'] =  0x0E
fxos8700_registers['data'] = {'accel': {}, 'mag': {}, 'temp': {}}
fxos8700_registers['data']['accel'] = {'x': 0x01, 'y': 0x03, 'z': 0x05}
fxos8700_registers['data']['mag'] = {'x': 0x33, 'y': 0x35, 'z': 0x37} 
fxos8700_registers['data']['temp'] = {'x': 0x51} 

fxos8700_config['columns'] = ['a_x', 'a_y', 'a_z', 'm_x', 'm_y', 'm_z']


class FXOS8700(I2CDevice):

    def __init__(self, master, address = None, accel_sensitivity = None, multiplexer_channel = None, clock_hz = 1e6):
        # accel_sensitivity: {2g, 4g, 8g, 16g}

        self.master = master

        self.registers = fxos8700_registers
        self.config = fxos8700_config
        self.defaults = fxos8700_defaults

        if address is None:
            self.address = self.defaults['address']
        else:
            self.address = address

        super(FXOS8700, self).__init__(master, self.address, multiplexer_channel, clock_hz)

        self.write8(self.registers['control']['accel'][0], 0x00)     # (Standby)
        # data = [self.InsistU8(0x01+reg) for reg in range(64)]
        # print (data)

        # Configure the gyroscope
        self.write8(self.registers['data_cfg'], self.config['sensitivity']['accel'][self.defaults['sensitivity']])
        # self.scale = self.config['scale']['mag'][self.defaults['sensitivity']]

        # High resolution
        # self.write8(self.registers['control']['accel'][1], 0x02)
        # normal resolution
        self.write8(self.registers['control']['accel'][1], 0x00)
        # Active, Normal Mode, Low Noise, 100Hz in Hybrid Mode
        # self.write8(self.registers['control']['accel'][0], 0x15) # 200 Hz
        # different mode (higher ODR)
        self.write8(self.registers['control']['accel'][0], 0x0D) # 400 Hz
        # self.write8(self.registers['control']['accel'][0], 0x05) # 800 Hz

        # Configure the magnetometer
        # Hybrid Mode, Over Sampling Rate = 16
        self.write8(self.registers['control']['mag'][0], 0x1F)
        # Jump to reg 0x33 after reading 0x06
        self.write8(self.registers['control']['mag'][1], 0x20) 


        self.data_registers = [ self.registers['data'][device]['x']+hilo \
                      for device in ['accel', 'mag'] \
                      for hilo in range(6) \
                    ] 


    def readU8(self, register):
        """Read an unsigned byte from the specified register."""
        self._idle()
        self._transaction_start()
        self._i2c_start()
        self._i2c_write_bytes([self._address_byte(False), register])
        # self._i2c_stop()
        self._i2c_idle()
        self._i2c_start()
        self._i2c_write_bytes([self._address_byte(True)])
        self._i2c_read_bytes(1)
        self._i2c_stop()
        response = self._transaction_end()
        try:
            self._verify_acks(response[:-1])
        except (IOError):
            return None
        return response[-1]


    def GetData(self):
        return bytearray([self.InsistU8(reg) for reg in self.data_registers])



#______________________________________________________________________
# gyroscope
#______________________________________________________________________
fxas21002_registers = {}
fxas21002_registers['status']              = 0x00
fxas21002_registers['identity']            = 0x0C # WHO_AM_I # 11010111   r
fxas21002_registers['data']                = {'x': 0x01, 'y': 0x03, 'z': 0x05}
fxas21002_registers['control']             = [0x0D, 0x13, 0x14] # CTRL_REG0|1|2 # 00000000   r/w


fxas21002_config = {}
fxas21002_defaults = {}

fxas21002_defaults['address']              = 0x21
fxas21002_config['ID']                     = 0xD7       # 1101 0111


fxas21002_config['sensitivity']            = {} # ctrl_reg0
fxas21002_config['sensitivity'][250]       = 0x03
fxas21002_config['sensitivity'][500]       = 0x02
fxas21002_config['sensitivity'][1000]      = 0x01
fxas21002_config['sensitivity'][2000]      = 0x00
fxas21002_defaults['sensitivity']          = 2000


fxas21002_config['scale']                  = {} # Table 35 of datasheet
fxas21002_config['scale'][250]             = 0.0078125
fxas21002_config['scale'][500]             = 0.015625
fxas21002_config['scale'][1000]            = 0.03125
fxas21002_config['scale'][2000]            = 0.0625


fxas21002_config['columns'] = ['g_x', 'g_y', 'g_z']


class FXAS21002(I2CDevice):
    address = None

    def __init__(self, master, address = None, multiplexer_channel = None, clock_hz = 1e6):
        self.master = master

        self.registers = fxas21002_registers
        self.config = fxas21002_config
        self.defaults = fxas21002_defaults
        
        if address is None:
            self.address = self.defaults['address']
        else:
            self.address = address

        super(FXAS21002, self).__init__(master, self.address, multiplexer_channel, clock_hz)        

        # print ('me', hex(self.readU8(fxas21002_registers['identity'])), hex(fxas21002_config['ID']))


        # activate
        self.write8(self.registers['control'][1], 0x00)     # (Standby)
        # self.write8(self.registers['control'][1], (1<<6))   # Reset
        self.write8(self.registers['control'][0], self.config['sensitivity'][self.defaults['sensitivity']]) # Set sensitivity
        # self.write8(self.registers['control'][1], 0x0e) # Active, 100 Hz
        # self.write8(self.registers['control'][1], 0x0a) # Active, 200 Hz
        self.write8(self.registers['control'][1], 0x06) # Active, 400 Hz
        # self.write8(self.registers['control'][1], 0x02) # Active, 800 Hz
        
        self.data_registers = [ self.registers['data']['x']+hilo \
                      for hilo in range(6) \
                    ] 

        TI.sleep(0.1) # 60 ms + 1/ODR



    def readU8(self, register):
        """Read an unsigned byte from the specified register."""
        self._idle()
        self._transaction_start()
        self._i2c_start()
        self._i2c_write_bytes([self._address_byte(False), register])
        # self._i2c_stop()
        self._i2c_idle()
        self._i2c_start()
        self._i2c_write_bytes([self._address_byte(True)])
        self._i2c_read_bytes(1)
        self._i2c_stop()
        response = self._transaction_end()
        try:
            self._verify_acks(response[:-1])
        except (IOError):
            return None
        return response[-1]


    def GetData(self):
        return bytearray([self.InsistU8(reg) for reg in self.data_registers])



#______________________________________________________________________
# combined accelerometer/gyroscope
#______________________________________________________________________
class NXP(object):
    def __init__(self, master = None, multiplexer_channel = None, clock_hz = 1e6):
        if master is None:
            self.master = FT232H(serial = FindDevices()[0]) # , multiplexer_address = 0x70
        else:
            self.master = master

        self.accel = FXOS8700(self.master, multiplexer_channel = multiplexer_channel, clock_hz = clock_hz) # accel
        self.gyro = FXAS21002(self.master, multiplexer_channel = multiplexer_channel, clock_hz = clock_hz) # gyro
        self.multiplex = multiplexer_channel is not None

        self.accel_registers = [ self.accel.registers['data']['accel']['x']+hilo \
                      for hilo in range(6) \
                    ] 
        self.gyro_registers = [ self.gyro.registers['data']['x']+hilo \
                      for hilo in range(6) \
                    ] 
        self.mag_registers = [ self.accel.registers['data']['mag']['x']+hilo \
                      for hilo in range(6) \
                    ] 

        self.config = {'columns': ['a_x', 'a_y', 'a_z', 'g_x', 'g_y', 'g_z', 'm_x', 'm_y', 'm_z']}


    def Route(self):
        self.accel.Route()


    def GetData(self):
        data = bytearray( \
              [self.accel.InsistU8(reg) for reg in self.accel_registers] \
            + [self.gyro.InsistU8(reg) for reg in self.gyro_registers] \
            + [self.accel.InsistU8(reg) for reg in self.mag_registers] \
            )
        return data



#______________________________________________________________________
# multiplexed sensors
#______________________________________________________________________
class MultiNXP(object):
    def __init__(self, master = None, multiplexer_channels = None, clock_hz = 1e6):

        if master is None:
            self.master = FT.FT232H(serial = FT.FindDevices()[0], multiplexer_address = 0x70)
        else:
            self.master = master

        self.multiplex = multiplexer_channels is not None
        self.channels = multiplexer_channels
        self.plexer = self.master.multiplexer
        self.Select = self.plexer.Select

        self.acceleromag = {}
        self.gyroscope = {}
        for ch in self.channels:
            self.acceleromag[ch] = FXOS8700(self.master, multiplexer_channel = ch, clock_hz = clock_hz) # accel
            self.gyroscope[ch] = FXAS21002(self.master, multiplexer_channel = ch, clock_hz = clock_hz) # gyro

        self.accel_registers = [ fxos8700_registers['data']['accel']['x']+hilo \
                      for hilo in range(6) \
                    ] 
        self.gyro_registers = [ fxas21002_registers['data']['x']+hilo \
                      for hilo in range(6) \
                    ] 
        self.mag_registers = [ fxos8700_registers['data']['mag']['x']+hilo \
                      for hilo in range(6) \
                    ] 

        self.config = {'columns': []}
        for ch in self.channels:
            self.config['columns'] += \
                                    [ 'a%i_x' % (ch), 'a%i_y' % (ch), 'a%i_z' % (ch) \
                                    , 'g%i_x' % (ch), 'g%i_y' % (ch), 'g%i_z' % (ch) \
                                    , 'm%i_x' % (ch), 'm%i_y' % (ch), 'm%i_z' % (ch) \
                                    ]


    def Route(self):
        pass

    def GetData(self):
        values = []
        for ch in self.channels:
            self.Select(ch)
            values += [self.acceleromag[ch].InsistU8(reg) for reg in self.accel_registers] \
                    + [self.gyroscope[ch].InsistU8(reg) for reg in self.gyro_registers] \
                    + [self.acceleromag[ch].InsistU8(reg) for reg in self.mag_registers]

        return bytearray(values)




################################################################################
### Single Sensor Test                                                           ###
################################################################################

def PlotSingleSensor(device, data):

    PreparePlot()


    cm = 1./2.54
    figwidth = 16 * cm
    figheight = 12 * cm
    fig = MPP.figure( \
                              figsize = (figwidth, figheight) \
                            , facecolor = None \
                            )
    # MPP.ion() # "interactive mode". Might e useful here, but i don't know. Try to turn it off later.

# define axis spacing
    fig.subplots_adjust( \
                              top    = 0.96 \
                            , right  = 0.98 \
                            , bottom = 0.09 \
                            , left   = 0.10 \
                            , wspace = 0.20 # column spacing \
                            , hspace = 0.20 # row spacing \
                            )

    columns = [0]
    titles = {'a': 'accelerometer', 'g': 'gyroscope', 'm': 'compass'}
    rows = ['a', 'g', 'm'] #
    # roes = [0]
    # print (data.columns)

    # define subplots
    gs = MP.gridspec.GridSpec( \
                                  len(rows) \
                                , len(columns) \
                                , height_ratios = [1]*len(rows) \
                                , width_ratios = [1]*len(columns) \
                                )


    ref_ax = None
    for r, rw in enumerate(rows):
        for c, col in enumerate(columns):

            if ref_ax is None:
                ax = fig.add_subplot(gs[r,c])
                ref_ax = ax
            else:
                ax = fig.add_subplot(gs[r,c], sharex = ref_ax) # , sharey = ref_ax


            # datacols = ["%s_%s" % (col,coord) for coord in coordinates]
            datacols = ["%s_%s" % (rw,coord) for coord in coordinates]

            ax.plot(data.index, data.loc[:, datacols].values )#* constants['g'])
            ax.legend(["$%s$" % (dc) for dc in datacols])


            PolishAx(ax)
            if c == 0:
                ax.set_ylabel(r"\textbf{%s}" % (titles[rw]) ) 
            # if r == 0: 
            #     ax.set_title()
            if r == len(rows)-1: 
                ax.set_xlabel('time (s)')

    MPP.show()


def TestSingleNXP():
    clock_hz = 1.5e6
    ft1 = FT232H(serial = FindDevices()[0], clock_hz = clock_hz) # 

    sr = 10000 # pause rate; less relevant
    recording_duration = 10. # seconds


    device = NXP(ft1, clock_hz = clock_hz) # 0,1,2
    signal = DeviceBufferLoader( device = device, generating_rate = sr ) #, max_length = 512 for post trigger

    data = Record(ft1, signal, recording_duration)
    print ('\t', data.shape[0]/recording_duration, 'Hz sampling rate')

    PlotSingleSensor(device, data)



################################################################################
### Multiplex Test                                                           ###
################################################################################
def PlotMultiplex(device, data):

    PreparePlot()


    cm = 1./2.54
    figwidth = 16 * cm
    figheight = 12 * cm
    fig = MPP.figure( \
                              figsize = (figwidth, figheight) \
                            , facecolor = None \
                            )
    # MPP.ion() # "interactive mode". Might e useful here, but i don't know. Try to turn it off later.

# define axis spacing
    fig.subplots_adjust( \
                              top    = 0.96 \
                            , right  = 0.98 \
                            , bottom = 0.09 \
                            , left   = 0.10 \
                            , wspace = 0.20 # column spacing \
                            , hspace = 0.20 # row spacing \
                            )

    columns = ['a', 'g', 'm']
    titles = {'a': 'accelerometer', 'g': 'gyroscope', 'm': 'compass'}
    rows = device.channels # [0] #
    # roes = [0]

    # define subplots
    gs = MP.gridspec.GridSpec( \
                                  len(rows) \
                                , len(columns) \
                                , height_ratios = [1]*len(rows) \
                                , width_ratios = [1]*len(columns) \
                                )


    ref_ax = None
    for r, rw in enumerate(rows):
        for c, col in enumerate(columns):

            if ref_ax is None:
                ax = fig.add_subplot(gs[r,c])
                ref_ax = ax
            else:
                ax = fig.add_subplot(gs[r,c], sharex = ref_ax) # , sharey = ref_ax


            # datacols = ["%s_%s" % (col,coord) for coord in coordinates]
            datacols = ["%s%i_%s" % (col,rw,coord) for coord in coordinates]

            ax.plot(data.index, data.loc[:, datacols].values )#* constants['g'])
            ax.legend(["$%s$" % (dc) for dc in datacols])


            PolishAx(ax)
            if c == 0:
                ax.set_ylabel('device %i' % (rw) ) 
            if r == 0: 
                ax.set_title(r"\textbf{%s}" % (titles[col]))
            if r == len(rows)-1: 
                ax.set_xlabel('time (s)')

    MPP.show()



def ReadySetGo(ft_unit):
    ft_unit.LED(7, True)
    print ('ready...')
    TI.sleep(.5)
    ft_unit.LED(7, False)
    TI.sleep(.5)

    ft_unit.LED(7, True)
    print ('set...')
    TI.sleep(.5)
    ft_unit.LED(7, False)
    TI.sleep(.5)

    print ('go!')


def Record(ft1, signal, recording_duration):
    ReadySetGo(ft1)
    start_time = TI.time()
    signal.Start()
    
    TI.sleep(recording_duration)
    signal.Stop()
    

    # print (signal)
    time, data = signal.RetrieveOutput()
    time -= start_time
    

    # print (signal, data.shape)
    data = PD.DataFrame(data, index = time, columns = signal.device.config['columns'])
    return data




def TestMultiplexNXP():
    clock_hz = 2.0e6
    ft1 = FT232H(serial = FindDevices()[0], multiplexer_address = 0x70, clock_hz = clock_hz) # 

    sr = 10000 # pause rate; less relevant
    recording_duration = 10. # seconds


    device = MultiNXP(ft1, multiplexer_channels = [0,1,2], clock_hz = clock_hz) # 0,1,2
    signal = DeviceBufferLoader( device = device, generating_rate = sr ) #, max_length = 512 for post trigger

    data = Record(ft1, signal, recording_duration)
    print ('\t', data.shape[0]/recording_duration, 'Hz sampling rate')

    PlotMultiplex(device, data)


################################################################################
### Multiplexed, Buffered NXP Wrapper                                        ###
################################################################################
class BufferedMultiNXP(DeviceBufferLoader):
    def __init__(self, ft_breakout, recording_duration = 1., multiplexer_channels = [], sampling_rate = 1e0, clock_hz = 1e6):

        self.ft_breakout = ft_breakout
        self.recording_duration = recording_duration
        self.multiplexer_channels = multiplexer_channels
        self.sampling_rate = sampling_rate

        self.device = MultiNXP(self.ft_breakout, multiplexer_channels = self.multiplexer_channels, clock_hz = clock_hz)
        self.signal = DeviceBufferLoader( device = self.device, generating_rate = self.sampling_rate ) #, max_length = 512 for post trigger


    def Record(self):
        self.signal.running = True
        self.start_time = TI.time()
        self.signal.Loop()
    

    def Start(self):
        self._executor = TH.Thread(target = self.Record)
        self._executor.daemon = True
        self._executor.start()


    def Stop( self ):
        self.signal.running = False
        if self._executor:
            self._executor.join()
        self._executor = None


    def RetrieveOutput(self, verbose = False):
        time, data = self.signal.RetrieveOutput()
        time -= self.start_time

        if verbose:
            print ('\t', NP.round(data.shape[0]/self.recording_duration, 1), ' Hz on the sensors.')

        return PD.DataFrame(data, index = time, columns = self.signal.device.config['columns'])


    def TestRecording(self):
        self.Start()
        TI.sleep(self.recording_duration)
        self.Stop()
    
        self.data = self.RetrieveOutput()

        PlotMultiplex(self.device, self.data)


def TestBufferedMultiplexedSensors():
    clock_hz = 2.0e6
    ft_breakout = FT232H(serial = FindDevices()[0], multiplexer_address = 0x70, clock_hz = clock_hz)
    sensors = BufferedMultiNXP(   ft_breakout \
                                , recording_duration = 3. \
                                , multiplexer_channels = [0,1,2] \
                                , sampling_rate = 1e6 \
                                , clock_hz = clock_hz \
                                )

    sensors.TestRecording()


################################################################################
### Force Plate Data                                                         ###
################################################################################
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
                , channel_labels = forceplate_settings['kistler']['channel_order'] \
                , scan_frq = 1e6 \
                ) \
        as ai:

        ai.Record()
        times, data = ai.RetrieveOutput()
        PlotKistlerForces(data)




def TestOscilloscope():
    with Oscilloscope( \
                  sampling_rate = 1e3 \
                , channel_labels = forceplate_settings['kistler']['channel_order'] \
                , scan_frq = 1e6 \
                ) \
        as osci:

        osci.Show(window = 6)
        



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
### Balance Score Calculation                                                ###
################################################################################
Euclid = lambda v: NP.sqrt(NP.sum(NP.power(v, 2), axis = 1))



def MakeColormap(groups):
    import matplotlib.colors as MPC
    import matplotlib.cm as CM

    lm = NP.arange(len(groups))
    colormap = CM.ScalarMappable(norm = MPC.Normalize(vmin = lm.min(), vmax = lm.max()), cmap = MPP.get_cmap('Dark2') )
    return {grp: colormap.to_rgba(nr) for nr, grp in enumerate(groups)}


def BalanceScore(forcetrace, test_timepoint = 10., interval = [-1., 4.], ax = None, color = None, offset = None):
    
    # deep copy to avoid changing data
    forcetrace = forcetrace.copy() 
    if 'time' in forcetrace.columns:
        forcetrace.set_index('time', inplace = True)
        forcetrace.index.name = 'time'
    elif ('time' not in forcetrace.index.name):
        raise Exception('time not found in force columns.')

    forcetrace = forcetrace.loc[NP.logical_and(forcetrace.index.values > test_timepoint+interval[0] \
                                                , forcetrace.index.values <= test_timepoint+interval[1]) \
                                , :]
    # print (forcetrace)

    # align to timepoint of interest
    forcetrace.index -= test_timepoint

    # calculate center of pressure
    forcetrace['C_x'] = forcetrace['M_y'] / forcetrace['F_z']
    forcetrace['C_y'] = -forcetrace['M_x'] / forcetrace['F_z']
    center_of_pressure = forcetrace.loc[:, ['C_x', 'C_y']]

    # get relevant interval
    t = center_of_pressure.index.get_level_values('time')
    t_reference = NP.logical_and(t > interval[0], t <= 0)
    t_focus = NP.logical_and(t >  0 , t <= interval[1])

    # center of pressure vectors
    pre_values = center_of_pressure.loc[t_reference, :].values
    post_values = center_of_pressure.loc[t_focus, :].values
    
    if ax is not None:
        ax.plot(t[t_reference], offset + Euclid( pre_values - NP.mean(pre_values, axis = 0).T ) , ls = '-', lw = 1, color = color )
        ax.plot(t[t_focus], offset + Euclid( post_values - NP.mean(pre_values, axis = 0).T ) , ls = '-', lw = 1, color = color )

    # take vector magnitude, thinned
    thin = 10
    pre_distance = Euclid( NP.diff(pre_values[::thin], axis = 0) )
    post_distance = Euclid( NP.diff(post_values[::thin], axis = 0) )


    # calculate log of post/pre average
    return NP.log(NP.mean(post_distance) / NP.mean(pre_distance))




def TestQuickAnalysis():

    random_scores = []
    all_scores = []

    recordings = NP.arange(2,12)
    colors = MakeColormap(recordings)
    yticks = []

    ax = MPP.gca()
    for rec in recordings:
        print ('recording', rec)
        test_force = PD.read_csv('tgif_data/20190517_daq%03.0f_forces.csv' % (rec), sep = ';')
        test_sync = PD.read_csv('tgif_data/20190517_daq%03.0f_sync.csv' % (rec), sep = ';')

        pulse_up = test_sync['type'].values == 'pulse_up'
        perturbation_sequence_begin = test_sync.loc[pulse_up, 'time'].values[0]

        yticks.append(0.2 * (rec - 2))

        actual_scores = []
        for pt in NP.arange(10., 48., 10.):
            test_timepoint = perturbation_sequence_begin + pt
            score = BalanceScore( \
                              forcetrace = test_force \
                            , test_timepoint = test_timepoint \
                            , interval = [-1., 3.] \
                            , ax = ax \
                            , color = colors[rec] \
                            , offset = 0.2 * (rec - 2)
                        )
            actual_scores.append(score)
            all_scores.append(score)
            # print ('\tstd ratio %.1f:\t' % (test_timepoint), score)

        # randomize
        for pt in NP.random.uniform(-10.,45.,100):
            test_timepoint = perturbation_sequence_begin + pt
            score = BalanceScore( \
                              forcetrace = test_force \
                            , test_timepoint = test_timepoint \
                            , interval = [-1., 3.] \
                        )
            if not NP.isnan(score):
                random_scores.append(score)

        print ('\tbalance scores actual: %.3f' % (NP.mean(actual_scores)), ',', list(map(lambda score: '%.3f' % (score), actual_scores)) )

    ax.axvline(0, color = 'k', ls = '-')
    ax.set_xlim([-1.,3.])
    ax.set_yticks(yticks)
    ax.set_yticklabels(['Sam', 'Mariëlle', 'Maaike', 'Glenn', 'Toon', 'Charlotte', 'Jamie', 'Wannes', 'Jan', 'Raoul'])
    ax.set_xlabel('time from perturbation')

    MPP.show()


    print ('balance score all actual: %.3f' % ( NP.mean(all_scores) ))
    print ('balance score random timepoint: %.3f' % ( NP.mean(random_scores) ))
    # MPP.hist(random_scores, density = True, histtype = 'step', label = 'random')
    # MPP.hist(all_scores, density = True, histtype = 'step', label = 'actual')

    import seaborn as SB
    SB.kdeplot(random_scores, label = 'random')
    SB.kdeplot(all_scores, label = 'actual')

    MPP.legend()
    MPP.show()



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
### Mission Control                                                          ###
################################################################################
if __name__ == "__main__":

    ### FT232H breakout function
    # TestGPIO(7)
    # TestGPIOInput(5)
    TestTrogger(pins = [3, 5])

    # TestSingleNXP()
    # TestMultiplexNXP()
    # TestBufferedMultiplexedSensors()

    ### MCC USB1608G DAQ function
    # TestMCCPinIn(pin_nr = 3)
    # TestMCCPinOut(pin_nr = 7)

    ### Force Plates
    # TestDAQAnalog()
    # TestOscilloscope()
    # TestForcePlate()

    # TestQuickAnalysis()

    # TestMultiDAQ()





    ## TODO:
    # check _REPEAT_DELAY reduction


