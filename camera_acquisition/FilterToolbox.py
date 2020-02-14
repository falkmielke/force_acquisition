import numpy as NP
import pandas as PD

import scipy.signal as SIG
import scipy.ndimage as NDI 
# import scipy.interpolate as INTP
import FourierToolbox as FT
import EMDToolbox as EMD


#######################################################################
### EMD                                                             ###
#######################################################################
def EmpiricalModeDecomposition(time, signal):
    intr_mode_fcns = EMD.EMD(signal, n_components = 3) # , stoplim = .001, fill_nan = False
    EMD.PlotEMD(signal, time = None, n_components = 5, reco_components = [1,2,3])



#######################################################################
### Frequency Based                                                 ###
#######################################################################
def PlotPowerSpectrum(signal, sampling_rate = 1., *args, **kwargs):
    # , window = 'blackmanharris', nperseg = 300, noverlap = 0.95, scaling='spectrum'
    #, window = 'hanning', nperseg = 100, noverlap = 0.8)


    f, Pxx = SIG.welch(signal, sampling_rate, *args, **kwargs)
    # f, Pxx = SIG.periodogram(intp_y.ravel(), 500)
    MPP.figure()
    MPP.semilogy(f, NP.sqrt(Pxx))
    MPP.xlabel('frequency [Hz]')
    MPP.ylabel('Linear spectrum [V RMS]')
    MPP.title('Power spectrum (scipy.signal.welch)')
    MPP.show()




def ButterworthLowpass(data, lowpass_frequency, sampling_rate, filter_order = 5):
    # apply lowpass filter to a signal, avoiding phase shift ("filtfilt")
    # sampling_rate: sampling frequency in rad/s
    # lowpass_frequency: cutoff frequency in rad/s
    # filter_order: order of filter

    # (1) filter design
    nyquist_rate = 0.5 * sampling_rate
    normal_cutoff = lowpass_frequency / nyquist_rate
    filter_parameters = SIG.butter(filter_order, normal_cutoff, btype = 'low', analog = False)

    # (2) apply filter
    b, a = filter_parameters
    filtered_signal = SIG.filtfilt(b, a, data, method = 'gust')

    return filtered_signal




#######################################################################
### Fourier Series                                                  ###
#######################################################################
# tbd



#######################################################################
### Mission Control                                                 ###
#######################################################################
if __name__ == "__main__":
    pass