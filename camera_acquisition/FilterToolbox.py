import numpy as NP
import pandas as PD

import scipy.signal as SIG
import scipy.ndimage as NDI 
# import scipy.interpolate as INTP
import FourierToolbox as FT
import EMDToolbox as EMD



#######################################################################
### Cut Signal                                                      ###
#######################################################################
def FlipStitch(signal, invert = True):

    sample0 = signal[0]
    sigdiff = NP.diff(signal)

    # sign = 1. if invert else 1.

    sigdiff_new = NP.append(sigdiff, sigdiff[::-1]) # *sign)
    return NP.cumsum(NP.append([sample0], sigdiff_new))


def Cut(data_in, interval = None, flipstitch = False, cyclize = False):

    if interval is None:
        data_cut = data_in.copy()
    else:
        time = data_in.index.values
        selection = NP.logical_and(time >= interval[0], time < interval[1])
        data_cut = data_in.loc[selection, :]

    time = data_cut.index.values

    data_out = data_cut # in case no more actions apply

    if flipstitch:
        # rotate signal by 180deg and attach it to the end to force periodicity.
        # note that the very last sample only enters once to avoid a kink.

        data_out = {}

        # flip and attach time
        data_out['time'] = FlipStitch(time, invert = False)

        # flip and append columns
        for col in data_cut.columns:
            values = data_cut[col].values
            data_out[col] = FlipStitch(values, invert = True)

        # restore data frame
        data_out = PD.DataFrame.from_dict(data_out).set_index('time', inplace = False, drop = True)


    else:
        if cyclize:
            # equally distribute the difference between first and last sample over the cycle
            for col in data_out.columns:
                values = data_out[col].values
                delta = values[-1] - values[0]

                data_out.loc[:, col] -= NP.linspace(0, delta, len(values), endpoint = False)


    return data_out



#######################################################################
### Empirical Mode Decomposition                                    ###
#######################################################################
def EMDFilter(time, signal, sampling_rate, n_components, retain_components = None, remove_components = None, *args, **kwargs):
    intr_mode_fcns = EMD.EMD(signal, n_components = n_components, *args, **kwargs) 

    if retain_components is not None:
        return NP.nansum(NP.stack(intr_mode_fcns[retain_components]), axis = 0)

    if remove_components is not None:
        return signal - NP.nansum(NP.stack(intr_mode_fcns[remove_components]), axis = 0)

    return signal.copy() # no components removed or retained




#######################################################################
### Frequency Based                                                 ###
#######################################################################
def PlotPowerSpectrum(signal, sampling_rate = 1., *args, **kwargs):
    # , window = 'blackmanharris', nperseg = 300, noverlap = 0.95, scaling='spectrum'
    #, window = 'hanning', nperseg = 100, noverlap = 0.8)


    f, Pxx = SIG.welch(signal, 1./sampling_rate, *args, **kwargs)
    # f, Pxx = SIG.periodogram(intp_y.ravel(), 500)

    import matplotlib.pyplot as MPP
    MPP.figure()
    MPP.semilogy(f, NP.sqrt(Pxx))
    MPP.xlabel('frequency [Hz]')
    MPP.ylabel('Linear spectrum [V RMS]')
    MPP.title('Power spectrum (scipy.signal.welch)')
    MPP.show()




def ButterworthLowpass(time, signal, sampling_rate, lowpass_frequency, filter_order = 5, zero_pad = None):
    # apply lowpass filter to a signal, avoiding phase shift ("filtfilt")
    # sampling_rate: sampling frequency in rad/s
    # lowpass_frequency: cutoff frequency in rad/s
    # filter_order: order of filter
    # zero_pad: padding pre&post, samples

    if zero_pad is None:
        zero_pad = 0

    if zero_pad > 0:
        pad = NP.zeros((zero_pad,))
        signal = NP.concatenate([pad, signal, pad], axis = 0)

    # (1) filter design
    nyquist_rate = sampling_rate / 2
    normal_cutoff = lowpass_frequency * nyquist_rate
    filter_parameters = SIG.butter(filter_order, normal_cutoff, btype = 'low', analog = False)

    # (2) apply filter
    b, a = filter_parameters
    # import matplotlib.pyplot as MPP
    # MPP.close()
    # MPP.plot(time,signal)
    # MPP.show()
    # print (NP.sum(NP.isnan(signal)))
    filtered_signal = SIG.filtfilt(b, a, signal, method = 'gust')

    if zero_pad > 0:
        filtered_signal = filtered_signal[zero_pad:-zero_pad]

    return filtered_signal




#######################################################################
### Fourier Series                                                  ###
#######################################################################
def FourierSeriesFilter(time, signal, sampling_rate, filter_order = 5, n_periods = 1):
    # apply a fourier series filter, presuming the signal is periodic
    time_zeroed = time - time[0]
    period = time_zeroed[-1]/n_periods
    fsd = FT.FourierSignal.FromSignal(time_zeroed, signal, order = filter_order, period = period)
    return fsd.Reconstruct(x_reco = time_zeroed, period = period)


#######################################################################
### Filter Application                                              ###
#######################################################################

# find sampling rate in Hz, given a time array in seconds
GetSamplingRate = lambda time_s: (NP.max(time_s)-NP.min(time_s))/len(time_s)

class Filter(object):
    # a wrapper class to apply filters to a data set

    # filter options. 
    # all functions herein must have the signature Fcn(signal, time, sampling_rate)
    # additional settings:
    #   lowpass: float Empirical Mode Decomposition ro_pad
    #   fsd: int filter_order, int n_periods
    #   emd: int n_components; list(ints) retain_components OR list(ints) remove_components 
    options = {'emd': EMDFilter \
                , 'lowpass': ButterworthLowpass \
                , 'fsd': FourierSeriesFilter \
                } 


    def __init__(self, data_in, sampling_rate = None):
        self._data = data_in.copy()
        self._time = self._data.index.values

        if sampling_rate is None:
            self._sampling_rate = GetSamplingRate(self._time)
        else:
            self._sampling_rate = sampling_rate



    def Apply(self, filter_choice = None, *filter_args, **filter_kwargs):
        # Apply a chosen filter. 
        # Note that some mandatory parameters must be passed, depending on filter choice.

        if filter_choice is None:
            return self._data

        # select a filter function
        FilterFunction = self.options.get(filter_choice, None)

        if FilterFunction is None:
            raise IOError("invalid filter choice: %s not defined. Choose from: [%s]" % (filter_choice, ",".join(self.options.keys())))

        # apply the filter to all columns:
        for col in self._data.columns:
            settings = filter_kwargs.copy()

            if filter_choice == 'emd':
                for key in ['retain_components', 'remove_components']:
                    if settings.get(key, None) is not None:
                        settings[key] = settings.get(key).get(col, None)


            self._data.loc[:, col] = FilterFunction( \
                                          self._time \
                                        , self._data[col].values \
                                        , self._sampling_rate \
                                        , *filter_args \
                                        , **settings \
                                        )


        return self._data


    def PlotPowerSpectrum(self, column, *args, **kwargs):
        # plots the power spectrum of a data column
        PlotPowerSpectrum(self._data[column].values, sampling_rate = self._sampling_rate, *args, **kwargs)

    def PlotEMD(self, columns, **settings):
        # plots the EMD and IMFs of a data column
        # */** int n_components; list(ints) reco_components; list(ints) delete_components 
        for col in columns:
            sub_settings = {'n_components': settings.get('n_components', 3)}
            for key in ['retain_components', 'remove_components']:
                if settings.get(key, None) is not None:
                    sub_settings[key] = settings.get(key).get(col, None)

            EMD.PlotEMD(self._data[col].values, time = self._time, title = col, **sub_settings)
            



def ApplyFilter(data_in, sampling_rate = None, filter_choice = None, *filter_args, **filter_kwargs):
    # data_in must be a data frame with time in rows.
    # filter options will be applied to all columns equally.

    filt = Filter(data_in, sampling_rate)
    data_filtered = filt.Apply(filter_choice, *filter_args, **filter_kwargs)

    return data_filtered 



#######################################################################
### Mission Control                                                 ###
#######################################################################
if __name__ == "__main__":
    pass