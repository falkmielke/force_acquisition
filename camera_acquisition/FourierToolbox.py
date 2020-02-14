################################################################################
### Libraries                                                                ###
################################################################################

import os as OS
import warnings as WRN
import numpy as NP
import numpy.linalg as LA
import pandas as PD



import matplotlib as MP
MP.use("TkAgg")
import matplotlib.pyplot as MPP
import matplotlib.colors as MPC
import matplotlib.cm as CM


################################################################################
### Global Specifications                                                    ###
################################################################################


# display precision
NP.set_printoptions(precision = 2)
PD.options.display.precision = 2
PD.options.display.float_format = lambda flt: "%.2f" % flt


################################################################################
### Plotting Convenience                                                     ###
################################################################################
def EqualLimits(ax):
    limits = NP.concatenate([ax.get_xlim(), ax.get_ylim()])
    max_lim = NP.max(NP.abs(limits))
    
    ax.set_xlim([-max_lim, max_lim])
    ax.set_ylim([-max_lim, max_lim])
    
    
def MakeSignalFigure():
    fig = MPP.figure(figsize = (24/2.54, 8/2.54), dpi=150)
    fig.subplots_adjust( \
                              top    = 0.82 \
                            , right  = 0.98 \
                            , bottom = 0.09 \
                            , left   = 0.10 \
                            , wspace = 0.08 # column spacing \
                            , hspace = 0.20 # row spacing \
                            )
    rows = [12]
    cols = [5,2]
    gs = MP.gridspec.GridSpec( \
                              len(rows) \
                            , len(cols) \
                            , height_ratios = rows \
                            , width_ratios = cols \
                            )
    signal_space = fig.add_subplot(gs[0]) # , aspect = 1/4
    signal_space.axhline(0, ls = '-', color = '0.5', lw = 1, zorder = 0)
    signal_space.set_xlabel(r'stride cycle')
    signal_space.set_ylabel(r'angle')
    signal_space.set_title('signal space')
    signal_space.set_xlim([0.,1.])


    frequency_space = fig.add_subplot(gs[1], aspect = 'equal')
    frequency_space.axhline(0, ls = '-', color = '0.5', lw = 1, zorder = 0)
    frequency_space.axvline(0, ls = '-', color = '0.5', lw = 1, zorder = 0)

    frequency_space.set_xlabel(r'$\Re(c_n)$')
    frequency_space.set_ylabel(r'$\Im(c_n)$')
    frequency_space.set_title('frequency space')

    frequency_space.yaxis.tick_right()
    frequency_space.yaxis.set_label_position("right")

    return fig, signal_space, frequency_space


family_colors = { \
                  'Giraffoidea': '#e6ab02' \
                , 'Bovidae': '#2e3092' \
                , 'Camelidae': '#6dcff6' \
                , 'Cervidae': '#00a651' \
                , 'Hippopotamidae': '#939598' \
                , 'Perissodactyla': '#95510a' \
                , 'Suina': '#f49ac2' \
                , 'Tragulidae': '#fff799' \
                }

supergroups = { \
                  6: 'Giraffoidea' \
                , 1: 'Bovidae' \
                , 3: 'Camelidae' \
                , 4: 'Cervidae' \
                , 8: 'Hippopotamidae' \
                , 2: 'Perissodactyla' \
                , 7: 'Suina' \
                , 5: 'Tragulidae' \
                }

def HierarchicalColorMap(grouping, cmap = "Dark2", continuous = False):
    # make a colormap from a taxonomic hierarchy
    # https://stackoverflow.com/questions/47222585/matplotlib-generic-colormap-from-tab10

    # print (grouping)
    colors = {}
    for sg, grp in grouping:
        supergroup = supergroups[int(sg)]
        colors[grp] = family_colors[supergroup]

    # main_groups = NP.unique([grp[0] for grp in grouping])

    # nc = len(main_groups)
    # nsc = [NP.sum(grouping[:, 0] == grp) for grp in main_groups]
    # # print (nsc)

    # # def categorical_cmap(nc, nsc, cmap="tab10", continuous=False):
    # if nc > MPP.get_cmap(cmap).N:
    #     raise ValueError("Too many categories for colormap.")
    # if continuous:
    #     ccolors = MPP.get_cmap(cmap)(NP.linspace(0,1,nc))
    # else:
    #     ccolors = MPP.get_cmap(cmap)(NP.arange(nc, dtype=int))

    # cols = NP.zeros((NP.sum(nsc), 3))
    # color_counter = 0
    # for i, c in enumerate(ccolors):
    #     chsv = MPC.rgb_to_hsv(c[:3])
    #     arhsv = NP.tile(chsv,nsc[i]).reshape(nsc[i],3)
    #     arhsv[:,1] = NP.linspace(chsv[1],0.25,nsc[i])
    #     arhsv[:,2] = NP.linspace(chsv[2],0.8,nsc[i])
    #     rgb = MPC.hsv_to_rgb(arhsv)

    #     cols[color_counter:color_counter+nsc[i],:] = rgb
    #     color_counter += nsc[i]


    # cmap = MPC.ListedColormap(cols)
    # # print (cols, cmap)

    # colors = {grp[1]: cols[nr, :] for nr, grp in enumerate(grouping)}
    return colors





################################################################################
### Maths                                                                    ###
################################################################################
#___________________________________________________________________________________________________
### complex numbers

two_pi_j = 1j*2*NP.pi
NormSum = lambda vec: NP.sum(vec, axis = 0)/vec.shape[0]


MakeComplex = lambda twocolumnvec: NP.add(twocolumnvec[:,0], 1j*twocolumnvec[:,1])
ComplexToMatrix = lambda complex_array: NP.stack([NP.real(complex_array), NP.imag(complex_array)], axis = 1)

StackComplexVector = lambda vector: NP.stack([NP.real(vector), NP.imag(vector)], axis = 1)


def CDF_AmpPhiTraFo(complex_dataframe):
    # transforms a data frame of complex numbers (re/im), of a structure as it is prepared in ComplexDictToDataFrame, 
    # to a similar one holding amplitudes and phases.
    polar_coefficients = {}
    remove_columns = []
    for coefficient_nr in complex_dataframe.columns.levels[0]:
        if (coefficient_nr, 'im') not in complex_dataframe.columns:
            complex_dataframe.loc[:, (coefficient_nr, 'im')] = 0
            remove_columns.append((coefficient_nr, 'phi'))

        polar_coefficients[(coefficient_nr, 'amp')] = Amplitude(complex_dataframe.loc[:, [(coefficient_nr, re_im) for re_im in ['re', 'im'] ]].values)
        polar_coefficients[(coefficient_nr, 'phi')] = Phase(complex_dataframe.loc[:, [(coefficient_nr, re_im) for re_im in ['re', 'im'] ]].values)

    polar_df = PD.DataFrame.from_dict(polar_coefficients)
    polar_df.index = complex_dataframe.index
    polar_df.drop(columns = remove_columns, inplace = True)

    return polar_df




def ComplexDictToDataFrame(complex_dict, amp_phi = False):
    # convenience function; take a dict of complex numbered arrays and turn it into a data frame.

    reals = PD.DataFrame.from_dict({idx: NP.real(vector) for idx, vector in complex_dict.items()}).T
    imags = PD.DataFrame.from_dict({idx: NP.imag(vector) for idx, vector in complex_dict.items()}).T
    
    # adjust columns
    n_coefficients = len(reals.columns)
    reals.columns = PD.MultiIndex.from_tuples([(nr, 're') for nr in range(n_coefficients)])
    imags.columns = PD.MultiIndex.from_tuples([(nr, 'im') for nr in range(n_coefficients)])
    
    # join
    coefficients = reals.join(imags, how = 'left')
        
    
    if amp_phi:
        coefficients = CDF_AmpPhiTraFo(coefficients)


    # note that, due to different signs and +/- pi ambiguity in the observations, 
    #   amps and phi do not match expectation from re and im in aggregated values.
    #   => scipy.stats.circmean should be appied to phases for real data aggregates.
    #   => amplitude and phase can be calculated back from aggregates.

    # return in better order
    return coefficients.loc[:, [(nr, coord) \
                                for nr in range(n_coefficients) \
                                for coord in ['re', 'im', 'amp', 'phi'] \
                                if not ( (nr == 0) and (coord in ['im', 'phi']) ) \
                                ]].dropna(axis = 1)


#___________________________________________________________________________________________________
### Fourier Series Decomposition
# take a signal from signal space to frequency space
# i.e. give Fourier coefficients c_n for a signal
SingleCoefficient = lambda t, T, n: NP.exp(-two_pi_j*n*t/T) / (2 if n == 0 else 1)
FourierSeriesDecomposition = lambda time, period, signal, order: \
                                        (2/( period*( NP.max(time)-NP.min(time) ) )) \
                                        * NP.array([ NP.sum(signal*SingleCoefficient(time, period, n))/len(time)  \
                                               for n in range(order+1) \
                                             ])

### Fourier Series Synthesis
# reconstruct a signal from its frequency space representation
# i.e. take coefficients and return the signal
FourierSummand = lambda t, T, cn, n: cn*NP.exp(two_pi_j*n*t/T) 
FourierFunctionT = lambda t, T, coeffs: NP.sum(NP.array([ FourierSummand(t, T, cn, n) \
                                                           for n,cn in enumerate(coeffs) \
                                                          ]))

FourierReconstruction = lambda time, period, coeffs: NP.real(period * (NP.max(time)-NP.min(time)) \
                                                    * NP.array([FourierFunctionT(t, period, coeffs) for t in time]) \
                                                            ) 

### Delay Theorem
# Bracewell, p. 111
# https://ece.umd.edu/~tretter/enee322/FourierSeries.pdf p15 Delay theorem
# https://dsp.stackexchange.com/questions/42341/fourier-series-time-shift-and-scaling
# http://www.thefouriertransform.com/transform/properties.php
# https://www.dsprelated.com/freebooks/mdft/Shift_Theorem.html

DelayFactor = lambda delay, n: NP.exp(two_pi_j*n*delay)


#___________________________________________________________________________________________________
### amplitude and phase
Amplitude = lambda values: NP.sqrt(NP.sum(NP.power(values, 2), axis = 1))
Phase = lambda values: NP.arctan2(-values[:,1],values[:,0])

ReConvertAmpPhi = lambda amp, phi: (  amp/(NP.sqrt(1+(NP.tan(phi))**2)) \
                                    , amp*NP.tan(phi)/(NP.sqrt(1+(NP.tan(phi))**2))  \
                                    )



#___________________________________________________________________________________________________
### angular transformations
# coordinate transformations
def Cart2Polar(x, y):
    r = NP.sqrt(x**2 + y**2)
    theta = NP.arctan2(y, x)
    return r, theta

def Polar2Cart(r, phi):
    x = r * NP.cos(phi)
    y = r * NP.sin(phi)
    return x, y


def WrapToInterval(vector, lower = 0, upper = 2*NP.pi):
    if not (type(vector) is int):
        vector = vector.copy()
    vector -= lower
    vector = NP.mod(vector, upper - lower)
    vector += lower

    return vector



# the opposite of NP.unwrap()
WrapAngle = lambda angle: NP.arctan2(NP.sin(angle), NP.cos(angle))

# smallest angle between two vectors
AngleBetween = lambda vec1, vec2: NP.arccos(NP.dot( vec1/LA.norm(vec1), vec2/LA.norm(vec2) )) 

# angle between two vectors, counter-clockwise from second to first point
AngleBetweenCCW = lambda p1, p2: (NP.arctan2(*p1[::-1]) - NP.arctan2(*p2[::-1])) % (2 * NP.pi)

# angle between two vectors, counter-clockwise, but wrapped to [-pi, pi]
DirectionalAngleBetween = lambda vector, reference_vector: \
        WrapToInterval( (NP.arctan2(vector[1], vector[0]) - NP.arctan2(reference_vector[1], reference_vector[0]) ) \
                    , lower = -NP.pi, upper = NP.pi \
                    )

# rotation matrix
RotationMatrixCCW = lambda ang: NP.array([ \
                                  [NP.cos(ang), -1*NP.sin(ang)] \
                                , [NP.sin(ang),    NP.cos(ang)] \
                                ])



#___________________________________________________________________________________________________



#___________________________________________________________________________________________________



################################################################################
### Fourier Series Decomposition                                             ###
################################################################################
### FSD of a Signal
class FourierSignal(object):
    """
    Wrapper object for a Fourier coefficient data frame of the structure:
         n    re    im
         0  0.06  0.00
         1  0.05  0.21
         2 -0.12  0.06
         ...

    Facilitates parsing of the coefficients to different formats.
    Multiple constructors: FromDataFrame (default behavior), FromComplexVector, FromSignal
    
    Can also be used to reconstruct arbitrary coefficients to a signal.
        
    """
### constructors
    def __init__(self, coeffs_C, label = None, raw_signal = None):
        # coeffs_C: a complex vector of coefficients
        self.label = label # for bookkeeping
        self._c = PD.DataFrame.from_dict({ \
                      'n': range(coeffs_C.shape[0]) \
                    , 're': NP.real(coeffs_C) \
                    , 'im': NP.imag(coeffs_C) \
                    }).set_index(['n'], inplace = False).loc[:,['re','im']]
        
        self.raw_signal = raw_signal # optional storage; not used
        
        
    @classmethod
    def FromDataFrame(cls, coeffs_df, label = None, raw_signal = None):
        # creates a Fourier Coefficients instance from data frame storage
        coefficients = cls(MakeComplex(coeffs_df.copy().values), label, raw_signal)
        return coefficients
        
    @classmethod
    def FromComplexVector(cls, coeffs_C, label = None, raw_signal = None):
        # creates a Fourier Coefficients instance from a complex vector of coefficients
        coefficients = cls(coeffs_C, label, raw_signal)
        return coefficients
        
    @classmethod
    def FromSignal(cls, time, signal, order = 5, period = 1., label = None):

        if len(signal) < (order * 2):
            WRN.warn("\nrecording %s (%i samples): Fourier Series order (%i) beyond Nyquist frequency." % (str(label), len(signal), order))

        # creates a Fourier Coefficients instance from a signal
        coeffs_C = FourierSeriesDecomposition(time, period, signal, order)
        
        signal_df = PD.DataFrame.from_dict({'time': time, 'signal': signal}).set_index('time', inplace = False)
        coefficients = cls(coeffs_C, label, raw_signal = signal_df)
        return coefficients
        
### frequency space to signal space
    def Reconstruct(self, x_reco = None, new_order = None, period = None):
        # reconstruct a signal 
        #   based on the given coefficients
        #   and a sampling vector

        if (new_order is None) or (new_order > (len(self))):
            new_order = len(self)
        if x_reco is None:
            x_reco = NP.linspace(0.,1.,100, endpoint = False)
        if period is None:
            period = x_reco[-1] + (x_reco[-1] - x_reco[0]) / len(x_reco)

        return FourierReconstruction( x_reco, period, self.GetVector() )

    
    def Plot(self, ax = None, x_reco = None, centered = True, y_offset = 0., plot_kwargs = {}, peak_kwargs = None):

        # replace missing arguments
        if x_reco is None:
            x_reco = NP.linspace(0.,1.,101, endpoint = True)
        if ax is None:
            ax = MPP.gca()

        # reconstruct signal
        y_reco = self.Reconstruct(x_reco, period = 1.)
        if centered:
            y_reco -= self.GetCentroid()

        # plot signal
        ax.plot(x_reco, y_offset+y_reco, **plot_kwargs)

        # scatter peak = phase as well
        if peak_kwargs is not None:
            # get curve peak
            peak = self.GetMainPhase()
            
            # invert to get trough
            inverse = self.Copy()
            Scale(inverse, -1.)
            trough =  WrapToInterval( inverse.GetMainPhase(), lower = 0., upper = 1.)

            # scatter
            x_minmax = NP.array([x_reco[0], trough, peak, x_reco[-1]])
            y_minmax = self.Reconstruct(x_minmax, period = 1.)
            if centered:
                y_minmax -= self.GetCentroid()

            ax.scatter(x_minmax[1:3], y_offset+y_minmax[1:3], **peak_kwargs)




    
### get attributes of the signal   
    def GetVector(self):
        # returns the coefficients as a complex vector
        return MakeComplex(self[:,:].values)

    def GetCoeffDataFrame(self):
        # returns the coefficients as a data frame
        return self[:,:]
    
    
    def GetCentroid(self):
        # returns the curve mean, which is the 0th coefficient
        return self[0, 're']

    def GetAmplitudes(self):
        # returns the amplitudes of coefficients
        # which is the Euclidean distance of all n>0 coefficients from center in complex plane
        return Amplitude(self._c.iloc[1:, :].values)

    def GetPhases(self):
        # returns the phases of coefficients
        # which is the complex angle of all n>0 coefficients in complex plane
        return Phase(self._c.iloc[1:, :].values)

    def GetMainPhase(self):
        # returns the default phase of the signal
        # which can actually be determined by the slope of the phases
        # http://www.dspguide.com/ch10/2.htm

        # use a ghost copy of the signal
        ghost = self.Copy()
        Center(ghost) # ... needs to be centered

        # first rotate first coefficients phase to approx zero to avoid +-pi discontinuity
        initial_phase = Phase(self[1:, :].values)[0] 
        initial_phase /= 2. * NP.pi

        # rotate the signal (explained below)
        ghost[1:, :] = ComplexToMatrix( NP.array( [coeff / DelayFactor(-initial_phase, n)  \
                                      for n, coeff in enumerate(ghost.GetVector()) if n > 0] ))
        

        # get phases and amplitudes
        phases = NP.unwrap( Phase(ghost[:, :].values) )

        # get phase differential
        # phases -= phases[0]
        d_phi = NP.diff(phases)
        
        # use amplitude weighted mean
        #phaseshift = NP.average(d_phi, weights = amplitudes)
        amplitudes = ghost.GetAmplitudes()
        if NP.all(amplitudes == 0):
            amplitudes = NP.ones(amplitudes.shape)

        # amplitudes /= amplitudes[0]
        phaseshift = NP.average(d_phi, weights = NP.divide(amplitudes, 1+NP.arange(len(amplitudes))))
        
        # bring to unit interval
        phaseshift /= 2. * NP.pi

        return WrapToInterval( initial_phase + phaseshift, lower = 0., upper = 1.)
    
    
    

### helper functions
    def DistanceFrom(self, reference):
        # Euclid distance in frequency space from another Fourier Series Decomposition
        return NP.sum(NP.abs(self[:,:].values - reference), axis = (0,1))

    def Copy(self):
        return FourierSignal.FromDataFrame(self._c, self.label, self.raw_signal)

    def __len__(self):
        # returns the order
        return self._c.shape[0]-1

    def __getitem__(self, selection):
        return self._c.loc[selection]

    def __setitem__(self, selection, values):
        self._c.loc[selection] = values

    def __add__(self, value):
        coeffs = self[:].copy()
        coeffs[0, 're'] += value
        return coeffs

    def __iadd__(self, value):
        self[0, 're'] += value
        return self

    def __sub__(self, value):
        coeffs = self[:].copy()
        coeffs[0, 're'] -= value
        return coeffs

    def __isub__(self, value):
        self[0, 're'] -= value
        return self


    def __mul__(self, value):
        coeffs = self[:].copy()
        coeffs[1:, :] *= value
        return coeffs

    def __imul__(self, value):
        self[1:, :] *= value
        return self

    def __div__(self, value):
        coeffs = self[:].copy()
        coeffs[1:, :] /= value
        return coeffs

    def __idiv__(self, value):
        self[1:, :] /= value
        return self
    
    
    def __str__(self):
        return """%s%s%s""" % ("_"*24, '\n' if self.label is None else "\n%s\n"%(self.label), str(self._c))





################################################################################
### Superimposition Operations                                               ###
################################################################################
"""
As was already visible in the object above, a signal in frequency space has three main transformation operations:

- **translation** (shifting in y-direction): adjust the mean of the signal
- **scaling**: increase/decrease the amplitude around the mean
- **rotation** (phase shifting): shift periodic signal in the time direction (= delay)


Below, functions are prepared to apply each of these transformations to the signal.
Also, reference points can be associated with each of these operations:

- translation <-> **center**: mean of the signal is zero
- scaling <-> **normalize**: amplitudes sum to one
- rotation <-> **dephase**: combined phase of the signal is zero, thus phase slope is also zero

Standardization functions transform a signal to these reference points.
"""

## Shifting
def Shift(fsd, shift):
    # shift coeffitients to change mean of the original signal
    fsd += shift # behavior defined by the "__iadd__" function above

def Center(fsd):
    # shift the signal to its mean, i.e. set first coefficient to zero
    mean = fsd.GetCentroid()
    Shift(fsd, -mean)
    return mean


## Scaling
def Scale(fsd, scaling):
    # scale the coeffitients by a factor
    fsd *= scaling

def Normalize(fsd):
    # scale the coefficients to unit amplitude
    amplitude = NP.sum(fsd.GetAmplitudes())
    Scale(fsd, 1./amplitude)
    return amplitude


## Rotation
def Rotate(fsd, delay):
    # uses the delay factor to rotate the coefficients in complex space
    # actually rotates this coefficient instance
    # ! delay should be a number within the unit interval [0; 1]; sign indicates direction
    # positive delay ==> clockwise rotation, shift to right
    fsd[1:, :] = ComplexToMatrix( NP.array( [coeff / DelayFactor(delay, n) for n, coeff in enumerate(fsd.GetVector()) if n > 0] ))

def DePhase(fsd):
    # rotates the coefficients to the best phase alignment
    phase = fsd.GetMainPhase()
    Rotate(fsd, -phase)
    return phase


### Standardization
def Standardize(fsd):
    # shift, scale and rotate to the default orientation

    ## center, i.e. shift the mean of the trace (Re(c0)) to zero
    shift_value = Center(fsd)

    ## normalize, i.e. scale so that FSD amplitudes sum to one
    scale_value = Normalize(fsd)

    ## dephase, i.e. rotate the cyclical curve to zero phase
    rot_value = DePhase(fsd)

    return shift_value, scale_value, rot_value




################################################################################
### Procrustes Alignment                                                     ###
################################################################################
"""
The operations introduced above are defined in analogy to Procrustes Superimposition in Geometric Morphometrics.

**When two signals are aligned (i.e. equal in mean, scale and phase), the difference that remains is a difference in shape.**

*Note that, because our example signal was constructed from Fourier coefficients (order $N = 5$), there will be no shape difference in this example.*
A procrustes object is defined below that aligns a signal with regard to a reference.
"""

class Procrustes(dict):
    """
    # scales, shifts and rotates a trace object to match the reference.
    """

    def __init__(self, fsd, reference, compute_distance = False, label = None):
        # calculates the alignment operators, but does not apply the transformations yet.

        if label is None:
            self.label = "PS %s onto %s" % (str(fsd.label), str(reference.label))
        else:
            self.label = label

        # use fourier decompositions
        reference_fsd = reference.Copy()
        signal_fsd = fsd.Copy()


    ## (1) Translation is first. It is independent of other transformations.
        # translation of the trace to their mean, i.e. set first coefficient to zero
        translation = signal_fsd.GetCentroid() - reference_fsd.GetCentroid()

        # center both signals, prior to further calculation
        Center(reference_fsd)
        Center(signal_fsd)

    ## (2) scaling is independent of rotation, so it follows.
        # scaling = amplitude ratio
        scale = NP.sum(signal_fsd.GetAmplitudes()) / NP.sum(reference_fsd.GetAmplitudes()) 

        # scale both fsd coefficient objects
        Normalize(reference_fsd)
        Normalize(signal_fsd)

    ## (3) finally, find optimal rotation
        rotation = signal_fsd.GetMainPhase() - reference_fsd.GetMainPhase()

        # phase-align both fsd's
        DePhase(reference_fsd)
        DePhase(signal_fsd)


    ### store results
        self['translation'] = -translation
        self['scaling'] = 1/scale
        self['rotation'] = -rotation

        self.fsd = signal_fsd
        self.reference = reference_fsd        

        if compute_distance:
            self.ComputeDistances()


    def ComputeDistances(self):
        # compute residual Euclidean distances
        signal_fsd = self.fsd.Copy()
        reference_fsd = self.reference.Copy()
        
        ### Euclid distance in frequency space
        euclid_distance = reference_fsd.DistanceFrom(self.fsd[:,:].values)

        self['residual_euclid'] = euclid_distance
        
    def HasDistances(self):
        return (self.get('residual_pd', None) is not None) or (self.get('residual_euclid', None) is not None)
        
    def GetDistances(self):
        if not self.HasDistances():
            self.ComputeDistances()
        return self['residual_euclid']


### Procrustes Superimposition
    def ApplyTo(self, fsd, skip_shift = False, skip_scale = False, skip_rotation = False):
        # shift, scale and rotate to a given reference

        ## shift
        if not skip_shift:
            Shift(fsd, self['translation'])
        
        ## scale
        if not skip_scale:
            Scale(fsd, self['scaling'])
        
        # rotate 
        if not skip_rotation:
            Rotate(fsd, self['rotation'])
        

### helpers
    def __str__(self):
        return "{label:s} \n\td_y = {translation:.2f}\n\td_A = {scaling:.2f}\n\td_phi = {rotation:.2f}".format( \
                                label = self.label \
                              , **self \
                             ) \
                + ("" if not self.HasDistances() \
                  else "\nresidual: {residual_euclid:.3f}".format(**self))
        


################################################################################
### Signal Averaging                                                         ###
################################################################################
"""
The alignment can be used to produce a better average of the traces. 
This is analogous to Generalized Procrustes analysis (GPA).

Note that averaging can happen without actual alignment of the traces.
"""
#___________________________________________________________________________________________________
### Averaging of coefficients
def AverageCoefficients(coefficient_df_list):
    coefficient_store = None
    n_observations = 0
    for coeff_df in coefficient_df_list:

        if coefficient_store is None:
            coefficient_store = coeff_df.copy()
        else:
            coefficient_store.loc[:, :] += coeff_df.values

        n_observations += 1

    coefficient_store.loc[:, :] /= n_observations

    return coefficient_store, n_observations



#___________________________________________________________________________________________________
# Average with Superimposition
def ProcrustesAverage(signals_raw, n_iterations = 1, skip_scale = True, post_align = False):
    # averages the signals in the input array in frequency space
    # by aligning copies of them prior to averaging
    # this avoids problems of interference
    # parameters:
        # post_align (optional): align the original signals to the average
        # skip_scale (optional): do not scale the average signal, therefore retaining amplitude
    
    signals = [sig.Copy() for sig in signals_raw]
    
    ### coarse alignment
    # based on first coefficient phase
    #for sig in signals:
    #    Rotate(sig, -sig.GetPhases()[0] / (2*NP.pi))
        
        
    ### initial averaging
    PreliminaryAverage = lambda signal_array: FourierSignal.FromDataFrame( \
                                                          AverageCoefficients([sig.GetCoeffDataFrame() \
                                                                               for sig in signal_array])[0] \
                                                         )
    initial_average = PreliminaryAverage(signals)

    # repeatedly calculate because the average changes with better alignment
    working_mean = None
    for it in range(n_iterations):

        # take the initial
        if working_mean is None:
            working_mean = initial_average

        # superimpose all traces to the current average
        for sig in signals:
            alignment = Procrustes(sig, working_mean, compute_distance = False)
            alignment.ApplyTo(sig, skip_scale = skip_scale)

        # get new average trace
        new_mean = PreliminaryAverage(signals)
        
        # align new mean trace back to previous one
        # (otherwise there is a drift in the traces due to rounding)
        Procrustes(new_mean, working_mean, compute_distance = False).ApplyTo(new_mean, skip_scale = skip_scale)
        working_mean = new_mean

    if post_align:
        for sig in signals_raw:
            Procrustes(sig, working_mean, compute_distance = False).ApplyTo(sig, skip_scale = False)
        
    return working_mean





################################################################################
### Limbs: Coupled Joint Angle Traces                                        ###
################################################################################
class Limb(dict):
    # joints of a limb, loosely coupled
    
    def __init__(self, array_dict, order = 5, label = None):

        self.label = label
        
        self.time = NP.linspace(0., 1., len(array_dict['carpal']), endpoint = False)
        
        # ground contact times are handled separately
        self.ground_contact = array_dict.get('gnd', None)
        array_dict.pop('gnd', None)
        self.roll_ground = 0

        # store all traces as FSD
        for joint, trace in array_dict.items():
            self[joint] = FourierSignal.FromSignal(self.time, trace, order = order, label = "%s %s" % (str(self.label), joint))
        
        self.coupled = True
        
    
    ## Shifting
    def Shift(self, *args, **kwargs):
        for joint in self.values():
            Shift(joint, *args, **kwargs)

    def Center(self, reference_joint = None):
        # shift the signal by the mean of one joint, i.e. set first coefficient to zero
        if reference_joint is None:
            # transform independently
            for trace in self.values():
                Center(trace)
            self.coupled = False
            
        else:
            self.Shift(-self[reference_joint].GetCentroid())


    ## Scaling
    def Scale(self, *args, **kwargs):
        for joint in self.values():
            Scale(joint, *args, **kwargs)

    def Normalize(self, reference_joint = None):
        # scale the coefficients to unit amplitude
        if reference_joint is None:
            # transform independently
            for trace in self.values():
                Normalize(trace)
            self.coupled = False
        else:
            self.Scale(1/NP.sum(self[reference_joint].GetAmplitudes()))


    ## Rotation
    def Rotate(self, *args, **kwargs):
        for joint in self.values():
            Rotate(joint, *args, **kwargs)

        # also rotate ground contact times
        if len(args) > 0:
            self.roll_ground += args[0]
        else: 
            self.roll_ground += kwargs.get('delay', 0)
            

    def DePhase(self, reference_joint = None):
        # rotates the coefficients to the best phase alignment
        if reference_joint is None:
            # transform independently
            for trace in self.values():
                DePhase(trace)
            self.coupled = False
        else:
            self.Rotate(-self[reference_joint].GetMainPhase())

            
    ### Standardization
    def Standardize(self, reference_joint = None):
        # shift, scale and rotate to the default orientation
        self.Center(reference_joint)
        self.Normalize(reference_joint)
        self.DePhase(reference_joint)

        
        
    ### Procrustes Alignment
    def ApplyProcrustes(self, superimposition, *args, **kwargs):
        # applies a Procrustes TraFo to all joints
        for trace in self.values():
            superimposition.ApplyTo(trace, *args, **kwargs)
    
    def AlignToReference(self, joint, reference_fsd, procrustes_kwargs = {}, superimposition_kwargs = {}):
        # perform Procrustes alignment and apply it to all joints
        superimposition = Procrustes(self[joint], reference_fsd, **procrustes_kwargs)
        self.ApplyProcrustes(superimposition, **superimposition_kwargs)
    
    def PrepareRelativeAlignment(self, focal_joint, skip_center = False, skip_normalize = False, skip_rotate = True):
        # Center and Normalize a focal joint, prior to Procrustes application,
        # so that the outcome after Procrustes will be relative mean/amplitude\
        # per default, phase is not nulled, because we want to see joints relative to the actual timeline

        if not skip_center:
            Center(self[focal_joint])

        if not skip_normalize:
            Normalize(self[focal_joint])

        if not skip_rotate:
            DePhase(self[focal_joint])
            self.coupled = False


    def Copy(self):
        limb_from_limb = Limb({k: v.Reconstruct(self.time) for k,v in self.items()}, order = len(self[[key for key in self.keys() if len(self[key]) > 0][0]]), label = self.label)
        limb_from_limb.coupled = self.coupled
        return limb_from_limb


    ### Convenience functions
    def GetDutyFactor(self):
        return NP.mean(self.ground_contact)

    def GetGroundContacts(self):
        if self.ground_contact is not None:
            return NP.roll(self.ground_contact, int(self.roll_ground*len(self.time)))


    def Plot(self, ax, title = None):
        
        for joint, trace in self.items():
            ax.plot(trace.Reconstruct(x_reco = self.time), label = joint)
            
        MPP.legend()
        if title is not None:
            ax.set_title(title)

            



################################################################################
### Testing Process                                                          ###
################################################################################
#___________________________________________________________________________________________________
### example data
coefficients = PD.DataFrame.from_dict({'re': [0.06,  0.05, -0.12, -0.03,  0.01, -0.00]
        , 'im': [0.00,  0.21,  0.06, -0.04, -0.01, -0.01]})
coefficients_C = MakeComplex(coefficients.values)

fsd = FourierSignal.FromComplexVector(coefficients_C)

# FSD parameters
period = 0.5
order = coefficients.shape[0]-1
time = NP.linspace(0.,1.,100, endpoint = False)


#___________________________________________________________________________________________________
def FormulaTest():
    ### testing Fourier formulas
    # example data
    coefficients = PD.DataFrame.from_dict({'re': [0.06,  0.05, -0.12, -0.03,  0.01, -0.00]
            , 'im': [0.00,  0.21,  0.06, -0.04, -0.01, -0.01]})
    coefficients_C = MakeComplex(coefficients.values)

    # FSD parameters
    period = 0.5
    order = coefficients.shape[0]-1
    x_reco = NP.linspace(0.,2.,200, endpoint = False)


    # reconstruct signal
    signal = FourierReconstruction(x_reco, period, coefficients_C)


    # decompose it again
    recovered_coefficients_C = FourierSeriesDecomposition(x_reco, period, signal, order)
    recovered_coefficients = ComplexToMatrix(recovered_coefficients_C)
    recovered_coefficients = PD.DataFrame(recovered_coefficients, columns = ['re', 'im'])
    print (recovered_coefficients)

    # and reconstruct once more
    recovered_signal = FourierReconstruction( x_reco, period, recovered_coefficients_C )


    ### plot for comparison
    fig, time_domain, frequency_domain = MakeSignalFigure()
    time_domain.plot(x_reco, signal, 'k-', lw = 2)
    time_domain.plot(x_reco, recovered_signal, 'r--', lw = 1)

    print ('mean', NP.mean(signal))
    time_domain.axhline(NP.mean(signal), ls = ':', color = '0.5', lw = 1)

    frequency_domain.scatter(coefficients.loc[:, 're'], coefficients.loc[:, 'im'], s = 50, marker = 'o', color = 'k')
    frequency_domain.scatter(recovered_coefficients.loc[:, 're'], recovered_coefficients.loc[:, 'im'], s = 50, marker = '+', color = 'r')
      
    EqualLimits(frequency_domain)
    MPP.show()


#___________________________________________________________________________________________________
def FSDTest():
    ### test this as before
    period = 0.5
    order = coefficients.shape[0]-1
    x_reco = NP.linspace(0.,2.,201, endpoint = True)
    time = NP.linspace(0.,1.,100, endpoint = False)

    fsd = FourierSignal.FromComplexVector(coefficients_C)

    signal = fsd.Reconstruct(x_reco = x_reco, period = period)

    restore_coeffs = FourierSignal.FromSignal(  time \
                                                    , fsd.Reconstruct(x_reco = time, period = 1.) \
                                                    , period = 1., order = order \
                                                   )
    print (restore_coeffs)

    MPP.plot(x_reco, signal, 'k-')
    MPP.show()

#___________________________________________________________________________________________________
def ShiftingTest():
    test_fsd = fsd.Copy()
    Shift(test_fsd, 0.2)
    time = NP.linspace(0.,1.,100, endpoint = False)

    ### plot for comparison
    fig, time_domain, frequency_domain = MakeSignalFigure()

    # modified signal
    time_domain.plot(time, test_fsd.Reconstruct(time), 'k-', lw = 2, label = 'shifted')
    time_domain.axhline(test_fsd.GetCentroid(), ls = ':', color = 'k', lw = 1)
    frequency_domain.scatter(test_fsd[:, 're'], test_fsd[:, 'im'], s = 50, marker = 'o', color = 'k')

    # standard signal
    Center(test_fsd)
    time_domain.plot(time, test_fsd.Reconstruct(time), ls = '-', lw = 1, color = (0.2,0.4,0.2), label = 'centered')
    frequency_domain.scatter(test_fsd[:, 're'], test_fsd[:, 'im'], s = 50, marker = 'x', color = (0.2,0.4,0.2))

    # reference signal
    time_domain.plot(time, fsd.Reconstruct(time), ls = '--', lw = 1, color = '0.5', label = 'original')
    time_domain.axhline(fsd.GetCentroid(), ls = ':', color = '0.5', lw = 1)
    frequency_domain.scatter(fsd[:, 're'], fsd[:, 'im'], s = 50, marker = '+', color = '0.5')

    EqualLimits(frequency_domain)
    time_domain.legend()
    MPP.show()


def ScalingTest():
    test_fsd = fsd.Copy()
    Scale(test_fsd, 1.5)
    time = NP.linspace(0.,1.,100, endpoint = False)

    ### plot for comparison
    fig, time_domain, frequency_domain = MakeSignalFigure()

    # modified signal
    time_domain.plot(time, test_fsd.Reconstruct(time), 'k-', lw = 2, label = 'scaled')
    time_domain.axhline(test_fsd.GetCentroid(), ls = ':', color = 'k', lw = 1)
    frequency_domain.scatter(test_fsd[:, 're'], test_fsd[:, 'im'], s = 50, marker = 'o', color = 'k')

    # standard signal
    Normalize(test_fsd)
    time_domain.plot(time, test_fsd.Reconstruct(time), ls = '-', lw = 1, color = (0.2,0.4,0.2), label = 'normalized')
    frequency_domain.scatter(test_fsd[:, 're'], test_fsd[:, 'im'], s = 50, marker = 'x', color = (0.2,0.4,0.2))

    # reference signal
    time_domain.plot(time, fsd.Reconstruct(time), ls = '--', lw = 1, color = '0.5', label = 'original')
    time_domain.axhline(fsd.GetCentroid(), ls = ':', color = '0.5', lw = 1)
    frequency_domain.scatter(fsd[:, 're'], fsd[:, 'im'], s = 50, marker = '+', color = '0.5')

    EqualLimits(frequency_domain)
    time_domain.legend()
    MPP.show()


def RotationTest():
    test_fsd = fsd.Copy()
    Rotate(test_fsd, -0.25)
    time = NP.linspace(0.,1.,100, endpoint = False)

    ### plot for comparison
    fig, time_domain, frequency_domain = MakeSignalFigure()

    # modified signal
    time_domain.plot(time, test_fsd.Reconstruct(time), 'k-', lw = 2, label = 'rotated')
    time_domain.axhline(test_fsd.GetCentroid(), ls = ':', color = 'k', lw = 1)
    frequency_domain.scatter(test_fsd[:, 're'], test_fsd[:, 'im'], s = 50, marker = 'o', color = 'k')

    # standard signal
    DePhase(test_fsd)
    time_domain.plot(time, test_fsd.Reconstruct(time), ls = '-', lw = 1, color = (0.2,0.4,0.2), label = 'aligned')
    frequency_domain.scatter(test_fsd[:, 're'], test_fsd[:, 'im'], s = 50, marker = 'x', color = (0.2,0.4,0.2))

    # reference signal
    time_domain.plot(time, fsd.Reconstruct(time), ls = '--', lw = 1, color = '0.5', label = 'original')
    time_domain.axhline(fsd.GetCentroid(), ls = ':', color = '0.5', lw = 1)
    frequency_domain.scatter(fsd[:, 're'], fsd[:, 'im'], s = 50, marker = '+', color = '0.5')

    EqualLimits(frequency_domain)
    time_domain.legend()
    MPP.show()


def StandardizationTest():
    test_fsd = fsd.Copy()
    time = NP.linspace(0.,1.,100, endpoint = False)

    Standardize(test_fsd)

    ### plot for comparison
    fig, time_domain, frequency_domain = MakeSignalFigure()

    # standard signal
    time_domain.plot(time, test_fsd.Reconstruct(time), ls = '-', lw = 1, color = (0.2,0.4,0.2), label = 'standardized')
    frequency_domain.scatter(test_fsd[:, 're'], test_fsd[:, 'im'], s = 50, marker = 'x', color = (0.2,0.4,0.2))

    # reference signal
    time_domain.plot(time, fsd.Reconstruct(time), ls = '--', lw = 1, color = '0.5', label = 'original')
    time_domain.axhline(fsd.GetCentroid(), ls = ':', color = '0.5', lw = 1)
    frequency_domain.scatter(fsd[:, 're'], fsd[:, 'im'], s = 50, marker = '+', color = '0.5')

    EqualLimits(frequency_domain)
    time_domain.legend()
    MPP.show()


#___________________________________________________________________________________________________
def TestProcrustesAlignment():
    test_fsd = fsd.Copy()

    # modify a signal
    Shift(test_fsd, -0.1)
    Scale(test_fsd, 1/1.2)
    Rotate(test_fsd, 0.1)

    # procrust the test signal onto the original one
    superimposition = Procrustes(test_fsd, fsd, compute_distance = False, label = 'test superimposition')
    superimposition.GetDistances()
    print (superimposition)

    ### plot for comparison
    fig, time_domain, frequency_domain = MakeSignalFigure()

    # modified signal
    time_domain.plot(time, test_fsd.Reconstruct(time), 'k-', lw = 2, label = 'modified')
    time_domain.axhline(test_fsd.GetCentroid(), ls = ':', color = 'k', lw = 1)
    frequency_domain.scatter(test_fsd[:, 're'], test_fsd[:, 'im'], s = 50, marker = 'o', color = 'k')

    # standard signal
    superimposition.ApplyTo(test_fsd)
    time_domain.plot(time, test_fsd.Reconstruct(time), ls = '-', lw = 1, color = (0.2,0.4,0.2), label = 'aligned')
    frequency_domain.scatter(test_fsd[:, 're'], test_fsd[:, 'im'], s = 50, marker = 'x', color = (0.2,0.4,0.2))

    # reference signal
    time_domain.plot(time, fsd.Reconstruct(time), ls = '--', lw = 3, color = '0.5', label = 'original')
    time_domain.axhline(fsd.GetCentroid(), ls = ':', color = '0.5', lw = 1)
    frequency_domain.scatter(fsd[:, 're'], fsd[:, 'im'], s = 50, marker = '+', color = '0.5')

    EqualLimits(frequency_domain)
    time_domain.legend()
    MPP.show()




#___________________________________________________________________________________________________
def TestAveraging():
    NP.random.seed(123)
    n_signals = 5
    noise = 0.02

    signals = []
    for nr in range(n_signals):
        sig = fsd.GetCoeffDataFrame().copy()
        sig.loc[:,:] += NP.random.normal(loc = 0.0, scale = noise, size = sig.shape)
        sig.loc[0,'im'] = 0
        
        sig = FourierSignal.FromDataFrame(sig, label = 'sig%i' % (nr)) 
        Shift(sig, NP.random.uniform(-0.1, 0.1, 1)[0])
        Scale(sig, NP.random.uniform(0.95, 1.05, 1)[0])
        Rotate(sig, NP.random.uniform(-0.1, 0.1, 1)[0])
        
        signals.append(sig)
        
        
    ### colors
    lm = NP.arange(len(signals))
    colormap = CM.ScalarMappable(norm = MPC.Normalize(vmin = lm.min(), vmax = lm.max()), cmap = MPP.get_cmap('Accent') )
    colors = {sig.label: colormap.to_rgba(nr) for nr, sig in enumerate(signals)}

        
    ### plot for comparison
    fig, time_domain, frequency_domain = MakeSignalFigure()
    fig.suptitle('signals before superimposition')

    for sig in signals:
        time_domain.plot(time, sig.Reconstruct(time), ls = '-', lw = 1, color = colors[sig.label], label = sig.label)
        frequency_domain.scatter(sig[:, 're'], sig[:, 'im'], s = 50, marker = 'o', color = colors[sig.label])


    unaligned_average = FourierSignal.FromDataFrame( \
                                          AverageCoefficients([sig.GetCoeffDataFrame() \
                                                               for sig in signals])[0] \
                                         )
    time_domain.plot(time, unaligned_average.Reconstruct(time), ls = '--', lw = 2, color = 'r', label = 'unaligned')
    frequency_domain.scatter(unaligned_average[:, 're'], unaligned_average[:, 'im'], s = 50, marker = '+', color = 'r')
        
    average_signal = ProcrustesAverage(signals, n_iterations = 5, skip_scale = True, post_align = False)
    time_domain.plot(time, average_signal.Reconstruct(time), ls = '-', lw = 2, color = 'k', label = 'aligned')
    frequency_domain.scatter(average_signal[:, 're'], average_signal[:, 'im'], s = 50, marker = 'x', color = 'k')

    EqualLimits(frequency_domain)
    time_domain.legend()
    



    procrustesses = {}
    for sig in signals:
        proc = Procrustes(sig, average_signal, compute_distance = True)
        proc.ApplyTo(sig, skip_scale = False)
        procrustesses[sig.label] = proc
        
    ### plot for comparison
    fig, time_domain, frequency_domain = MakeSignalFigure()
    fig.suptitle('signals after superimposition')

    for sig in signals:
        time_domain.plot(time, sig.Reconstruct(time), ls = '-', lw = 1, color = colors[sig.label], label = sig.label)
        frequency_domain.scatter(sig[:, 're'], sig[:, 'im'], s = 50, marker = 'o', color = colors[sig.label])


    time_domain.plot(time, average_signal.Reconstruct(time), ls = '-', lw = 1, color = 'k', label = 'mean')
    frequency_domain.scatter(average_signal[:, 're'], average_signal[:, 'im'], s = 50, marker = 'x', color = 'k')

    EqualLimits(frequency_domain)
    time_domain.legend()
    MPP.show()




#___________________________________________________________________________________________________
def CollectTestLimbs(filtering = {'species': ['Equus caballus', 'Lama glama']}):
    import DataAggregationToolbox as DAT

    master_data = DAT.LoadMasterData(from_backup = True)

    # filter for horses and lamas
    selection = master_data.index.labels[0] >= 0 # NP.ones((master_data.shape[0],))
    for parameter, selected in filtering.items():
        selection = NP.logical_and( selection,  \
                                [sel in selected for sel in master_data[parameter].values] \
                                )

    master_data = master_data.loc[selection, :]

    # make limbs
    limbs = {}
    for idx, info in master_data.iterrows():
        limbs[idx] = DAT.ComputeAngles(  DAT.PointsFromFile(info['trace_file']) \
                                    , direction_towards_right = info['direction_towards'].upper() == 'R' \
                                   )

    return limbs

#___________________________________________________________________________________________________
def TestLimbCoupling():
    limbs = CollectTestLimbs()

    # go testing
    test_limb = NP.random.choice(list(limbs.values()))


    fig = MPP.figure()

    test_joints = Limb(test_limb)
    test_joints.Plot(fig.add_subplot(5,1,1), title = 'as is')


    test_joints.Rotate(-0.3)
    test_joints.Plot(fig.add_subplot(5,1,2), title = 'rotated')


    test_joints.Standardize('carpal')
    test_joints.Plot(fig.add_subplot(5,1,3), title = 'standardized by carpal')


    reference = test_joints['elbow'].Copy()
    Standardize(reference)

    Rotate(reference, 0.1)
    Scale(reference, 0.5)
    Shift(reference, -2.6)

    test_joints.AlignToReference('elbow', reference)
    test_joints.Plot(fig.add_subplot(5,1,4), title = 'Procrusted')


    test_joints.Standardize()
    test_joints.Plot(fig.add_subplot(5,1,5), title = 'coupled = %s' % (str(test_joints.coupled)) )


    MPP.show()

#___________________________________________________________________________________________________
def TestJointAverageAlignment():

    ### selection
    colors = {'Equus': (0.2, 0.2, 0.6), 'Giraffa': (0.2, 0.6, 0.2), 'Tapirus': (0.6, 0.2, 0.2), 'Alces': (0.6, 0.6, 0.2)}
    limbs = {}

    reference_joint = 'carpal'
    focal_joint = 'elbow'
    reco_time = NP.linspace(0., 1., 100, endpoint = False)
    

    ### collect data
    for genus in colors.keys():
        limbdata = CollectTestLimbs(filtering = {'genus': [genus]})
        limbs[genus] = [Limb(lmb, label = idx) for idx, lmb in limbdata.items()]


    ### find averages
    average_refjoints = {group: ProcrustesAverage( \
                                [lmb[reference_joint] for lmb in some_limbs] \
                                , n_iterations = 5, skip_scale = False, post_align = False) \
                        for group, some_limbs in limbs.items() \
                        }

    ### align averages with each other
    global_average = ProcrustesAverage( \
                                [groupaverage_joint for groupaverage_joint in average_refjoints.values()] \
                                , n_iterations = 5, skip_scale = False, post_align = True \
                                )



    ### now align all of them:
    for group, group_limbs in limbs.items():
        for lmb in group_limbs:
            # standardize the shift and amplitude of the focal joint to give relative values
            lmb.PrepareRelativeAlignment(focal_joint, skip_rotate = True)

            # align all joints, based on the reference
            lmb.AlignToReference(reference_joint, average_refjoints[group])

    ### prepare figure
    fig, time_domain, frequency_domain = MakeSignalFigure()

    ### store the relevant data
    focal_averages = []

    ### iterate groups
    for group, group_limbs in limbs.items():

        # plot all observations
        for lmb in group_limbs:
            time_domain.plot(  reco_time \
                    , lmb[reference_joint].Reconstruct(reco_time) \
                    , color = '0.5' \
                    , alpha = 0.2 \
                    , lw = 1 \
                    , ls = ':' \
                    , zorder = 10 \
                    )
            time_domain.plot(  reco_time \
                    , lmb[focal_joint].Reconstruct(reco_time) \
                    , color = colors[group] \
                    , alpha = 0.2 \
                    , lw = 1 \
                    , ls = '-' \
                    , zorder = 10 \
                    )

            frequency_domain.scatter( \
                      lmb[focal_joint][1:, 're'] \
                    , lmb[focal_joint][1:, 'im'] \
                    , s = 30 \
                    , marker = '+' \
                    , color = colors[group] \
                    , alpha = 0.2 \
                    )

        # plot group reference joint average
        time_domain.plot(  reco_time \
                , average_refjoints[group].Reconstruct(reco_time) \
                , color = colors[group] \
                , lw = 2 \
                , ls = '-' \
                , zorder = 20 \
                )


        # calculate and plot group focal joint average
        focal_average = ProcrustesAverage( \
                                [lmb[focal_joint] for lmb in group_limbs] \
                                , n_iterations = 5, skip_scale = False, post_align = False)
        time_domain.plot(  reco_time \
                , focal_average.Reconstruct(reco_time) \
                , color = colors[group] \
                , lw = 2 \
                , ls = '-' \
                , zorder = 20 \
                , label = group \
                )
        frequency_domain.scatter( \
                  focal_average[1:, 're'] \
                , focal_average[1:, 'im'] \
                , s = 50 \
                , marker = 'o' \
                , color = colors[group] \
                , alpha = 1. \
                )

        focal_averages.append(focal_average)


    ### polish plot
    time_domain.set_xlim([reco_time[0], reco_time[-1]])
    time_domain.legend()
    EqualLimits(frequency_domain)
    MPP.show()


    ### align averages with each other
    global_focus = ProcrustesAverage( \
                                [groupaverage_joint for groupaverage_joint in focal_averages] \
                                , n_iterations = 5, skip_scale = False, post_align = False \
                                )


# Two variants of analyzing the focal joint:
# (1) collect Fourier coefficients without Procrustes to the mean
#       -> all variation will be in the coefficients
# (2) spin out shift, scale, rotation and residual
#       (a) variation will be split in the affine and anisotropic parts
#       (b) coefficients contain more detail about the (anisotropic) residual


    ### gather distances of all traces from global focal average
    focal_coefficients_pre = {}
    focal_procrustesses = {}
    focal_coefficients_post = {}
    for group, group_limbs in limbs.items():
        # store data
        for lmb in group_limbs:
            label = (group, lmb.label[0], lmb.label[1], focal_joint)

    # (1) pre-procrustes coefficients
            focal_coefficients_pre[label] = lmb[focal_joint].GetVector()

    # (2a) variation split
            focal_procrustesses[label] = Procrustes(lmb[focal_joint], global_focus, compute_distance = True, label = "{:s} {:.0f} {:.0f} {:s}".format(*label) )


    # (2b) residual
            focal_procrustesses[label].ApplyTo(lmb[focal_joint])
            focal_coefficients_post[label] = lmb[focal_joint].GetVector()




    ### organize differences in a data frame
    differences = PD.DataFrame.from_dict({idx: {**proc}  \
                                          for idx, proc in focal_procrustesses.items()}).T

    ### print quantitative results, group aggregated
    def DropColumnLevel(df, drop_level):
        df.columns = df.columns.droplevel( drop_level )
        return df.dropna(axis = 1)


    print (ComplexDictToDataFrame(focal_coefficients_pre, amp_phi = True).groupby(level = 0).agg([NP.mean]))
    print (CDF_AmpPhiTraFo(  DropColumnLevel( ComplexDictToDataFrame(focal_coefficients_pre, amp_phi = False).groupby(level = 0).agg([NP.mean]), drop_level = 2)  ))
    print (differences.groupby(level = 0).agg([NP.mean, NP.std]))
    print (ComplexDictToDataFrame(focal_coefficients_post).groupby(level = 0).agg([NP.mean]))




def RotationAnimation():

    time = NP.linspace(0.,1.,100, endpoint = False)
    reference_fsd = fsd.Copy()
    Standardize(reference_fsd)



    point_storage = []
    for rot in NP.linspace(0., 1., 200, endpoint = False):
        ### plot for comparison
        fig, signal_domain, frequency_domain = MakeSignalFigure()

        # modified signal
        signal_domain.plot(time, reference_fsd.Reconstruct(time), 'k-', lw = 2, label = 'original', zorder = 20)
        #signal_domain.axhline(test_fsd.GetCentroid(), ls = ':', color = 'k', lw = 1)
        frequency_domain.scatter(reference_fsd[1:, 're'], reference_fsd[1:, 'im'], s = 50, marker = 'o', edgecolor = 'k', facecolor = 'none', zorder = 20)

        # standard signal
        test_fsd = reference_fsd.Copy()
        Rotate(test_fsd, rot)
        signal_domain.plot(time, test_fsd.Reconstruct(time), ls = '-', lw = 1, color = (0.2,0.2,0.6), label = 'phase shifted', zorder = 30)
        frequency_domain.scatter(test_fsd[1:, 're'], test_fsd[1:, 'im'], s = 50, marker = 'd', color = (0.2,0.2,0.6), zorder = 30)
        
        for pt in point_storage:
            frequency_domain.scatter(pt[:, 0], pt[:, 1], s = 10, marker = '.', color = '0.0', zorder = 10, alpha = 0.2)

        if (rot > 0) and ((rot*100) % 2 == 0):
            point_storage.append(test_fsd[1:, ['re', 'im']].values)

        frequency_domain.set_xlim([-0.6, 0.6])
        EqualLimits(frequency_domain)
        signal_domain.legend(bbox_to_anchor=(0.28, -0.1), ncol = 2, fontsize = 8)#)

        fig.savefig('rotation_animation/rot%03.0f.png' % (rot*1000), dpi = 150, transparent = False )
        MPP.close()


    # ffmpeg -framerate 20 -pattern_type glob -i 'rotation_animation/rot*.png' -c:v libx264 -preset veryslow rotation_animation/phaseshift.mp4 
     # -vf "scale=800:-1"
    # convert -loop 0 rotation_animation/rot*.png rotation_animation/phaseshift_web.gif


    # ffmpeg -i rotation_animation/phaseshift.mp4 -vf fps=10 rotation_animation/phaseshift_web.gif
    # -vf "scale=800:-1,crop=in_w:in_h-160"




################################################################################
### Mission Control                                                          ###
################################################################################
if __name__ == "__main__":
    # FormulaTest()

    # FSDTest()

    # ShiftingTest()
    # ScalingTest()
    # RotationTest()
    # StandardizationTest()

    # TestProcrustesAlignment()

    # TestAveraging()


    # TestLimbCoupling()
    TestJointAverageAlignment()

    pass


    # TODO:
    """

    """