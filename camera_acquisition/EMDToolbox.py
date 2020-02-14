
import numpy as NP
import scipy.signal as SIG 
import scipy.interpolate as INTP

### Empirical Mode Decomposition
# https://github.com/srcole/binder_emd
# https://srcole.github.io/2016/01/18/emd/
# https://www.sciencedirect.com/science/article/pii/S016502701400017X
def EMD(signal, n_components = 3, stoplim = .001, fill_nan = False):
    """Perform empirical mode decomposition to extract 'n_components' components out of the signal 'x'."""
    
    residual = signal.copy()
    time = NP.arange(len(residual))
    imfs = NP.zeros(n_components,dtype=object)
    for i in range(n_components):
        r_t = residual
        is_imf = False
        
        while not is_imf:
            # Identify peaks and troughs
            pks = SIG.argrelmax(r_t)[0]
            trs = SIG.argrelmin(r_t)[0]
            
            # Interpolate extrema
            pks_r = r_t[pks]
            fip = INTP.InterpolatedUnivariateSpline(pks,pks_r,k=3)
            pks_t = fip(time)
            
            trs_r = r_t[trs]
            fitr = INTP.InterpolatedUnivariateSpline(trs,trs_r,k=3)
            trs_t = fitr(time)
            
            # Calculate mean
            mean_t = (pks_t + trs_t) / 2
            mean_t = _emd_complim(mean_t, pks, trs)
            
            # Assess if this is an IMF (only look in time between peaks and troughs)
            sdk = _emd_comperror(r_t, mean_t, pks, trs)
            
            # if not imf, update r_t and is_imf
            if sdk < stoplim:
                is_imf = True
            else:
                r_t = r_t - mean_t
                
        
        if fill_nan:
            r_t[:NP.max((NP.min(pks),NP.min(trs)))] = NP.nan
            r_t[NP.min((NP.max(pks),NP.max(trs))) + 1:] = NP.nan
        imfs[i] = r_t
        residual = residual - imfs[i] 
        
    return imfs


def _emd_comperror(h, mean, pks, trs):
    """Calculate the normalized error of the current component"""
    samp_start = NP.max((NP.min(pks),NP.min(trs)))
    samp_end = NP.min((NP.max(pks),NP.max(trs))) + 1
    return NP.sum(NP.abs(mean[samp_start:samp_end]**2)) / NP.sum(NP.abs(h[samp_start:samp_end]**2))


def _emd_complim(mean_t, pks, trs):
    samp_start = NP.max((NP.min(pks),NP.min(trs)))
    samp_end = NP.min((NP.max(pks),NP.max(trs))) + 1
    mean_t[:samp_start] = mean_t[samp_start]
    mean_t[samp_end:] = mean_t[samp_end]
    return mean_t


def PlotEMD(signal, time = None, n_components = 5, reco_components = None, stoplim = .001, fill_nan = False):
### EMF
    intr_mode_fcns = EMD(signal, n_components, stoplim, fill_nan)
    if time is None:
        time = NP.linspace(0., 1., len(signal), endpoint = False)

    MPP.figure(figsize=(12,12))
    for i in range(len(intr_mode_fcns)):
        MPP.subplot(len(intr_mode_fcns),1,i+1)
        MPP.plot(time,signal-NP.nanmean(signal),color='0.6')
        MPP.plot(time,intr_mode_fcns[i],'k')
        MPP.ylabel('IMF '+NP.str(i))
        if i == len(intr_mode_fcns)-1:
            MPP.xlabel('Time (s)')

    # MPP.ylim([-1000,1000])
    MPP.show()


    if reco_components is not None:
        reconstructed = NP.sum(NP.stack(intr_mode_fcns[reco_components]), axis = 0)
        MPP.plot(time, signal-NP.nanmean(signal), color='0.6')
        MPP.plot(time, reconstructed-NP.nanmean(reconstructed), color='0.')


        MPP.show()


    return intr_mode_fcns
