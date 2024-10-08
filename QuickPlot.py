import os as OS
import re as RE
import numpy as NP
import pandas as PD
import scipy.stats as STATS
import sympy as SYM
import sympy.physics.mechanics as MECH

import matplotlib as MP
MP.use('TkAgg')
import matplotlib.pyplot as MPP


# PD.set_option('precision', 5)
SYM.init_printing(use_latex=False)




#######################################################################
### Force Plate Calibration Parameters                              ###
#######################################################################
def LoadCalibrationData():
    ### meta info
    
    forceplate_info = PD.DataFrame.from_dict({ \
                      1: ['A', 'blue' , 'Z17097'   , 1253300 , forceplate_sensitivity , -0.06*0., -0.12*0., '', topmount_padding * (-1e-3) ] \
                    , 2: ['B', 'blue' , 'Z17097Q01', 5201448 , forceplate_sensitivity , +0.06, -0.12, '', topmount_padding * (-1e-3) ] \
                    , 3: ['C', 'green', 'Z17097Q01', 5201449 , forceplate_sensitivity , -0.06, +0.12, '', topmount_padding * (-1e-3) ] \
                    , 4: ['D', 'green', 'Z17097'   , 4656828 , forceplate_sensitivity , +0.06, +0.12, '', topmount_padding * (-1e-3) ] \
                        }).T
    forceplate_info.columns = ['label', 'daq', 'model', 'serial', 'sensitivity', 'center_x', 'center_y', 'inversion', 'padding']
    forceplate_info.index.name = 'fp_nr'
    # print (forceplate_info)

    ### padding
    padding = PD.read_csv('calibration/Kistler_padding_results.csv', sep = ';')
    padding.index = [int(fp[2]) for fp in padding['fp'].values]
    padding.rename(columns = {'p_z': 'pad0'}, inplace = True)

    forceplate_info = forceplate_info.join(padding.loc[:, 'pad0'])


    ### slope
    slopes = PD.read_csv('calibration/Kistler_slopes_result.csv', sep = ',').set_index('Unnamed: 0', inplace = False)
    slopes = slopes.loc[[fp for fp in slopes.index.values if 'slope_fp' in fp ], :]
    slopes.index.name = 'fp_nr'
    slopes.index = [int(fp[-1])+1 for fp in slopes.index.values]
    slopes.rename(columns = {'mean': 'slope'}, inplace = True)
    
    forceplate_info = forceplate_info.join(slopes.loc[:, 'slope'])
    

    return forceplate_info




#######################################################################
### Force Plate Algebra                                             ###
#######################################################################
def VoltsToNewtons(force_raw, forceplate_info):
    # F=Vout⋅g⋅sensitivity/slope
    forces = force_raw.copy()


    for col in forces.columns:
        label = col[0] # forceplate letter
        forces.loc[:, col] *= 9.81 * forceplate_info['sensitivity'] / forceplate_info['slope']


    return forces


# vector (de)composition helpers
coordinates = ['x', 'y', 'z']
VectorComponents = lambda vec, rf: [vec.dot(rf.x), vec.dot(rf.y), vec.dot(rf.z)]
MakeVector = lambda components, coords, n = 3: sum(NP.multiply(components[:n], coords[:n]))
ChangeAFrame = lambda vec, rf_old, rf_new: MakeVector(VectorComponents(vec, rf_old), [rf_new.x, rf_new.y, rf_new.z])

SmallAngleApprox = lambda ang: [(SYM.sin(ang), ang), (SYM.cos(ang), 1), (SYM.tan(ang), ang)]

def PrintSolution(solution_dict):
    # print and substitute in known values
    for param, eqn in solution_dict.items():
        print ('\n', '_'*20)
        print (param)
        print ('_'*20)
        SYM.pprint(eqn) # .subs(fp_dimensions)


def ForceplateFormulae():
    ### reference frame of the force plate
    world = MECH.ReferenceFrame('N')
    origin = MECH.Point('O')
    origin.set_vel(world, 0)

    phi = SYM.symbols('phi_{x:z}') # force plate rotation in the world
    forceplate = world.orientnew('fp', 'Body', phi[::-1], 'ZYX')

    # center point = force plate center of mass
    forceplate_offset = SYM.symbols('c_{x:z}')
    center = origin.locatenew('C', MakeVector(forceplate_offset, [world.x, world.y, world.z]))

    # force plate coordinate system
    x = forceplate.x
    y = forceplate.y
    z = forceplate.z
    coordinate_symbols = [x, y, z]

    # notation: 
    #    i ... leg index
    #    j ... coordinate index

    # impact point
    pj = SYM.symbols('p_{x:z}')
    impact = center.locatenew('P', MakeVector(pj, coordinate_symbols))

    # impact position vector
    rc = impact.pos_from(center)

    # leg points, relative to force plate center
    n_legs = 4
    leg_ids = range(n_legs)
    leg_positions = NP.array([(k,l) for k, l in [[+1,+1], [-1,+1], [-1,-1], [+1,-1]] ])

    # leg vectors
    qj = SYM.symbols('q_{x:y}') # plate dimensions
    rq = [leg_positions[i,0]*qj[0]*x + leg_positions[i,1]*qj[1]*y + 0*z for i in leg_ids] 

    # leg points
    legs = [center.locatenew('Q_{%i}' % (i), rq[i]) for i in leg_ids]

    # vector from leg to impact
    sq = [leg.pos_from(impact) for leg in legs] # leg - impact


    ### Balance of Forces
    fij = NP.array(SYM.symbols('f_{:4x:z}')).reshape(n_legs, -1)
    Fi = [MakeVector(fij[i,:], coordinate_symbols) for i in leg_ids]
    Fj = SYM.symbols('F_{x:z}')

    impact_components = Fj
    reactn_components = VectorComponents(sum(Fi), forceplate)

    force_balances = [SYM.Eq(impact_components[coord], reactn_components[coord]) for coord, _ in enumerate(coordinate_symbols) ]
    force_subs = [(imp, rcn) for imp, rcn in zip(impact_components, reactn_components)]


    ### Balance of Moments
    # free moment
    Tz = SYM.symbols('T_{z}')


    # moments of the impact force on the whole object
    impact_moments = rc.cross(MakeVector(Fj, coordinate_symbols))
    impact_moments += Tz*z

    # moments of the reaction forces on the legs on the whole object (i.e. center)
    reactn_moments = [rq[i].cross(MakeVector(fij[i, :], coordinate_symbols )) \
                      for i in leg_ids]

    reactn_moments = [vc.factor(pj+qj) for vc in VectorComponents(sum(reactn_moments), forceplate)]

    moment_equations = [(imp - rcn).subs(force_subs).simplify() \
                        for imp, rcn in zip(VectorComponents(impact_moments, forceplate), reactn_moments)]

    ### Solution
    fx01, fx23, fy03, fy12 = SYM.symbols('fx_{01}, fx_{23}, fy_{03}, fy_{12}')
    fz = SYM.symbols('fz_{:4}')

    input_parameters = [fx01, fx23, fy03, fy12] + [f for f in fz]

    group_subs = [ (fij[0,0]+fij[1,0], fx01) \
                 , (fij[2,0]+fij[3,0], fx23) \
                 , (fij[0,1]+fij[3,1], fy03) \
                 , (fij[1,1]+fij[2,1], fy12) \
                 ] + [ \
                 (fij[i,2], fz[i]) for i in leg_ids \
                 ]

    moment_equations = [SYM.factor(mmnt, qj+pj).subs(group_subs) for mmnt in moment_equations]
    solvents = [pj[0], pj[1]]

    p_solutions = [ \
                 {solvents[s]: SYM.factor(sln, pj+qj) \
                      for s, sln in enumerate(solution)} \
                 for solution in SYM.nonlinsolve(moment_equations, solvents) \
                ][0]

    # free torque
    tz_equation = moment_equations[2].subs([(pnt, sol) for pnt, sol in p_solutions.items()]).factor(qj+pj)
    tz_solution = [sol for sol in SYM.solveset(tz_equation, Tz)][0]

    # contact point / centre of pressure
    cop_solutions = {Tz: tz_solution.simplify(), **p_solutions}

    ### Resubstitution
    force_solutions = {imp: rcn.subs(group_subs) for imp, rcn in force_subs}
    Mj = SYM.symbols('M_{x:z}')
    moment_solutions = VectorComponents(impact_moments.subs([(param, sol) for param, sol in cop_solutions.items()]), forceplate)

    moment_solutions = {Mj[j]: moment_solutions[j].subs( \
                                                        [(force, sol) for force, sol in force_solutions.items()] \
                                                       ).simplify() \
                        for j, _ in enumerate(coordinate_symbols)}

    ### Compined Solution
    all_solutions = {**force_solutions, **moment_solutions, **cop_solutions}
    # PrintSolution(all_solutions)
    
    # padding pj[2] has to come from the data.
    fp_dimensions = [(qj[0], 0.035), (qj[1], 0.075)]
    
    all_solutions = {param: eqn.subs(fp_dimensions) for param, eqn in all_solutions.items()}
    input_parameters.append(pj[2])


    return all_solutions, input_parameters




def ConvertForceCoordinates(force_formulae, input_parameters, forceplate_info, forces_brute):

    ### (1) re-organize forces: dict of force plates
    rawmeasure_labels = measured_columns.copy()
    # print (input_parameters)
    # print (forces_brute.columns)

    measurement = forces_brute.loc[:, [col for col in rawmeasure_labels]]

    ### padding is critical. Make sure the sign is correct! (i.e. negative on Kistler)
    measurement['pad0'] = forceplate_info['pad0'] + forceplate_info['padding']
    # print (measurement.head())

    rawmeasure_labels.append('pad0')


    ### (2) apply force plate formulae
    out_parameters = list(force_formulae.keys())
    ComponentsToParameters = { str(param): \
                                  SYM.lambdify(input_parameters, eqn, "numpy") \
                              for param, eqn in force_formulae.items()\
                             }

    datavectors = [measurement.loc[:,col].values for col in rawmeasure_labels]


    calculated_values = PD.DataFrame.from_dict({param: eqn(*datavectors) for param, eqn in ComponentsToParameters.items()})
    calculated_values.index = measurement.index

    forces = calculated_values

    ### (3) transform to world coordinate system
    for coord in coordinates:
        if coord in forceplate_info['inversion']:
            coordinate_columns = [col for col in forces.columns if '{%s}' % (coord) in col]
            forces.loc[:, coordinate_columns] *= -1

    # # mask contact points outside the force plate
    # forces.loc[0.012 < NP.abs(forces['p_{x}'].values), 'p_{x}'] = NP.nan
    # forces.loc[0.020 < NP.abs(forces['p_{y}'].values), 'p_{y}'] = NP.nan
    
    # # mask sub-threshold forces
    # threshold_force = 0.4 # N
    # mask = NP.sum(NP.abs(forces.loc[:, ['F_{x}','F_{y}','F_{z}']].values), axis = 1) < threshold_force
    # forces.loc[mask, 'p_{x}'] = NP.nan
    # forces.loc[mask, 'p_{y}'] = NP.nan

    # shift to world coordinates
    forces.loc[:, 'p_{x}'] += forceplate_info['center_x']
    forces.loc[:, 'p_{y}'] += forceplate_info['center_y']

    return forces




#######################################################################
### Plotting                                                        ###
#######################################################################

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

    # MP.rcParams['text.latex.preamble'] = [\
    #               r'\usepackage{upgreek}'
    #             , r'\usepackage{cmbright}'
    #             , r'\usepackage{sansmath}'
    #             ]

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


def FullDespine(ax):
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(bottom = False)
    ax.tick_params(left = False)
    ax.set_xticks([])
    ax.set_yticks([])


def MakeFigure(rows = [1], cols = [1], dimensions = [16,12]):
    
    style = 'dark_background'
    PreparePlot()

    if style is not None:
        # print(MPP.style.available)
        MPP.style.use(style) #'seaborn-paper'
    # custom styles can go to ~/.config/matplotlib/stylelib
    # originals in /usr/lib/python3.6/site-packages/matplotlib/mpl-data/stylelib
    



    
# set figure size with correct font size
    # to get to centimeters, the value is converted to inch (/2.54) 
    #                        and multiplied with fudge factor (*1.25).
    # The image then has to be scaled down from 125% to 100% to remove the fudge scaling.
    cm = 1./2.54
    figwidth  = dimensions[0] * cm
    figheight = dimensions[1] * cm

# define figure
    # columnwidth = 455.24411 # from latex \showthe\columnwidth
    fig = MPP.figure( \
                              figsize = (figwidth, figheight) \
                            , facecolor = None \
                            , dpi = 300 \
                            )
    # MPP.ion() # "interactive mode". Might e useful here, but i don't know. Try to turn it off later.

# define axis spacing
    fig.subplots_adjust( \
                              top    = 0.98 \
                            , right  = 0.98 \
                            , bottom = 0.16 \
                            , left   = 0.16 \
                            , wspace = 0.10 # column spacing \
                            , hspace = 0.10 # row spacing \
                            )

# # a supertitle for the figure; relevant if multiple subplots
    # fig.suptitle( r"Falk's Reaktionszeiten, 17.-22. Oktober 2015" )

# define subplots
    gs = MP.gridspec.GridSpec( \
                                  len(rows) \
                                , len(cols) \
                                , height_ratios = rows \
                                , width_ratios = cols \
                                )


    return fig, gs


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
### Recording                                                       ###
#######################################################################
class ForceRecording(dict):

    def __init__(self, file = None):

        if file is None:
            raise IOError("no file given")

        self.files = {'force': file}

        self.LoadForces()

    def Copy(self):
        copy_object = ForceRecording(self.rec_nr)
        for key in self.keys():
            copy_object[key] = self[key].copy()

        return copy_object


    def LoadForces(self):

        # print (forceplate_info)

        self['force'] = PD.read_csv(self.files['force'], sep = ';').set_index('time', inplace = False)


        self.forceplate_info = LoadCalibrationData().loc[selected_forceplate, :]

        ### UNIT CONVERSION
        if convert_to_newtons:
            self.ConvertToNewtons()

        if convert_to_cartesian:
            self.CalculateCartesian()

        # print (self['force'])


    def ConvertToNewtons(self):
        self['force'] = VoltsToNewtons(self['force'], self.forceplate_info)

    def CalculateCartesian(self):
        force_formulae, input_parameters = ForceplateFormulae()
        force_cartesian = ConvertForceCoordinates(force_formulae, input_parameters, self.forceplate_info, self['force'].loc[:, measured_columns])
        for col in force_cartesian.columns:
            self['force'][col] = force_cartesian[col].values



    def __len__(self):
        return self['force'].shape[0]



    def ShowData(self, display_columns, rows = None, dimensions = [24, 18], show = False, suffix = '', figax = None, plot_kwargs = None):

        components = [comp for comp in display_columns.keys()]

        if not (figax is None):
            fig, ax = figax
            ref_ax = ax[components[0]]
        else: 
            if rows is None:
                rows = [1] * len(components)

            fig, gs = MakeFigure(rows = rows, cols = [1], dimensions = dimensions)

            ax = {}
            ref_ax = None
            for nr, comp in enumerate(components):
                if ref_ax is None:
                    ax[comp] = fig.add_subplot(gs[nr])
                    ref_ax = ax[comp]
                else:
                    ax[comp] = fig.add_subplot(gs[nr], sharex = ref_ax)

        time = self['force'].index.values

        plotargs = dict(  ls = '-' \
                        , lw = 1 \
                        , alpha = 0.6 \
                        )
        if plot_kwargs is not None:
            for arg, setting in plot_kwargs.items():
                plotargs[arg] = setting
            

        for comp in components:
            for col in display_columns[comp]:
                trace = self['force'].loc[:, col].values
                # trace = self.Filter(trace)
                
                ax[comp].plot( \
                                      time \
                                    , trace \
                                    , color = colors[col] \
                                    , label = r"$%s$%s" % (col, ' ' + suffix)  \
                                    ,**plotargs
                                    )

            PolishAx(ax[comp])
            ax[comp].set_ylabel('voltage' if not (convert_to_newtons) else '%s (N)' % ('moments' if comp == 'moments' else 'force'))

            if not (comp == components[-1]):
                # ax[comp].spines['bottom'].set_visible(False)
                # ax[comp].tick_params(bottom = False)
                ax[comp].get_xaxis().set_visible(False)

        ref_ax.set_xlim([NP.min(time), NP.max(time)])            
        ax[components[-1]].set_xlabel('time (s)')

        if figax is None:
            ax[components[0]].legend(loc = 1, fontsize = 6)

        if show:
            MPP.show()

        return (fig, ax)


    def Baseline(self, interval):
        # deduce baseline voltage
        time = self['force'].index.values
        for col in self['force'].columns:
            selection = NP.logical_and(time >= interval[0], time < interval[1])
            self['force'].loc[:, col] -= NP.nanmean(self['force'].loc[selection, col].values)


    def Cut(self, interval = None, flipstitch = False, cyclize = False):
        self['force'] = Cut(self['force'], interval = interval, flipstitch = flipstitch, cyclize = cyclize)



#######################################################################
### User Settings                                                   ###
#######################################################################

measured_columns = ['Fx12', 'Fx34', 'Fy14', 'Fy23', 'Fz1', 'Fz2', 'Fz3', 'Fz4']

show_columns = { \
              'forces': ['F_{x}', 'F_{y}', 'F_{z}'] \
            , 'moments': ['M_{x}', 'M_{y}', 'M_{z}'] \
          }
colors = {   'F_{x}': (0.5,0.5,0.9), 'F_{y}': (0.5,0.9,0.5), 'F_{z}': (0.9,0.5,0.5) \
            , 'M_{x}': (0.5,0.5,0.9), 'M_{y}': (0.5,0.9,0.5), 'M_{z}': (0.9,0.5,0.5) \
          }


forceplate_sensitivity = 5000
topmount_padding = (34.9) # mm 
selected_forceplate = 1
convert_to_newtons = True
convert_to_cartesian = True


#######################################################################
### Mission Control                                                 ###
#######################################################################
if __name__ == "__main__":
    # rec_raw = ForceRecording(file = 'references/trial1.csv')
    # rec_raw = ForceRecording(file = 'references/trial2.csv')
    rec_raw = ForceRecording(file = 'data/20220907_lizards_rec001_force.csv')
    rec_raw.Baseline(interval = [0., 1.])
    rec_raw.Cut(interval = [1.0, 3.0])


    rec_raw.ShowData(display_columns = show_columns, show = True, rows = [3, 1], plot_kwargs = {'ls': '-', 'lw': 0.5, 'alpha': 0.6, 'zorder': 10})

