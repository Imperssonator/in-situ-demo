#!/user/bin/env python

# Basic
import numpy as np
import pandas as pd

# Plotting
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True,
                 'xtick.top': True,
                 'xtick.direction': 'in',
                 'ytick.right': True,
                 'ytick.direction': 'in',
                 'font.sans-serif': 'Arial',
                 'font.size': 14,
                 'figure.dpi': 96
                })

# Peak Fitting, optimization
from lmfit.models import GaussianModel, LorentzianModel, ExponentialModel, ConstantModel, LinearModel, VoigtModel

def fit_peaks(x, y, df_peak_init, fix=[], method='leastsq'):
    
    df_new = df_peak_init.copy()
    out_dict = {}
    
    # Run peak fits for each set of peaks
    for i in df_new['set'].unique():
        
        df_peaks = df_new.loc[df_new.set==i]

        # Build bounded x and y vectors
        lb = df_peaks['fit_lb'].iloc[0]
        ub = df_peaks['fit_ub'].iloc[0]
        lb_ind = int(np.where(x>=lb)[0][0])
        ub_ind = int(np.where(x>=ub)[0][0])
        xb = x[lb_ind:ub_ind]
        yb = y[lb_ind:ub_ind]

        # Initialize Baseline model
        comp_mod = []
        
        if df_peaks['bg'].iloc[0]=='lin':
            bg_mod = LinearModel(prefix='bg_')
            pars = bg_mod.make_params(m=0,b=yb.min())
            comp_mod.append(bg_mod)
        elif df_peaks['bg'].iloc[0]=='exp':
            bg_mod = ExponentialModel(prefix='bg_')
            pars = bg_mod.guess(y,x=x)
            comp_mod.append(bg_mod)
        else:
            bg_mod = ConstantModel(prefix='bg_')
            pars = bg_mod.make_params(c=yb.min())
            comp_mod.append(bg_mod)

        # Add peaks
        for index, peak in df_peaks.iterrows():
            prefix = peak['name']+'_'
            
            # Select peak model
            if peak['model']=='voigt':
                peak_temp  = VoigtModel(prefix=prefix)
            elif peak['model']=='gauss':
                peak_temp = GuassianModel(prefix=prefix)
            else:
                peak_temp = GuassianModel(prefix=prefix)

            # Set peak parameter guesses + vary or fix
            pars.update(peak_temp.make_params())
            param_guesses = [p.split('_')[0] for p in peak.index if 'guess' in p]
#             print(param_guesses)
            
            for p in param_guesses:
                pars[prefix+p].set(peak[p+'_guess'])
                
                if p+'_lb' in peak.index:
                    pars[prefix+p].set(min=peak[p+'_lb'])
                if p+'_ub' in peak.index:
                    pars[prefix+p].set(max=peak[p+'_ub'])
                
                if p in fix:
#                     print('fixing', p)
                    pars[prefix+p].set(vary=False)
            
            # No negative peaks
            pars[prefix+'amplitude'].set(min=0)
            
            # Add peak to the composite model
            comp_mod.append(peak_temp)

        # Build composite model
        comp_mod = np.sum(comp_mod)
        out = comp_mod.fit(yb, pars, x=xb, method=method)
        params_dict=out.params.valuesdict()

        # Store peak features in original dataframe
        for peak, prop in [s.split('_') for s in list(params_dict.keys())]:
            df_new.loc[df_new.name==peak,prop] = params_dict[peak+'_'+prop]
        out_dict[i] = out
        
    return df_new, out_dict


def plot_peak_fits(x, y, df_peaks, out_dict, log_scale=True, ax=None):

    color_list = ['#3cb44b','#0082c8','#f58231','#911eb4','#800000','#000080','#808000']
    
    if ax is None:
        f1 = plt.figure(figsize=(5,4))
        ax = plt.gca()
    
    if log_scale:
        ax.semilogy(x, y, 'r-')
    else:
        ax.plot(x, y, 'r-')
        
    for i in df_peaks['set'].unique():
        
        out = out_dict[i]
        df_set = df_peaks.loc[df_peaks.set==i]
        
        # Build bounded x and y vectors
        lb = df_set['fit_lb'].iloc[0]
        ub = df_set['fit_ub'].iloc[0]
        lb_ind = int(np.where(x>=lb)[0][0])
        ub_ind = int(np.where(x>=ub)[0][0])
        xb = x[lb_ind:ub_ind]
        yb = y[lb_ind:ub_ind]

        # Add peaks to plot
        comps = out.eval_components(x=xb)
        
        if log_scale:
            ax.semilogy(xb, out.eval(x=xb), '-', color = color_list[i])
            for c in comps:
                try:
                    ax.semilogy(xb, comps[c], '--', color = color_list[i])
                except:
                    const_eval = np.ones(xb.shape)*comps[c]
                    ax.semilogy(xb, const_eval, '--', color = color_list[i])
        else:
            ax.plot(xb, out.eval(x=xb), '-', color = color_list[i])
            for c in comps:
                try:
                    ax.plot(xb, comps[c], '--', color = color_list[i])
                except:
                    const_eval = np.ones(xb.shape)*comps[c]
                    ax.plot(xb, const_eval, '--', color = color_list[i])

    ax.set_ylim( bottom=max(1,np.min(y)/10), top=np.max(y)*1.3 )
    
    return ax