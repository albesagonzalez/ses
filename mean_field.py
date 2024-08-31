import numpy as np


def get_mean_field_terms(t, post_i, pre_j, sp, p):
    T_ax_free = sp["eta_ax"]*sp["w_max"]/(sp["lmbda"]p["j"][pre_j])
    T_de_free = sp["eta_de"]*sp["w_max"]/(sp["lmbda"]p["i"][post_i])


    if (t < T_ax_free) and (t < T_de_free):
        S_de = sp["lmbda"]*sp["N_pre_act"]*p["i"][post_i]*t
        S_ax = sp["lmbda"]*sp["N_post_act"]*p["j"][pre_j]*t
        w = sp["lmbda"]*p["ij"][post_i, pre_j]*t
        fp_w = None

    elif  (t < T_ax_free) and (t > T_de_free):
        #T_ax_cond = (sp["N_post_act"]*p["j"][pre_j]*t - e_de_to_ax[post_i]*p["i"][post_i])/(t - T_de_free) + T_de_free
        if (t < T_ax_cond):
            S_ax = sp["N_post_act"]*p["j"][pre_j]*t - e_de_to_ax[post_i]*p["i"][post_i])*(t - T_de_free)
            S_de = sp["w_de_max"]
            tau_w = 1/(e_ax[pre_j]*p["j"][pre_j])
            fp_w = tau_w*sp["lmbda"]*p["ij"][post_i, pre_j]
            w_0 = sp["lmbda"]*p["ij"][post_i, pre_j]*T_de_free
            beta = 1 - np.exp(-(t - T_ax_free)/tau_w)
            w = (1 - beta)* + beta*fp_w

        else:
            S_ax = sp["w_ax_max"]
            S_de = sp["w_de_max"]
            tau_w = 1/(e_ax[pre_j]*p["j"][pre_j] + e_de[post_i]*p["i"][post_i])
            fp_w = tau_w*sp["lmbda"]*p["ij"][post_i, pre_j]
            beta = 1 - np.exp(-(t - T_ax_free)/tau_w)
            w = (1 - beta)*sp["lmbda"]*p["ij"][post_i, pre_j]*T_ax_free + beta*fp_w

    elif  (t > T_ax_free) and (t < T_de_free):
        T_de_cond = (sp["N_pre_act"]*p["i"][post_i]*t - e_ax_to_de[pre_j]*p["j"][pre_j])/(t - T_ax_free) + T_ax_free
        if (t < T_ax_cond):
            S_ax_dt = sp["w_ax_max"]
            S_de_dt = sp["N_pre_act"]*p["i"][post_i]*t - e_ax_to_de[pre_j]*p["j"][pre_j])*(t - T_ax_free)
            w_0 = sp["lmbda"]*p["ij"][post_i, pre_j]*T_de_free 
            tau_w = 1/(e_ax[pre_j]*p["j"][pre_j])
            fp_w = tau_w*sp["lmbda"]*p["ij"][post_i, pre_j]
            beta = 1 - np.exp(-(t - T_ax_free)/tau_w)
            w = (1 - beta)w_0 + beta*fp_w

        else:
            S_ax = sp["w_ax_max"]
            S_de = sp["w_de_max"]
            w_0 =  1/(e_ax[pre_j]*p["j"][pre_j])*sp["lmbda"]*p["ij"][post_i, pre_j]
            tau_w = 1/(e_ax[pre_j]*p["j"][pre_j] + e_de[post_i]*p["i"][post_i])
            fp_w = tau_w*sp["lmbda"]*p["ij"][post_i, pre_j]
            beta = 1 - np.exp(-(t - T_ax_free - T_de_cond)/tau_w)
            w = (1 - beta)*w_0 + beta*fp_w
        
