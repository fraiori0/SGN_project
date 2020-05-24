import numpy as np
import scipy as sp
import quaternion as quat
import matplotlib
import matplotlib.pyplot as plt
import robot_navigation as rnav
from math import pi as pi
import os
import operator
import random
import hyperopt as hp
import pandas as pd
import pickle


matplotlib.use('TkAgg')
parentDirectory = os.path.abspath(os.getcwd())

def simulate(params, t_tot=30, folds=7):
    #paramas = dict(a, lamb, b, alpha, zeta)
    dt = 1./100.
    dt_INS = 1./100.
    dt_UWB = 1./5.
    dt_video = 1/20. 
    dt_out = dt_UWB*25.
    steps = int(t_tot/dt)
    steps_INS = int(t_tot/dt_INS)
    steps_UWB = int(t_tot/dt_UWB)
    steps_out = int(t_tot/dt_out)
    np.random.seed
    #########################################################
    # Filter parameter
    uwb_anchors_pos = np.array([
        [-5.,-1.,0.],
        [1.,-1.,0.],
        [1.,5.,0.],
        [-5,5.,0.]
    ]) # positions of the UWB tags
    sigma_UWB = (0.1)
    sigma_INS = np.array([0.06,0.06])
    l = 20 #sliding window length
    #zeta = 52. #outliers detection treshold
    # Unicycle control
    uni = rnav.Unicycle(sigma_INS,sigma_UWB,uwb_anchors_pos,l=l,**params)
    uni.set_backstepping_gains(4,10,10,20,20)
    v_uni_kin = 0.5
    w_uni_kin = 0.15
    # Initial values 
    uni.set_state(np.array((0.,0.)),v_uni_kin,pi,w_uni_kin,0.,0.)
    p0,v0,q0,w0,vd0,wd0 = uni.return_as_3D_with_quat()
    uni.navig.xn.p = p0.copy()+np.array((0.5,0.5,0.))
    uni.navig.xn.v = v0.copy()
    uni.navig.xn.q = q0.copy()
    uni.navig.x_INS.p = uni.navig.xn.p.copy()
    uni.navig.x_INS.v = uni.navig.xn.v.copy()
    uni.navig.x_INS.q = uni.navig.xn.q.copy()
    #########################################################
    error_k = np.zeros((folds,))
    #########################################################
    for k in range(folds):
        state_uni_ta=np.zeros((steps,3))
        p_navig_ta=np.zeros((steps_INS,3))
        ###
        for step in range(steps):
            #
            t = dt*step
            # Unicycle control
            p,v,q,w,vd,wd = uni.return_as_3D_with_quat()
            R_bg = (quat.as_rotation_matrix(q)).T # global to body
            vd_body = R_bg.dot(vd) # INS generate measurement in body frame coordinates
            w_body = R_bg.dot(w)
            # Sub-system timing
            INSbool = not (step % int(dt_INS/dt)) #module division
            UWBbool = not (step % int(dt_UWB/dt))
            OUTbool = not (step % int(dt_out/dt))
            video_bool = not (step % int(dt_video/dt))
            step_video= step // int(dt_video/dt) #floor division
            step_INS = step // int(dt_INS/dt)
            step_UWB = step // int(dt_UWB/dt)
            ##########
            # FILTER #
            ##########
            if INSbool:
                # INS prediction
                am_data,wm_data = uni.navig.generate_INS_measurement(vd_body,w_body)
                uni.navig.INS_predict_xn_nominal_state(dt_INS,am_data,wm_data)
                uni.navig.INS_predict_x_INS(dt_INS,am_data,wm_data)
            if UWBbool:
                # UWB measurement
                if OUTbool:
                    UWB_data = uni.navig.generate_UWB_measurement(p,interference=[-0.2,0.2],outlier=True, out_mag=2.)
                else:
                    UWB_data = uni.navig.generate_UWB_measurement(p,interference=[-0.2,0.2])
                uni.navig.UWB_measurement(dt_UWB,UWB_data)
                uni.navig.compute_z()
                #
                uni.navig.update_Q_TMP(dt_UWB)
                #uni.navig.update_Q(dt_UWB)
                uni.navig.update_F(dt_UWB,uni.navig.am_prev.copy(),uni.navig.wm_prev.copy())
                ###uni.navig.predict_dx_error_state()
                uni.navig.predict_P_error_state_covariance(dt_UWB)
                # Update innovation
                uni.navig.update_epsilon_innovation()
                uni.navig.update_S_theoretical_innovation_covariance()
                uni.navig.update_Sn_estimated_innovation_covariance()
                # Outlier detection
                uni.navig.update_D_theoretical_zzT_expectation()
                outlier_detected = uni.navig.check_for_outlier()
                ##while or only an if?
                if (outlier_detected):
                    #print("DETECTED")
                    uni.navig.update_epsilon_innovation(hold=True)
                    uni.navig.update_Sn_estimated_innovation_covariance()
                    uni.navig.update_D_theoretical_zzT_expectation()
                    ## only with while cycle:
                    #outlier_detected = uni.navig.check_for_outlier()
                # Fuzzy filter
                uni.navig.apply_fuzzy_filter(no_fuzzy=False)
                # Estimate Measurement Noise Covariance
                uni.navig.update_Rn_theoretical_estimated_MNC()
                #uni.navig.R = uni.navig.Rn.copy()
                uni.navig.update_R_estimated_MNC()
                # Compute Kalman Gain and Update error state
                uni.navig.update_Kalman_gain()
                uni.navig.update_dx_and_P()
                # Update nominal state and reset
                uni.navig.update_xn()
                uni.navig.reset_dx_error_state()
            # Save data
            state_uni_ta[step,:2]= uni.p.copy()
            if INSbool:
                p_navig_ta[step_INS,:] = uni.navig.xn.p.copy()
            # End step operations
            #uni.step_simulation(dt,tau_v,tau_th)
            uni.step_simulation_KIN(dt,v_uni_kin,w_uni_kin)
            if not (step % int(dt_INS/dt)):
                uni.navig.iterations += 1
            #print(np.round(100*step/steps,2),'%')
            if INSbool:
                step_INS +=1
            if UWBbool:
                step_UWB +=1
            #print('%f' %(np.round(100*step/steps, 1)))
        ###
        index_TMP=[x for x in range(0,steps, (steps//steps_INS))]
        error_nav_ta = state_uni_ta[index_TMP,:2] - p_navig_ta[:,:2]
        error_norm_ta = np.linalg.norm(error_nav_ta,axis=1)
        error_squared = error_norm_ta**2
        error_k[k] = dt*np.sum(error_squared)
    error_mean = np.mean(error_k)
    error_std = np.std(error_k)
    loss_k = -1/((np.log(1 + error_k)/(np.log(1000)))**2)
    print(np.round(loss_k,3))
    print(params)
    loss_std = np.std(loss_k)
    loss_mean = -1/((np.log(1 + error_mean)/(np.log(1000)))**2)
    return {'loss':loss_mean, 'loss_variance':loss_std, 'status': hp.STATUS_OK}

def Bayesian_tuning(param_dist,points_start,iterations=10, save=True, filename='tuning_trial', retrieve_trial=False,save_trial=False):
    if retrieve_trial:
        trials = pickle.load(open("./hyperopt_tuning/"+filename+".p", "rb"))
    else:
        trials = hp.Trials()
    best_param = hp.fmin(
        simulate,
        param_dist,
        algo=hp.tpe.suggest,
        max_evals=iterations,
        points_to_evaluate=points_start,
        trials=trials,
    )
    best_param_values = [val for val in best_param.values()]
    losses = [x['result']['loss'] for x in trials.trials]
    vals = [x['misc']['vals']for x in trials.trials]
    std_devs = [x['result']['loss_variance'] for x in trials.trials]
    for val, std_dev, loss in zip(vals,std_devs,losses):
        print('Loss: %f (%f)   Param:%s' %(loss,std_dev,val))
        best_param_values = [x for x in best_param.values()]
    print("Best loss obtained: %f\n with parameters: %s" % (min(losses), best_param_values))
    if save:
        dict_csv={}
        dict_csv.update({'Score' : []})
        dict_csv.update({'Std Dev' : []})
        for key in vals[0]:
            dict_csv.update({key : []})
        for index,val in enumerate(vals):
            for key in val:
                dict_csv[key].append((val[key])[0])
            dict_csv['Score'].append(losses[index])
            dict_csv['Std Dev'].append(std_devs[index])
        df = pd.DataFrame.from_dict(dict_csv, orient='columns')
        df.to_csv(path_or_buf=("./hyperopt_tuning/"+filename+'.csv'),sep=',', index_label='Index')
    if save_trial:
        pickle.dump(trials, open("./hyperopt_tuning/"+filename+".p", "wb"))
    return trials

### SIMULATE
#########################
# params={
#     'a':    0.99999,
#     'lamb': 1.00001,
#     'b':    0.999,
#     'alpha':0.434,
#     'zeta': 52.
# }
params={
    'a':    hp.hp.uniform('a',0.96,1),
    'lamb': hp.hp.loguniform('lamb',np.log(1.),np.log(1.5)),
    'b':    hp.hp.uniform('b',0.96,1),
    'alpha':hp.hp.uniform('alpha',0.2,0.9),
    'zeta': hp.hp.uniform('zeta',2,10)
}
points_start=[
    {'a':0.99999, 'lamb':1.00001, 'b':0.999, 'alpha':0.434, 'zeta':52.},
    {'a':0.984, 'lamb':1.000065, 'b':0.9815, 'alpha':0.267, 'zeta':55.},
    {'a':0.9942, 'lamb':1.00112, 'b':0.9938, 'alpha':0.482, 'zeta':65},
    {'a':0.99154, 'lamb':1.0280, 'b':0.99617, 'alpha':0.79934, 'zeta':56.69},
    {'a':0.9829, 'lamb':1.0472, 'b':0.9865, 'alpha':0.6021, 'zeta':62.6},
    {'a': 0.9841, 'lamb': 1.00212, 'b': 0.9972, 'alpha': 0.6054, 'zeta': 67.43}
]
Bayesian_tuning(
    params,
    points_start,
    iterations=12,
    save=True,
    filename='fuzzy_norm2_l20_R4MNC_RNepsilon_run2',
    retrieve_trial=False,
    save_trial=True
    )