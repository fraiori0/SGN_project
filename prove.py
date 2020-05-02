#from math import *
import numpy as np
import scipy as sp
import quaternion as quat
import matplotlib
import matplotlib.pyplot as plt
import robot_navigation as rnav
matplotlib.use('TkAgg')

###
uwb_anchors_pos = np.array([
    [-5.,-1.,0.],
    [1.,-1.,0.],
    [1.,5.,0.],
    [-5,5.,0.],
])
sigma_UWB = 0.1
sigma_INS = np.array([0.06,0.06])
navig = rnav.Navigator(sigma_INS,sigma_UWB,uwb_anchors_pos,0.98,10,2.,0.98,1.,4.)
###
steps = 20
#
TMP = np.array((0.,0.,0.))
pUWB_mes_TA = np.zeros((steps,3))
vUWB_mes_TA = np.zeros((steps,3))
epsilon_TA = np.zeros((steps,6))
z_TA = np.zeros((steps,6))
xn_p_TA = np.zeros((steps,3))
xn_v_TA = np.zeros((steps,3))
dx_p_TA = np.zeros((steps,3))
dx_v_TA = np.zeros((steps,3))
dx_p_TA_pre = np.zeros((steps,3))
dx_v_TA_pre = np.zeros((steps,3))
#
for i in range(steps):
    dt = 1/10.

    UWB_data = navig.generate_UWB_measurement(np.array((0.,0.,0.)))
    #UWB_data = np.random.random((4,))
    am_data,wm_data = navig.generate_INS_measurement(np.array((0.,0.,0.)),np.array((0.,0.,0.)))
    #
    navig.INS_predict_xn_nominal_state(dt,am_data,wm_data)
    #
    navig.UWB_measurement(dt,UWB_data)
    #
    navig.update_Q(dt)
    navig.update_F(dt,am_data,wm_data)
    navig.predict_dx_error_state()
    navig.predict_P_error_state_covariance(dt)
    dx_p_TA_pre[i] = navig.dx.p
    dx_v_TA_pre[i] = navig.dx.v
    #
    navig.update_epsilon_innovation()
    navig.update_Sn_estimated_innovation_covariance()
    navig.update_Rn_theoretical_estimated_MNC()
    #
    navig.update_D_theoretical_zzT_expectation()
    outlier_detected = navig.check_for_outlier()
    #
    # while or only an if?
    if (outlier_detected):
        print("DETECTED")
        navig.update_epsilon_innovation()
        navig.update_Sn_estimated_innovation_covariance()
        navig.update_Rn_theoretical_estimated_MNC()
        navig.update_D_theoretical_zzT_expectation()
        outlier_detected = navig.check_for_outlier()
    #
    navig.apply_fuzzy_filter()
    #
    navig.update_R_estimated_MNC()
    print(np.round(np.linalg.norm(navig.epsilon[-1]),3))
    print(np.round(navig.H.dot(navig.dxapp),2))
    #
    navig.update_Kalman_gain()
    navig.update_dx_and_P()
    ##
    xn_p_TA[i] = navig.xn.p
    xn_v_TA[i] = navig.xn.v
    dx_p_TA[i] = navig.dx.p
    dx_v_TA[i] = navig.dx.v
    ##
    navig.update_xn()
    #
    TMP = TMP + navig.dx.p
    navig.reset_dx_error_state()
    #
    navig.iterations+=1
    ###
    pUWB_mes_TA[i] = navig.pmUWB
    vUWB_mes_TA[i] = navig.vmUWB
    epsilon_TA[i] = navig.epsilon[-1]
    z_TA[i] = navig.z
    


print(navig.xn.p)
print(navig.xn.v)
# print(navig.Rn)
# print(np.round(sp.linalg.pinv(navig.H.dot(navig.P.dot(navig.H.T))),3))
# print(np.round(navig.R,3))
# print(np.round(navig.K,3))
# print(navig.sigma)
# print(np.round())
##########################
# fig, axs = plt.subplots(1,3)
# axs[0].plot(pUWB_mes_TA[:,0],color="xkcd:teal")
# axs[0].plot(vUWB_mes_TA[:,0],color="xkcd:light teal")
fig_2, axs_2 = plt.subplots(2,3)
xn_fusion_TA = np.hstack((xn_p_TA,xn_v_TA))
dx_fusion_TA = np.hstack((dx_p_TA,dx_v_TA))
dx_fusion_TA_pre = np.hstack((dx_p_TA_pre,dx_v_TA_pre))
for i in range(6):
    axs_2[i//3,i%3].plot(epsilon_TA[:,i])
    #axs_2[i//3,i%3].plot(z_TA[:,i])
    axs_2[i//3,i%3].plot(xn_fusion_TA[:,i])
    axs_2[i//3,i%3].plot(dx_fusion_TA[:,i],color='xkcd:salmon')
    axs_2[i//3,i%3].plot(dx_fusion_TA_pre[:,i],color='xkcd:teal')
plt.show()