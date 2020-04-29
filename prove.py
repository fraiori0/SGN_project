#from math import *
import numpy as np
import scipy as sp
import quaternion as quat
import matplotlib.pyplot as pyplot
import robot_navigation as rnav

v=np.array((1,3,2))
vq = rnav.vec_to_0quat(v)

diag=(sp.linalg.block_diag(np.eye(3)*1.**2,   np.eye(3)*2.**2,  np.eye(3)*3.**2,  np.eye(3)*4.**2)) + np.array(list(range(12)))


uwb_anchors_pos = np.array([
    [1.,2.,3.],
    [1.,1.,1.],
    [0.,-4.,3.],
    [0.5,2.,1.],
    [3.,2.,0.2],
    [2.,0.,2.]
])

navig = rnav.Navigator([0.4,0.2,0.15,0.01],uwb_anchors_pos,0.96,10,0.5,0.96,1,2.)

for i in range(40):
    dt = 1/10.

    UWB_data = (3. - (-3.))*np.random.random((6,)) -3. 
    am_data = (1. - (-1.))*np.random.random((3,)) -1.
    wm_data = (1. - (-1.))*np.random.random((3,)) -1.

    navig.INS_predict_xn_nominal_state(dt,am_data,wm_data)

    navig.UWB_measurement(dt,UWB_data)

    navig.update_F(dt,am_data,wm_data)
    navig.predict_dx_error_state()
    navig.predict_P_error_state_covariance(dt)

    navig.update_epsilon_innovation()
    navig.update_Sn_estimated_innovation_covariance()
    navig.update_Rn_theoretical_estimated_MNC()

    navig.update_D_theoretical_zzT_expectation()
    outlier_detected = navig.check_for_outlier()

    # while or only an if?
    if (outlier_detected):
        navig.update_epsilon_innovation()
        navig.update_Sn_estimated_innovation_covariance()
        navig.update_Rn_theoretical_estimated_MNC()
        navig.update_D_theoretical_zzT_expectation()
        outlier_detected = navig.check_for_outlier()

    navig.apply_fuzzy_filter()

    navig.update_R_estimated_MNC()

    navig.update_Kalman_gain()
    navig.update_dx_and_P()
    navig.update_xn()

    navig.reset_dx_error_state()

    navig.iterations+=1


print(np.round(navig.K,2))
print(navig.K.shape)
# print(navig.sigma)
# print(np.round())