import numpy as np
import scipy as sp
import quaternion as quat
import matplotlib
import matplotlib.pyplot as plt
import robot_navigation as rnav
from math import pi as pi
import os
import imageio
import imageio.plugins.pillow
import time

matplotlib.use('TkAgg')
parentDirectory = os.path.abspath(os.getcwd())


### SIMULATION PARAMETERS
#########################
dt = 1./100.
dt_INS = 1./100.
dt_UWB = 1./5.
dt_video = 1/20. 
dt_out = dt_UWB*25.
t_tot = 20.
steps = int(t_tot/dt)
steps_INS = int(t_tot/dt_INS)
steps_UWB = int(t_tot/dt_UWB)
steps_out = int(t_tot/dt_out)
np.random.seed(int(time.time()))

### TRAJECTORY GENERATION
#########################
def generate_unicycle_trajectory_straight(t_tot, p_start, p_end):
    #p_start/end as numpy arrays
    v = np.linalg.norm(p_start - p_end)/t_tot
    theta = np.arctan2(*list(p_end - p_start))
    w = 0.
    def trajectory_generator(t):
        p = p_start + v * np.array(np.cos(theta),np.sin(theta)) * t
        return p,v,theta,w
    return trajectory_generator
def generate_unicycle_trajectory_circle(A, f):
    #p_start/end as numpy arrays
    x = lambda t: A*np.cos(2*pi*f*t)
    y = lambda t: A*np.sin(2*pi*f*t)
    v = A*2*pi*f
    th = lambda t: pi/2 + (2*pi*f*t)# % (2*pi)
    w = 2*pi*f
    def trajectory_generator(t):
        return np.array((x(t),y(t))),v,th(t),w
    return trajectory_generator
p_start = np.array((-0.8,-0.8))
p_end = np.array((0.8,0.8))
circle_prop = {'A':0.8, 'f':0.1}
#traj_gen = generate_unicycle_trajectory_straight(t_tot, p_start, p_end)
traj_gen = generate_unicycle_trajectory_circle(**circle_prop)
p_des_uni_ta = np.zeros((steps,2))
v_des_uni_ta = np.zeros(steps)
theta_des_uni_ta = np.zeros(steps)
w_des_uni_ta = np.zeros(steps)
for i in range(steps):
    p_des_uni_ta[i] = traj_gen(dt*i)[0]
    v_des_uni_ta[i] = traj_gen(dt*i)[1]
    theta_des_uni_ta[i] = traj_gen(dt*i)[2]
    w_des_uni_ta[i] = traj_gen(dt*i)[3]

### INITIALIZATION
#########################
# Filter parameter
uwb_anchors_pos = np.array([
    [-5.,-1.,0.],
    [1.,-1.,0.],
    [1.,5.,0.],
    [-5,5.,0.]
]) # positions of the UWB tags
sigma_UWB = (0.1)
sigma_INS = np.array([0.06,0.06])
a = 0.9784 # sliding window fading coefficient, usually [0.95,0.99]
l = 12 # sliding window length
lamb = 1.012 # >1, parameter for the R innovation contribution weight
b = 0.9976 # forgetting factor of the R innovation contribution weight, usually [0.95,0.99]
alpha = 0.5 # secondary regulatory factor for R innovation
zeta = 4.5 #outliers detection treshold
# values for v0.5, w0.15
### SET 1 l20 (F) 4RMNC RNepsilon:  (a=0.986, l=20, lamb=1.004, b=0.989, alpha=0.856, zeta=55.4)
### SET 1 l12 (F) 4RMNC RNepsilon:  (a=0.9985, l=12, lamb=1.0504, b=0.992, alpha=0.4235, zeta=61.73)
### SET 1 l12 (F) 4RMNC RNepsilon:  (a=0.9784, l=12, lamb=1.0119, b=0.9976, alpha=0.4577, zeta=5.8)
# Unicycle control
uni = rnav.Unicycle(sigma_INS,sigma_UWB,uwb_anchors_pos,a,l,lamb,b,alpha,zeta)
uni.set_backstepping_gains(4,10,10,20,20)
v_uni_kin = 0.5
w_uni_kin = 0.15
# Initial values 
# uni.set_state(p_des_uni_ta[0],v_uni_kin,theta_des_uni_ta[0],w_uni_kin,0.,0.)
uni.set_state(p_des_uni_ta[0],v_uni_kin,pi,w_uni_kin,0.,0.)
p0,v0,q0,w0,vd0,wd0 = uni.return_as_3D_with_quat()
uni.navig.xn.p = p0.copy()+np.array((0.5,0.5,0.))
uni.navig.xn.v = v0.copy()
uni.navig.xn.q = q0.copy()
uni.navig.x_INS.p = uni.navig.xn.p.copy()
uni.navig.x_INS.v = uni.navig.xn.v.copy()
uni.navig.x_INS.q = uni.navig.xn.q.copy()
# Another navigator system for comparison without using fuzzy filter
navig2 = rnav.Navigator(sigma_INS,sigma_UWB,uwb_anchors_pos,a,l,lamb,b,alpha,zeta)
navig2.xn.p = uni.navig.xn.p.copy()
navig2.xn.v = uni.navig.xn.v.copy()
navig2.xn.q = uni.navig.xn.q.copy()
navig2.x_INS.p = uni.navig.xn.p.copy()
navig2.x_INS.v = uni.navig.xn.v.copy()
navig2.x_INS.q = uni.navig.xn.q.copy()


### DATA STORAGE and video
#########################
# Unicycle
save_video = True
if save_video:
    fig_anim = plt.figure(figsize=(8,8),dpi=96)
    ax_anim = fig_anim.add_subplot(111)
    writer = imageio.get_writer(os.path.join(parentDirectory, "videos/navigation_BOtuned_corr.mp4"), fps=int(1/dt_video))
error_uni_ta=np.zeros((steps,3))
state_uni_ta=np.zeros((steps,3))
state_d_uni_ta=np.zeros((steps,3))
# Navigation system
#NAVIGATION
p_navig_ta=np.zeros((steps_INS,3))
th_navig_ta=np.zeros((steps_INS,))
v_navig_ta=np.zeros((steps_INS,3))
w_navig_ta=np.zeros((steps_INS,))
#UWB
p_UWB_ta=np.zeros((steps_UWB,3))
v_UWB_ta=np.zeros((steps_UWB,3))
out_marked_ta=[]
out_marked_times=[]
out_true_ta=[]
out_true_times=[]
#INS ONLY (dead reckoning)
p_INS_ta=np.zeros((steps_INS,3))
th_INS_ta=np.zeros((steps_INS,))
v_INS_ta=np.zeros((steps_INS,3))
#NAVIGATOR 2
p_navig2_ta=np.zeros((steps_INS,3))
th_navig2_ta=np.zeros((steps_INS,))
v_navig2_ta=np.zeros((steps_INS,3))
w_navig2_ta=np.zeros((steps_INS,))

### SIMULATION
#########################
step_INS=0
step_UWB=0
t=0
for step in range(steps):
    #
    t = dt*step

    # Unicycle control
    uni.trajectory_tracking_backstepping_step1(dt,p_des_uni_ta[step],v_des_uni_ta[step],theta_des_uni_ta[step],w_des_uni_ta[step])
    tau_v, tau_th = uni.trajectory_tracking_backstepping_step2()
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
        # navig2
        navig2.INS_predict_xn_nominal_state(dt_INS,am_data,wm_data)
        navig2.INS_predict_x_INS(dt_INS,am_data,wm_data)
        navig2.am_prev=uni.navig.am_prev.copy()
        navig2.wm_prev=uni.navig.wm_prev.copy()
        navig2.am=uni.navig.am.copy()
        navig2.wm=uni.navig.wm.copy()
    if UWBbool:
        # UWB measurement
        if OUTbool:
            UWB_data = uni.navig.generate_UWB_measurement(p,interference=[-0.2,0.2],outlier=True, out_mag=2.)
        else:
            UWB_data = uni.navig.generate_UWB_measurement(p,interference=[-0.2,0.2])
        uni.navig.UWB_measurement(dt_UWB,UWB_data.copy())
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
            print("DETECTED")
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
        ####################################################################################################
        # SAME FOR NAVIG2
        navig2.UWB_measurement(dt_UWB,UWB_data.copy())
        navig2.compute_z()
        navig2.update_Q_TMP(dt_UWB)
        navig2.update_F(dt_UWB,navig2.am_prev.copy(),navig2.wm_prev.copy())
        navig2.predict_P_error_state_covariance(dt_UWB)
        # Update innovation
        navig2.update_epsilon_innovation()
        navig2.update_S_theoretical_innovation_covariance()
        navig2.update_Sn_estimated_innovation_covariance()
        # Outlier detection
        navig2.update_D_theoretical_zzT_expectation()
        outlier_detected2 = navig2.check_for_outlier()
        ##while or only an if?
        if (outlier_detected2):
            navig2.update_epsilon_innovation(hold=True)
            navig2.update_Sn_estimated_innovation_covariance()
            navig2.update_D_theoretical_zzT_expectation()
            ## only with while cycle:
            #outlier_detected = uni.navig.check_for_outlier()
        # Fuzzy filter
        navig2.apply_fuzzy_filter(no_fuzzy=True)
        # Estimate Measurement Noise Covariance
        navig2.update_Rn_theoretical_estimated_MNC()
        navig2.update_R_estimated_MNC()
        # Compute Kalman Gain and Update error state
        navig2.update_Kalman_gain()
        navig2.update_dx_and_P()
        # Update nominal state and reset
        navig2.update_xn()
        navig2.reset_dx_error_state()
    ###########
    # Save data
    error_uni_ta[step,:] = uni.e.copy()
    state_uni_ta[step,:2]= uni.p.copy()
    state_uni_ta[step,2]= uni.theta
    state_d_uni_ta[step,:2]= v[0:2]
    state_d_uni_ta[step,2]= uni.w
    if INSbool:
        p_navig_ta[step_INS,:] = uni.navig.xn.p.copy()
        th_navig_ta[step_INS] = (quat.as_rotation_vector(uni.navig.xn.q)[2])%(2*pi)
        v_navig_ta[step_INS] = uni.navig.xn.v.copy()
        p_INS_ta[step_INS,:] = uni.navig.x_INS.p.copy()
        th_INS_ta[step_INS] = (quat.as_rotation_vector(uni.navig.x_INS.q)[2])%(2*pi)
        v_INS_ta[step_INS] = uni.navig.x_INS.v.copy()
        # navig2
        p_navig2_ta[step_INS,:] = navig2.xn.p.copy()
        th_navig2_ta[step_INS] = (quat.as_rotation_vector(navig2.xn.q)[2])%(2*pi)
        v_navig2_ta[step_INS] = navig2.xn.v.copy()
    if UWBbool:
        p_UWB_ta[step_UWB,:] = uni.navig.pmUWB.copy()
        v_UWB_ta[step_UWB,:] = uni.navig.vmUWB.copy()
        if outlier_detected:
            out_marked_ta.append(uni.navig.pmUWB.copy())
            out_marked_times.append(t)
        if OUTbool:
            out_true_ta.append(uni.navig.pmUWB.copy())
            out_true_times.append(t)

    # Save frame
    if save_video:
        if video_bool:
            pass
            uni.draw_artists(fig_anim,ax_anim)
            #plt.plot(p_des_uni_ta[:(step-1),0],p_des_uni_ta[:(step-1),1],figure=fig_anim, color="xkcd:light teal")
            plt.plot(p_UWB_ta[:step_UWB,0],p_UWB_ta[:step_UWB,1],color="xkcd:orange", label="UWB", marker="1", linestyle="None")
            plt.plot([x[0] for x in out_true_ta],[x[1] for x in out_true_ta],color="xkcd:green", label="Outliers (injected)", marker="1", linestyle="None")
            plt.plot([x[0] for x in out_marked_ta],[x[1] for x in out_marked_ta],color="xkcd:black", label="Outliers (marked)", marker="o", fillstyle='none', linestyle="None")
            plt.plot(state_uni_ta[:step,0],state_uni_ta[:step,1],color="xkcd:teal", label="Real")
            plt.plot(p_navig_ta[:step_INS,0],p_navig_ta[:step_INS,1],color="xkcd:dark salmon", label="Estimated")
            plt.plot(p_INS_ta[:step_INS,0],p_INS_ta[:step_INS,1],color="xkcd:light salmon", ls='--',label="INS only (dead reckoning)")
            #
            ax_anim.set_aspect('equal')
            x_lim_TMP = (state_uni_ta[step_INS,0]-1.5,state_uni_ta[step_INS,0]+1.5)
            y_lim_TMP = (state_uni_ta[step_INS,1]-1.5,state_uni_ta[step_INS,1]+1.5)
            ax_anim.set(xlim=x_lim_TMP, ylim=y_lim_TMP)
            ax_anim.set_title("Position estimation")
            fig_anim.canvas.draw()
            img = np.frombuffer(fig_anim.canvas.tostring_rgb(), dtype='uint8')
            img  = img.reshape(fig_anim.canvas.get_width_height()[::-1]+(3,))
            writer.append_data(img)
            ax_anim.clear()
    
    # End step operations
    uni.step_simulation_KIN(dt,v_uni_kin,w_uni_kin)
    if not (step % int(dt_INS/dt)):
        uni.navig.iterations += 1
        navig2.iterations += 1
    if INSbool:
        step_INS +=1
    if UWBbool:
        step_UWB +=1

### PLOT
#########################
save_figs=False
fig_name_trj = os.path.join(parentDirectory, "graphs/trajectory_4.png")
fig_name_nav = os.path.join(parentDirectory, "graphs/navigation_4.png")
fig_name_comp = os.path.join(parentDirectory, "graphs/SHFAF_SHAF_comparison_4.png")
fig_size=(10,8)
### Unicycle control (useful for checking back-stepping behaviour)
# fig_uni,axs_uni = plt.subplots(2,2)
# axs_uni[0,0].plot(error_uni_ta[:,0], color="xkcd:light teal", label="x")
# axs_uni[0,0].plot(error_uni_ta[:,1], color="xkcd:dark teal", label="y")
# axs_uni[0,0].plot(error_uni_ta[:,2], color="xkcd:salmon", label="th")
# axs_uni[0,0].set_title("Error")
# axs_uni[0,1].plot(state_uni_ta[:,0], color="xkcd:light teal", label="x")
# axs_uni[0,1].plot(state_uni_ta[:,1], color="xkcd:dark teal", label="y")
# axs_uni[0,1].plot(state_uni_ta[:,2], color="xkcd:salmon", label="th")
# axs_uni[0,1].set_title("State")
# axs_uni[1,0].plot(p_des_uni_ta[:,0], color="xkcd:light teal", label="x")
# axs_uni[1,0].plot(p_des_uni_ta[:,1], color="xkcd:dark teal", label="y")
# axs_uni[1,0].plot(theta_des_uni_ta, color="xkcd:salmon", label="th")
# axs_uni[1,0].set_title("Desired")
# handles_uni, labels_uni = axs_uni[0,0].get_legend_handles_labels()
# fig_uni.legend(handles_uni, labels_uni, loc='center right')

### Trajectory
fig_trj,axs_trj = plt.subplots(figsize=fig_size)
axs_trj.plot(p_UWB_ta[:,0],p_UWB_ta[:,1],color="xkcd:orange", label="UWB meas.", marker="1", linestyle="None")
axs_trj.plot(
    [x[0] for x in out_true_ta],[x[1] for x in out_true_ta],
    color="xkcd:green", label="Outliers (injected)", marker="o", fillstyle='full', linestyle="None")
axs_trj.plot(
    [x[0] for x in out_marked_ta],[x[1] for x in out_marked_ta],
    color="xkcd:black", label="Outliers (marked)", marker="o", fillstyle='none', linestyle="None")
#axs_trj[0].plot(p_des_uni_ta[:,0],p_des_uni_ta[:,1],color="salmon", label="desired")
axs_trj.plot(p_navig_ta[:,0],p_navig_ta[:,1],color="xkcd:dark salmon", label="Estimated")
axs_trj.plot(p_INS_ta[:,0],p_INS_ta[:,1],color="xkcd:light salmon", ls='--', label="INS only (dead reckoning)")
axs_trj.plot(state_uni_ta[:,0],state_uni_ta[:,1],color="xkcd:teal", label="Real")
#axs_trj.plot(p_navig2_ta[:,0],p_navig2_ta[:,1],color="xkcd:dark yellow", label="SHAF")
axs_trj.set(xlabel="X [m]",ylabel="Y [m]",title="Trajectory [x-y]",xlim=(-4,2),ylim=(-8,2),aspect='equal')
axs_trj.set_aspect('equal')
handles_trj, labels_trj = axs_trj.get_legend_handles_labels()
fig_trj.legend(handles_trj, labels_trj, loc='upper right')
if save_figs:
    fig_trj.savefig(fig_name_trj)

### Navigation 
UNI_ta = np.linspace(0,t_tot,state_uni_ta.shape[0])
INS_ta = np.linspace(0,t_tot,p_navig_ta.shape[0])
UWB_ta = np.linspace(0,t_tot,p_UWB_ta.shape[0])
fig_nav,axs_nav = plt.subplots(2,2,figsize=fig_size)
axs_nav[0,0].plot(UWB_ta,p_UWB_ta[:,0],color="xkcd:orange", label="UWB meas.", marker="1", linestyle="None")
axs_nav[0,0].plot(
    out_true_times, [x[0] for x in out_true_ta],
    color="xkcd:green", label="Outliers", marker="o", fillstyle='full', linestyle="None")
axs_nav[0,0].plot(
    out_marked_times, [x[0] for x in out_marked_ta],
    color="xkcd:black", label="Outliers", marker="o", fillstyle='none', linestyle="None")
axs_nav[0,0].plot(INS_ta,p_navig_ta[:,0],color="xkcd:dark salmon", label="Estimated")
axs_nav[0,0].plot(INS_ta,p_INS_ta[:,0],color="xkcd:light salmon",ls='--', label="INS only (dead reckoning)")
axs_nav[0,0].plot(UNI_ta,state_uni_ta[:,0],color="xkcd:teal", label="Real")
axs_nav[0,0].set(xlabel="Time [s]",ylabel="X [m]",title='X pos')
#
axs_nav[0,1].plot(
    out_true_times, [x[1] for x in out_true_ta],
    color="xkcd:green", label="Outliers", marker="o", fillstyle='full', linestyle="None")
axs_nav[0,1].plot(
    out_marked_times, [x[1] for x in out_marked_ta],
    color="xkcd:black", label="Outliers", marker="o", fillstyle='none', linestyle="None")
axs_nav[0,1].plot(UWB_ta,p_UWB_ta[:,1],color="xkcd:orange", label="UWB meas.", marker="1", linestyle="None")
axs_nav[0,1].plot(UNI_ta,state_uni_ta[:,1],color="xkcd:teal", label="Real")
axs_nav[0,1].plot(INS_ta,p_navig_ta[:,1],color="xkcd:dark salmon", label="Estimated")
axs_nav[0,1].plot(INS_ta,p_INS_ta[:,1],color="xkcd:light salmon",ls='--', label="INS only (dead reckoning)")
axs_nav[0,1].set(xlabel="Time [s]",ylabel="Y [m]", title='Y pos')
# axs_nav[0,2].plot(UNI_ta,state_uni_ta[:,2],color="xkcd:teal", label="Real")
# axs_nav[0,2].plot(INS_ta,th_navig_ta,color="xkcd:dark salmon", label="Estimated")
# axs_nav[0,2].plot(INS_ta,th_navig_ta,color="xkcd:light salmon", label="INS only (dead reckoning)")
# axs_nav[0,2].set_title("Theta")
#
axs_nav[1,0].plot(UWB_ta,v_UWB_ta[:,0],color="xkcd:orange", label="UWB meas.", marker="1", linestyle="None")
axs_nav[1,0].plot(INS_ta,v_navig_ta[:,0],color="xkcd:dark salmon", label="Estimated")
axs_nav[1,0].plot(INS_ta,v_INS_ta[:,0],color="xkcd:light salmon",ls='--', label="INS only (dead reckoning)")
axs_nav[1,0].plot(UNI_ta,state_d_uni_ta[:,0],color="xkcd:teal", label="Real")
axs_nav[1,0].set(xlabel="Time [s]",ylabel="Vx [m/s]", title='Vel x')
axs_nav[1,1].plot(UWB_ta,v_UWB_ta[:,1],color="xkcd:orange", label="UWB meas.", marker="1", linestyle="None")
axs_nav[1,1].plot(INS_ta,v_navig_ta[:,1],color="xkcd:dark salmon", label="Estimated")
axs_nav[1,1].plot(INS_ta,v_INS_ta[:,1],color="xkcd:light salmon",ls='--', label="INS only (dead reckoning)")
axs_nav[1,1].plot(UNI_ta,state_d_uni_ta[:,1],color="xkcd:teal", label="Real")
axs_nav[1,1].set(xlabel="Time [s]",ylabel="Vy [m/s]", title='Vel y')
# axs_nav[1,2].plot(UNI_ta,state_d_uni_ta[:,2],color="xkcd:teal", label="actual")
# axs_nav[1,2].plot(INS_ta,w_navig_ta,color="xkcd:salmon", label="estimated")
# axs_nav[1,2].set_title("W (ang.speed)")
handles_nav, labels_nav = axs_nav[0,0].get_legend_handles_labels()
fig_nav.legend(handles_nav, labels_nav, loc='upper right')
if save_figs:
    fig_nav.savefig(fig_name_nav)

### Comparison SHFAF-SHAF
fig_comp, axs_comp = plt.subplots(figsize=(9,5))
axs_comp.plot(INS_ta,np.linalg.norm((state_uni_ta[:,0:2]-p_navig_ta[:,0:2]),axis=1), label="SHFAF", color="xkcd:teal")
axs_comp.plot(INS_ta,np.linalg.norm((state_uni_ta[:,0:2]-p_navig2_ta[:,0:2]),axis=1), label="SHAF", color="xkcd:salmon", linestyle='--')
#axs_comp.set_title("Performance comparison")
axs_comp.set(xlabel="Time [s]",ylabel="Position error [m]", title="Performance comparison")
handles_comp, labels_comp = axs_comp.get_legend_handles_labels()
fig_comp.legend(handles_comp, labels_comp, loc='upper right')
if save_figs:
    fig_comp.savefig(fig_name_comp)

###
plt.show()

### END ROUTINE
#########################
if save_video:
    writer.close()

#print(p_navig_ta[1900:2000])