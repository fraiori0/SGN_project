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

matplotlib.use('TkAgg')
parentDirectory = os.path.abspath(os.getcwd())


### SIMULATION PARAMETERS
#########################
dt = 1./200.
dt_INS = 1./100.
dt_UWB = 1./5.
dt_video = 1/20. 
t_tot = 50.
steps = int(t_tot/dt)
steps_INS = int(t_tot/dt_INS)
steps_UWB = int(t_tot/dt_UWB)
save_video = False

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
    [-5,5.,0.],
]) # positions of the UWB tags
sigma_UWB = (0.1)
sigma_INS = np.array([0.06,0.06])
a = 0.999 #sliding window fading coefficient, usually [0.95,0.99]
l = 30 #sliding window length
lamb = 1.1 # >1, parameter for the R innovation contribution weight
b = 0.999 #forgetting factor of the R innovation contribution weight, usually [0.95,0.99]
alpha = 0.1 #secondary regulatory factor for R innovation
zeta = 3. #outliers detection treshold
# Unicycle control
uni = rnav.Unicycle(sigma_INS,sigma_UWB,uwb_anchors_pos,a,l,lamb,b,alpha,zeta)
uni.set_backstepping_gains(4,10,10,20,20)
v_uni_kin = 0.4
w_uni_kin = 0.1
# Initial values 
# uni.set_state(p_des_uni_ta[0],v_uni_kin,theta_des_uni_ta[0],w_uni_kin,0.,0.)
uni.set_state(p_des_uni_ta[0],v_uni_kin,pi,w_uni_kin,0.,0.)
p0,v0,q0,w0,vd0,wd0 = uni.return_as_3D_with_quat()
uni.navig.xn.p = p0.copy()
uni.navig.xn.v = v0.copy()
uni.navig.xn.q = q0.copy()
uni.navig.x_INS.p = p0.copy()
uni.navig.x_INS.v = v0.copy()
uni.navig.x_INS.q = q0.copy()


### DATA STORAGE and video
#########################
# Unicycle
if save_video:
    fig_anim = plt.figure()
    ax_anim = fig_anim.add_subplot(111)
    writer = imageio.get_writer(os.path.join(parentDirectory, "videos/prova_video_circle.mp4"), fps=int(1/dt))
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
#INS ONLY (dead reckoning)
p_INS_ta=np.zeros((steps_INS,3))
th_INS_ta=np.zeros((steps_INS,))
v_INS_ta=np.zeros((steps_INS,3))


### SIMULATION
#########################
step_INS=0
step_UWB=0
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
    INSbool = not (step % int(dt_INS/dt))
    UWBbool = not (step % int(dt_UWB/dt))

    ##########
    # FILTER #
    ##########
    if INSbool:
        am_data,wm_data = uni.navig.generate_INS_measurement(vd_body,w_body)
        uni.navig.INS_predict_xn_nominal_state(dt_INS,am_data,wm_data)
        uni.navig.INS_predict_x_INS(dt_INS,am_data,wm_data)
    if UWBbool:
        UWB_data = uni.navig.generate_UWB_measurement(p)
        uni.navig.UWB_measurement(dt_UWB,UWB_data)
        uni.navig.compute_z()
        #
        uni.navig.update_Q_TMP(dt_UWB)
        #uni.navig.update_Q(dt_UWB)
        uni.navig.update_F(dt_UWB,uni.navig.am_prev.copy(),uni.navig.wm_prev.copy())
        ###uni.navig.predict_dx_error_state()
        uni.navig.predict_P_error_state_covariance(dt_UWB)
        #
        uni.navig.update_epsilon_innovation()
        uni.navig.update_S_theoretical_innovation_covariance()
        uni.navig.update_Sn_estimated_innovation_covariance()
        # uni.navig.update_D_theoretical_zzT_expectation()
        # outlier_detected = uni.navig.check_for_outlier()
        ##while or only an if?
        # while (outlier_detected):
        #     print("DETECTED")
        #     uni.navig.update_epsilon_innovation(hold=True)
        #     uni.navig.update_Sn_estimated_innovation_covariance()
        #     uni.navig.update_D_theoretical_zzT_expectation()
        #     outlier_detected = uni.navig.check_for_outlier()
        # Fuzzy filter
        # uni.navig.apply_fuzzy_filter()
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
    if UWBbool:
        p_UWB_ta[step_UWB,:] = uni.navig.pmUWB
        v_UWB_ta[step_UWB,:] = uni.navig.vmUWB

    # Save frame
    if save_video:
        if not (step % int(dt_video/dt)):
            uni.draw_artists(fig_anim,ax_anim)
            plt.plot(p_des_uni_ta[:(i-1),0],p_des_uni_ta[:(i-1),1],figure=fig_anim, color="xkcd:teal")
            plt.plot(state_uni_ta[:(i-1),0],state_uni_ta[:(i-1),1],figure=fig_anim, color="xkcd:salmon")
            ax_anim.set_aspect('equal')
            ax_anim.set(xlim=(-1, 1), ylim=(-1, 1))
            ax_anim.set_title("Straight trajectory tracking")
            fig_anim.canvas.draw()
            img = np.frombuffer(fig_anim.canvas.tostring_rgb(), dtype='uint8')
            img  = img.reshape(fig_anim.canvas.get_width_height()[::-1]+(3,))
            writer.append_data(img)
            ax_anim.clear()
    
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
### PLOT
#########################
### Unicycle control
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
fig_trj,axs_trj = plt.subplots()
axs_trj.plot(state_uni_ta[:,0],state_uni_ta[:,1],color="xkcd:teal", label="Real")
#axs_trj[0].plot(p_des_uni_ta[:,0],p_des_uni_ta[:,1],color="salmon", label="desired")
axs_trj.plot(p_navig_ta[:,0],p_navig_ta[:,1],color="xkcd:salmon", label="Estimated")
axs_trj.plot(p_INS_ta[:,0],p_INS_ta[:,1],color="xkcd:light salmon", label="INS only (dead reckoning)")
axs_trj.plot(p_UWB_ta[:,0],p_UWB_ta[:,1],color="xkcd:orange", label="UWB", marker="1", linestyle="None")
axs_trj.set_title("Trajectory [x-y]")
axs_trj.set_aspect('equal')
handles_trj, labels_trj = axs_trj.get_legend_handles_labels()
fig_trj.legend(handles_trj, labels_trj, loc='center right')

### Navigation 
UNI_ta = np.linspace(0,t_tot,state_uni_ta.shape[0])
INS_ta = np.linspace(0,t_tot,p_navig_ta.shape[0])
UWB_ta = np.linspace(0,t_tot,p_UWB_ta.shape[0])
fig_nav,axs_nav = plt.subplots(2,3)
axs_nav[0,0].plot(UNI_ta,state_uni_ta[:,0],color="xkcd:teal", label="actual")
axs_nav[0,0].plot(INS_ta,p_navig_ta[:,0],color="xkcd:salmon", label="estimated")
axs_nav[0,0].plot(INS_ta,p_INS_ta[:,0],color="xkcd:light salmon", label="INS only (dead reckoning)")
axs_nav[0,0].plot(UWB_ta,p_UWB_ta[:,0],color="xkcd:orange", label="UWB meas.", marker="1", linestyle="None")
axs_nav[0,0].set_title("X")
axs_nav[0,1].plot(UNI_ta,state_uni_ta[:,1],color="xkcd:teal", label="actual")
axs_nav[0,1].plot(INS_ta,p_navig_ta[:,1],color="xkcd:salmon", label="estimated")
axs_nav[0,1].plot(INS_ta,p_INS_ta[:,1],color="xkcd:light salmon", label="INS only (dead reckoning)")
axs_nav[0,1].plot(UWB_ta,p_UWB_ta[:,1],color="xkcd:orange", label="UWB meas.", marker="1", linestyle="None")
axs_nav[0,1].set_title("Y")
axs_nav[0,2].plot(UNI_ta,state_uni_ta[:,2],color="xkcd:teal", label="actual")
axs_nav[0,2].plot(INS_ta,th_navig_ta,color="xkcd:salmon", label="estimated")
axs_nav[0,2].set_title("Theta")
axs_nav[1,0].plot(UNI_ta,state_d_uni_ta[:,0],color="xkcd:teal", label="actual")
axs_nav[1,0].plot(INS_ta,v_navig_ta[:,0],color="xkcd:salmon", label="estimated")
axs_nav[1,0].plot(INS_ta,v_INS_ta[:,0],color="xkcd:light salmon", label="INS only (dead reckoning)")
axs_nav[1,0].plot(UWB_ta,v_UWB_ta[:,0],color="xkcd:orange", label="UWB meas.", marker="1", linestyle="None")
axs_nav[1,0].set_title("Vx")
axs_nav[1,1].plot(UNI_ta,state_d_uni_ta[:,1],color="xkcd:teal", label="actual")
axs_nav[1,1].plot(INS_ta,v_navig_ta[:,1],color="xkcd:salmon", label="estimated")
axs_nav[1,1].plot(INS_ta,v_INS_ta[:,1],color="xkcd:light salmon", label="INS only (dead reckoning)")
axs_nav[1,1].plot(UWB_ta,v_UWB_ta[:,1],color="xkcd:orange", label="UWB meas.", marker="1", linestyle="None")
axs_nav[1,1].set_title("Vy")
axs_nav[1,2].plot(UNI_ta,state_d_uni_ta[:,2],color="xkcd:teal", label="actual")
axs_nav[1,2].plot(INS_ta,w_navig_ta,color="xkcd:salmon", label="estimated")
axs_nav[1,2].set_title("W (ang.speed)")
handles_nav, labels_nav = axs_nav[0,0].get_legend_handles_labels()
fig_nav.legend(handles_nav, labels_nav, loc='lower center')
#
plt.show()

### END ROUTINE
#########################
if save_video:
    writer.close()

#print(p_navig_ta[1900:2000])