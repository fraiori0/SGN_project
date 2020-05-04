import numpy as np
import scipy as sp
import quaternion as quat
import matplotlib
import matplotlib.pyplot as plt
import robot_navigation as rnav
from math import pi as pi

matplotlib.use('TkAgg')

### SIMULATION PARAMETERS
#########################
dt = 1./200.
dt_INS = 1./100.
dt_UWB = 1./5.
t_tot = 30.
steps = int(t_tot/dt)

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
p_start = np.array((-0.8,-0.8))
p_end = np.array((0.8,0.8))
traj_gen = generate_unicycle_trajectory_straight(t_tot, p_start, p_end)
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
sigma_UWB = 0.1
sigma_INS = np.array([0.06,0.06])
a = 0.98 #sliding window fading coefficient, usually [0.95,0.99]
l = 20 #sliding window length
lamb = 1.3 # >1, parameter for the R innovation contribution weight
b = 0.96 #forgetting factor of the R innovation contribution weight, usually [0.95,0.99]
alpha = 0.01 #secondary regulatory factor for R innovation
zeta = 3. #outliers detection treshold
uni = rnav.Unicycle(sigma_INS,sigma_UWB,uwb_anchors_pos,a,l,lamb,b,alpha,zeta)
uni.set_backstepping_gains(1,30,30,100,200)
uni.set_state((p_des_uni_ta[0]+np.array((0.,0.1))),0.,theta_des_uni_ta[0],0.,0.,0.)

### DATA STORAGE
#########################
error_uni_ta=np.zeros((steps,3))
state_uni_ta=np.zeros((steps,3))

### SIMULATION
#########################
for step in range(steps):
    #
    t = dt*step
    # Unicycle control
    uni.trajectory_tracking_backstepping_step1(dt,p_des_uni_ta[step],v_des_uni_ta[step],theta_des_uni_ta[step],w_des_uni_ta[step])
    tau_v, tau_th = uni.trajectory_tracking_backstepping_step2()
    # Filter
    
    # Save data
    error_uni_ta[step,:] = uni.e 
    state_uni_ta[step,:2]= uni.p
    state_uni_ta[step,2]= uni.theta
    # End step operations
    uni.step_simulation(dt,tau_v,tau_th)
    #uni.navig.iterations += 1

### PLOT
#########################
# Unicycle
fig_uni,axs_uni = plt.subplots(1,2)
axs_uni[0].plot(error_uni_ta[:,0], color="xkcd:light teal", label="x")
axs_uni[0].plot(error_uni_ta[:,1], color="xkcd:dark teal", label="y")
axs_uni[0].plot(error_uni_ta[:,2], color="xkcd:salmon", label="th")
axs_uni[0].set_title("Error")
axs_uni[1].plot(state_uni_ta[:,0], color="xkcd:light teal", label="x")
axs_uni[1].plot(state_uni_ta[:,1], color="xkcd:dark teal", label="y")
axs_uni[1].plot(state_uni_ta[:,2], color="xkcd:salmon", label="th")
axs_uni[1].set_title("State")
handles_uni, labels_uni = axs_uni[0].get_legend_handles_labels()
fig_uni.legend(handles_uni, labels_uni, loc='center right')

plt.show()