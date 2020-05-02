import numpy as np
import scipy as sp
import quaternion as quat
import matplotlib
import matplotlib.pyplot as plt
import robot_navigation as rnav
import matplotlib.animation as animation
import imageio
import imageio.plugins.pillow
from math import pi as pi
matplotlib.use('TkAgg')

def generate_unicycle_trajectory_straight(t_tot, p_start, p_end):
    #p_start/end as numpy arrays
    v = np.linalg.norm(p_start - p_end)/t_tot
    theta = np.arctan2(*list(p_end - p_start))
    w = 0.
    def trajectory_generator(t):
        p = p_start + v * np.array(np.cos(theta),np.sin(theta)) * t
        return p,v,theta,w
    return trajectory_generator

# def generate_unicycle_trajectory_polygon(t_tot, r, dr, eta):
#     #p_start/end as numpy arrays
#     x = lambda t: np.cos(2*pi*t)*(r+dr*np.cos(eta*2*pi*t))
#     y = lambda t: np.sin(2*pi*t)*(r+dr*np.cos(eta*2*pi*t))
#     vx = lambda t: (-2*pi*(r*np.sin(2*pi*t)+dr*np.cos(2*eta*pi*t)*np.sin(2*pi*t)+dr*eta*np.cos(2*pi*t)*np.sin(2*eta*pi*t)))
#     vy = lambda t: (2*pi*(r*np.cos(2*pi*t)+dr*np.cos(2*pi*t)*np.cos(2*eta*pi*t)-dr*eta*np.sin(2*pi*t)*np.sin(2*eta*pi*t)))
#     v = lambda t: np.linalg.norm([vx(t),vy(t)])
#     ax = lambda t: -4*pi**2*(np.cos(2*pi*t)*(r+dr*(eta**2+1)*np.cos(2*pi*eta*t))-2*dr*eta*np.sin(2*pi*t)*np.sin(2*pi*eta*t))
#     ay = lambda t: -4*pi**2*(np.sin(2*pi*t)*(r+dr*(eta**2+1)*np.cos(2*pi*eta*t))+2*dr*eta*np.cos(2*pi*t)*np.sin(2*pi*eta*t))
#     theta = lambda t: np.arctan2(vx(t),vy(t))
#     def trajectory_generator(t,dt):
#         p_val = np.array((x(t/t_tot),y(t/t_tot)))
#         v_val = v(t/t_tot)
#         theta_val = theta(t/t_tot)
#         w_val = (theta(t/t_tot)-theta(t/t_tot-dt))/dt
#         return p_val,v_val,theta_val,w_val
#     return trajectory_generator

# def generate_unicycle_trajectory_polygon(t_tot, r, dr, eta):
#     #p_start/end as numpy arrays
#     x = lambda t: np.cos(2*pi*t)*(r+dr*np.cos(eta*2*pi*t))
#     y = lambda t: np.sin(2*pi*t)*(r+dr*np.cos(eta*2*pi*t))
#     vx = lambda t: (-2*pi*(r*np.sin(2*pi*t)+dr*np.cos(2*eta*pi*t)*np.sin(2*pi*t)+dr*eta*np.cos(2*pi*t)*np.sin(2*eta*pi*t)))
#     vy = lambda t: (2*pi*(r*np.cos(2*pi*t)+dr*np.cos(2*pi*t)*np.cos(2*eta*pi*t)-dr*eta*np.sin(2*pi*t)*np.sin(2*eta*pi*t)))
#     v = lambda t: np.linalg.norm([vx(t),vy(t)])
#     ax = lambda t: -4*pi**2*(np.cos(2*pi*t)*(r+dr*(eta**2+1)*np.cos(2*pi*eta*t))-2*dr*eta*np.sin(2*pi*t)*np.sin(2*pi*eta*t))
#     ay = lambda t: -4*pi**2*(np.sin(2*pi*t)*(r+dr*(eta**2+1)*np.cos(2*pi*eta*t))+2*dr*eta*np.cos(2*pi*t)*np.sin(2*pi*eta*t))
#     theta = lambda t: np.arctan2(vx(t),vy(t))
#     def trajectory_generator(t,dt):
#         p_val = np.array((x(t/t_tot),y(t/t_tot)))
#         v_val = v(t/t_tot)
#         theta_val = theta(t/t_tot)
#         w_val = (theta(t/t_tot)-theta(t/t_tot-dt))/dt
#         return p_val,v_val,theta_val,w_val
#     return trajectory_generator


dt = 1/200
t_tot = 20.
p_start = np.array((-0.8,-0.8))
p_end = np.array((0.8,0.8))
steps = int(t_tot/dt)
###
traj_gen = generate_unicycle_trajectory_straight(t_tot, p_start, p_end)
p_des_time_array = np.zeros((steps,2))
v_des_time_array = np.zeros(steps)
theta_des_time_array = np.zeros(steps)
w_des_time_array = np.zeros(steps)
for i in range(steps):
    p_des_time_array[i] = traj_gen(dt*i)[0]
    v_des_time_array[i] = traj_gen(dt*i)[1]
    theta_des_time_array[i] = traj_gen(dt*i)[2]
    w_des_time_array[i] = traj_gen(dt*i)[3]
###
fig_anim = plt.figure()
ax_anim = fig_anim.add_subplot(111)
writer = imageio.get_writer("./videos/Straigth trajectory.mp4", fps=int(1/dt))
###
uwb_anchors_pos = np.array([
    [1.,2.,3.],
    [1.,1.,1.],
    [0.,-4.,3.],
    [0.5,2.,1.],
    [3.,2.,0.2],
    [2.,0.,2.]
])

unicy = rnav.Unicycle([0.4,0.2,0.15,0.01],uwb_anchors_pos,0.96,10,0.5,0.96,1,2.)
unicy.set_backstepping_gains(1,20,20,100,200)
unicy.set_state(p_des_time_array[0],0.,theta_des_time_array[0],0.,0.,0.)

error_time_array=np.zeros((steps,3))
state_time_array=np.zeros((steps,3))

for step in range(int(t_tot/dt)):
    t = dt*step
    unicy.trajectory_tracking_backstepping_step1(dt,p_des_time_array[step],v_des_time_array[step],theta_des_time_array[step],w_des_time_array[step])
    tau_v, tau_th = unicy.trajectory_tracking_backstepping_step2()
    error_time_array[step,:] = unicy.e 
    state_time_array[step,:2]= unicy.p
    state_time_array[step,2]= unicy.theta
    #unicy.step_simulation(dt,0.1,0.001)
    unicy.step_simulation(dt,tau_v,tau_th)
    ###
    # unicy.draw_artists(fig_anim,ax_anim)
    # plt.plot(p_des_time_array[:,0],p_des_time_array[:,1],figure=fig_anim, color="xkcd:salmon")
    # ax_anim.set_aspect('equal')
    # ax_anim.set(xlim=(-3, 3), ylim=(-3, 3))
    # ax_anim.set_title("Straight trajectory tracking")
    # fig_anim.canvas.draw()
    # img = np.frombuffer(fig_anim.canvas.tostring_rgb(), dtype='uint8')
    # img  = img.reshape(fig_anim.canvas.get_width_height()[::-1]+(3,))
    # writer.append_data(img)
    # ax_anim.clear()
    ###
    print(step*100/steps,"%")
###
writer.close()

print("des: ",np.round(p_des_time_array[0,:],3))
print("state: ",np.round(state_time_array[:3,:],3))
###
fig_st,axs_st = plt.subplots(1,2)
axs_st[0].plot(error_time_array[:,0], color="xkcd:light teal", label="x")
axs_st[0].plot(error_time_array[:,1], color="xkcd:dark teal", label="y")
axs_st[0].plot(error_time_array[:,2], color="xkcd:salmon", label="th")
axs_st[0].set_title("Error")
axs_st[1].plot(state_time_array[:,0], color="xkcd:light teal", label="x")
axs_st[1].plot(state_time_array[:,1], color="xkcd:dark teal", label="y")
axs_st[1].plot(state_time_array[:,2], color="xkcd:salmon", label="th")
axs_st[1].set_title("State")
handles_st, labels_st = axs_st[0].get_legend_handles_labels()
fig_st.legend(handles_st, labels_st, loc='center right')
###
fig_trj,axs_trj = plt.subplots(1,2)
axs_trj[0].plot(state_time_array[:,0],state_time_array[:,1],color="xkcd:teal", label="actual")
axs_trj[0].plot(p_des_time_array[:,0],p_des_time_array[:,1],color="salmon", label="desired")
axs_trj[0].set_title("Trajectory [x-y]")
axs_trj[0].plot(state_time_array[:,0],state_time_array[:,1],color="xkcd:teal", label="actual")
axs_trj[0].plot(p_des_time_array[:,0],p_des_time_array[:,1],color="salmon", label="desired")
axs_trj[0].set_title("Trajectory vibe check")
###
plt.show()
###



