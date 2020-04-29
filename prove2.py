import numpy as np
import scipy as sp
import quaternion as quat
import matplotlib
import matplotlib.pyplot as plt
import robot_navigation as rnav
import matplotlib.animation as animation
import imageio
import imageio.plugins.pillow
matplotlib.use('TkAgg')

def generate_unicycle_trajectory(t_tot, p_start, p_end):
    #p_start/end as numpy arrays
    v = np.linalg.norm(p_start - p_end)/t_tot
    theta = np.arctan2(*list(p_end - p_start))
    w = 0.
    def trajectory_generator(t):
        p = p_start + v * np.array(np.cos(theta),np.sin(theta)) * t
        return p,v,theta,w
    return trajectory_generator


dt = 1/400
t_tot = 4.
p_start = np.array((-0.8,-0.8))
p_end = np.array((0.8,0.8))
steps = int(t_tot/dt)

traj_gen = generate_unicycle_trajectory(t_tot, p_start, p_end)
p_time_array = np.zeros((steps,2))
v_time_array = np.zeros(steps)
theta_time_array = np.zeros(steps)
w_time_array = np.zeros(steps)
for i in range(steps):
    p_time_array[i] = traj_gen(dt*i)[0]
    v_time_array[i] = traj_gen(dt*i)[1]
    theta_time_array[i] = traj_gen(dt*i)[2]
    w_time_array[i] = traj_gen(dt*i)[3]

fig = plt.figure()
ax = fig.add_subplot(111)
writer = imageio.get_writer("./woh.mp4", fps=int(1/dt))

uwb_anchors_pos = np.array([
    [1.,2.,3.],
    [1.,1.,1.],
    [0.,-4.,3.],
    [0.5,2.,1.],
    [3.,2.,0.2],
    [2.,0.,2.]
])

unicy = rnav.Unicycle([0.4,0.2,0.15,0.01],uwb_anchors_pos,0.96,10,0.5,0.96,1,2.)
unicy.set_backstepping_gains(2,1,1,100,1000)
p0 = p_start
unicy.set_state(p0,0.,3/4,0.,0.,0.)


for step in range(int(t_tot/dt)):
    t = dt*step
    traj_gen(t)
    unicy.trajectory_tracking_backstepping_step1(dt,p_time_array[i],v_time_array[i],theta_time_array[i],w_time_array[i])
    tau_v, tau_th = unicy.trajectory_tracking_backstepping_step2()
    #unicy.step_simulation(dt,0.1,0.001)
    unicy.step_simulation(dt,tau_v,tau_th)
    #
    unicy.draw_artists(fig,ax)
    #
    plt.plot(p_time_array[:,0],p_time_array[:,0],figure=fig, color="xkcd:salmon")
    #
    ax.set_aspect('equal')
    ax.set(xlim=(-1, 1), ylim=(-1, 1))
    ax.set_title("woh")
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    img  = img.reshape(fig.canvas.get_width_height()[::-1]+(3,))
    writer.append_data(img)
    ax.clear()

    print(step*100/steps,"%")


writer.close()
