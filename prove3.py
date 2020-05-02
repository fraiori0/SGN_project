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
from math import pi as pi

eta = 7
r = 2
dr = 0.2
dt = 1/100
t_array = np.linspace(0.,1.,int(1/dt))
x = lambda t: np.cos(2*pi*t)*(r+dr*np.cos(eta*2*pi*t))
y = lambda t: np.sin(2*pi*t)*(r+dr*np.cos(eta*2*pi*t))
vx = lambda t: (-2*pi*(r*np.sin(2*pi*t)+dr*np.cos(2*eta*pi*t)*np.sin(2*pi*t)+dr*eta*np.cos(2*pi*t)*np.sin(2*eta*pi*t)))
vy = lambda t: (2*pi*(r*np.cos(2*pi*t)+dr*np.cos(2*pi*t)*np.cos(2*eta*pi*t)-dr*eta*np.sin(2*pi*t)*np.sin(2*eta*pi*t)))
v = lambda t: np.linalg.norm([vx(t),vy(t)])
ax = lambda t: -4*pi**2*(np.cos(2*pi*t)*(r+dr*(eta**2+1)*np.cos(2*pi*eta*t))-2*dr*eta*np.sin(2*pi*t)*np.sin(2*pi*eta*t))
ay = lambda t: -4*pi**2*(np.sin(2*pi*t)*(r+dr*(eta**2+1)*np.cos(2*pi*eta*t))+2*dr*eta*np.cos(2*pi*t)*np.sin(2*pi*eta*t))
theta = lambda t: np.arctan2(vx(t),vy(t))
w = lambda t,dt: (theta(t)-theta(t-dt))/dt
x_array = x(t_array)
y_array = y(t_array)
vx_array = vx(t_array)
vy_array = vy(t_array)
ax_array = ax(t_array)
ay_array = ay(t_array)
theta_array = theta(t_array)
w_array = w(t_array,dt)

print(theta_array)

fig,ax = plt.subplots(2,2)
ax[0,0].plot(t_array,x_array, color="xkcd:dark salmon")
ax[0,0].plot(t_array,vx_array, color="xkcd:salmon")
#ax[0,0].plot(t_array,ax_array, color="xkcd:light salmon")
#ax[0,0].set_aspect('equal')
ax[1,0].plot(t_array,theta_array, color="xkcd:dark salmon")
ax[1,0].plot(t_array,w_array, color="xkcd:salmon")
plt.show()