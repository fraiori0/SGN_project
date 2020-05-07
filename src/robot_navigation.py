import numpy as np
import scipy as sp
from scipy.spatial.transform import Rotation as Rot
import quaternion as quat
import matplotlib
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import matplotlib.animation as animation
from math import pi as pi


def rotate_vector(q,v):
    """Rotate vector v with the quaternion q

    Parameters
    ----------
    q : np.quaternion\n
    v : np.array(3,float)
    Returns
    -------
    np.array(3,float)
       rotated vector
    """
    # v' = q ⊗ v ⊗ q*
    qunitary = q.normalized()
    s = qunitary.w
    r = qunitary.vec
    return v + 2*np.cross(r,(s*v + np.cross(r,v)))

def vec_to_0quat(v):
    #v is supposed to be a np.array with shape (3,)
    return quat.as_quat_array(np.insert(v,0,0.))

def skew(v):
    """skew matrix from 3d-vector
    """
    return np.array([[0.,-v[2], v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])

class Navigator_State:
    def __init__(self):
        self.p = np.array((0.,0.,0.))
        self.v = np.array((0.,0.,0.))
        self.q = np.quaternion(1, 0, 0 ,0)
        self.ab = np.array((0.,0.,0.))
        self.wb = np.array((0.,0.,0.))
        self.pd = self.v
        self.vd = np.array((0.,0.,0.))
        self.qd = np.quaternion(0, 0, 0 ,0)
        #self.abd = np.array((0.,0.,0.))
        #self.wbd = np.array((0.,0.,0.))
    def set_state(self,p0,v0,q0,ab0,w0):
        #TODO add check on numpy type and dimensions
        self.p = p0
        self.v = v0
        self.q = q0
        self.ab = ab0
        self.wb = w0
    def backward_euler(self,dt,vd,qd,abd,wbd):
        self.p += dt*self.v
        self.v += dt*vd
        self.q = (self.q + (qd * dt)).normalized()
        self.ab += dt*abd
        self.wb += dt*wbd
    def as_array_15(self):
        #return as an array, including only the vector component of the quaternion
        return np.array((*self.p, *self.v, *(self.q.vec), *self.ab, * self.wb))
        

class Navigator:
    def __init__(self,sigma_INS,sigma_UWB,UWB_anchors_pos, a, l, lamb, b, alpha, zeta, MNCsigma=0.0136):
        """Init class object

        Parameters
        ----------
        sigma_INS : [list(4,float)]
            std_dev of INS measurement (an,wn)\n
        sigma_UWB : [float]
            std_dev of UWB noise\n
        UWB_anchors_pos : [np.array((#anchors,3),float)]
            array containing as rows the positions in the global 
            reference frame of the UWB anchors\n
        a : [float]
            sliding window fading coefficient, usually [0.95,0.99],
            see eqn(36)\n
        l : [int]
            sliding window length, see eqn(36)\n
        lamb : [float]
            parameter for the R innovation contribution weight, 
            see eqn (27)\n
        b : [float]
            forgetting factor of the R innovation contribution weight, 
            usually [0.95,0.99], see eqn (27)\n
        alpha : [float]
            secondary regulatory factor for R innovation, see eqn (27) and Remark(2) of the paper. 
            larger values approach a true R with fewer iterations but may lead to unstable estimates
            and viceversa\n
        zeta : [float]
            outliers detection treshold\n
        MNCsigma : [float]
            std_dev for initializing Rn
        """
        self.iterations = 0
        self.g = np.array([0.,0.,-9.81])
        self.xn = Navigator_State() #nominal state
        self.xn_prev = Navigator_State() #previous instance of the nominal state (used for F?)
        self.x_INS = Navigator_State() # state updated only by the INS measurements, no correction of any type
        self.dx = Navigator_State() #error state
        self.dxapp = np.zeros((15,)) #error state with approximate rotation (\delta x in the paper) (p,v,q,ab,wb)
        self.am_prev = np.array((0.,0.,0.))
        self.wm_prev = np.array((0.,0.,0.))
        self.am = np.array((0.,0.,0.))
        self.wm = np.array((0.,0.,0.)) # previous INS measurements are stored since are needed for F computation?
        self.P = np.zeros(((15,15)))
        self.sigma_INS = np.array([sigma_INS[0],sigma_INS[1]])
        self.sigma_INS_RW = np.array([sigma_INS[0],sigma_INS[1]])#INS integration random walk variance, updated each step
        self.sigma_UWB = sigma_UWB
        self.update_Q(0.0)
        self.Gamma_n = np.block(  [ [np.zeros((3,3)),  np.zeros((3,3)),    np.zeros((3,3)),    np.zeros((3,3))],
                                    [np.eye(12)]    ])
        self.F = np.zeros((15,15))
        self.H = np.block([ [np.eye(3),np.zeros((3,3)),np.zeros((3,3)),np.zeros((3,3)),np.zeros((3,3))],
                            [np.zeros((3,3)),np.eye(3),np.zeros((3,3)),np.zeros((3,3)),np.zeros((3,3))]
                            ])
        self.UWB_anchors_pos = UWB_anchors_pos
        self.set_G()
        self.set_b_precomp()
        self.pmUWB_prev = np.array((0.,0.,0.))
        self.pmUWB = np.array((0.,0.,0.))
        self.vmUWB = np.array((0.,0.,0.))
        self.z = np.array((0.,0.,0.,0.,0.,0.))
        self.set_sliding_window_properties(a,l) # parameters for computing omega in eqn(36)
        self.epsilon = np.zeros((self.l_par,6)) # most recent as last element
        self.Sn = np.eye(6) # estimated epsilon covariance (eqn(36))
        self.S = np.eye(6) # theorethical epsilon covariance (eqn(23))
        self.Rn = MNCsigma*np.eye(6) # theoretical MNC estimate
        self.R = MNCsigma*np.eye(6) # estimated MNC
        self.set_R_innovation_weight_par(lamb,b, alpha) #parameters of R innovation contribution weight, eqn (27)
        self.fuzzy_setup()
        self.rfz = 0.
        self.sfz = 1.
        self.zeta = zeta
        self.D = np.eye(6) #just for initialization purpose

    def compute_xnd(self,am,wm):
        """derivative of the nominal state, eq.(4) of the paper

        Parameters
        ----------
        am : np.array(3,float)
            measured acceleration in body-frame (output of INS accelerometer)\n
        wm : np.array(3,float)
            measured angular speed in body-frame (output of INS gyroscope)

        Returns
        -------
        (vnd,qnd,anbd,wnbd)
            derivatives, to be passed when calling self.xn.backward_euler()
        """
        vnd = rotate_vector(self.xn.q, (am-self.xn.ab))# + self.g 
        qnd = (self.xn.q * vec_to_0quat(wm-self.xn.wb))/2.
        anbd = 0.
        wnbd = 0.
        return (vnd,qnd,anbd,wnbd)

    def set_G(self):
        """Set self.G based on the given absolute position of the UWB anchors (eqn(11) of the paper)

        Parameters
        ----------
        uwb_anchors_pos : np.array((#anchors,3),float)
            each row should contain the (x,y,z) position of the i-th tag

        Returns
        -------
        np.array((),float)
            G
        """
        self.G = self.UWB_anchors_pos[0,:] - self.UWB_anchors_pos[1:,:]
        return self.G

    def set_b_precomp(self):
        # precompute part of eqn (12) so it's faster to calculate the estimated position from UWB data
        self.b_precomp =  np.sum(((self.UWB_anchors_pos[0,:])**2-(self.UWB_anchors_pos[1:,:])**2), axis=1)

    def UWB_measurement(self,dt,UWB_data):
        """Compute the estimated position and velocity based on UWB measurement (eqns 13 & 14)

        Parameters
        ----------
        dt : [float]
            time from the previous measurement\n
        UWB_data : [np.array((#anchors,),float)]
            array containing the measured distance from the UWB anchors

        Returns
        -------
        [tuple(pm,vm)]
            pm and vm are the estimated position and velocity (each as a np.array((3,),float))
        """
        # eqns (13 & 14) of the paper
        self.pmUWB_prev=self.pmUWB
        b_add = (UWB_data[1:]**2)-(UWB_data[0]**2)
        self.pmUWB = (sp.linalg.pinv(self.G).dot(b_add + self.b_precomp))/2.
        self.vmUWB = (self.pmUWB-self.pmUWB_prev)/dt 
        return self.pmUWB.copy(), self.vmUWB.copy()
    
    def compute_z(self):
        # self.z = np.array((*(self.pmUWB-self.xn.p), *(self.vmUWB-self.xn.v)))
        self.z = np.array((*(self.pmUWB-self.x_INS.p), *(self.vmUWB-self.x_INS.v)))
        return self.z.copy()

    def INS_predict_xn_nominal_state(self,dt,am,wm):
        """Predict the nominal state with INS measurement

        Parameters
        ----------
        dt : [float]
            time from the last prediction with INS\n
        am : [np.array((3,),float)]
            acceleration INS measurement\n
        wm : [np.array((3,),float)]
            angular speed INS measurement

        Returns
        -------
        [Navigator_State]
            nominal state of the robot
        """
        # store previous state
        self.xn_prev.p=self.xn.p.copy()
        self.xn_prev.v=self.xn.v.copy()
        self.xn_prev.q=self.xn.q.copy()
        self.xn_prev.ab=self.xn.ab.copy()
        self.xn_prev.wb=self.xn.wb.copy()
        # predict
        derivatives = self.compute_xnd(am,wm)
        self.xn.backward_euler(dt,*derivatives)

    def INS_predict_x_INS(self,dt,am,wm):
        """Predict the state only with INS measurement

        Parameters
        ----------
        dt : [float]
            time from the last prediction with INS\n
        am : [np.array((3,),float)]
            acceleration INS measurement\n
        wm : [np.array((3,),float)]
            angular speed INS measurement

        Returns
        -------
        [Navigator_State]
            nominal state of the robot
        """
        # predict
        derivatives = self.compute_xnd(am,wm)
        self.x_INS.backward_euler(dt,*derivatives)

    def update_Q(self,dt):
        self.Q = sp.linalg.block_diag(np.eye(3)*((self.sigma_INS[0]*dt)**2),   np.eye(3)*((self.sigma_INS[1]*dt)**2),
            np.eye(3)*((self.sigma_INS_RW[0]*dt*self.iterations)**2),  np.eye(3)*((self.sigma_INS_RW[1]*dt*self.iterations)**2))
        return self.Q
    def update_Q_TMP(self,dt):
        self.Q = sp.linalg.block_diag(np.eye(3)*((self.sigma_INS[0]*dt)**2),   np.eye(3)*((self.sigma_INS[1]*dt)**2),
            np.eye(3)*((self.sigma_INS_RW[0]*dt)**2),  np.eye(3)*((self.sigma_INS_RW[1]*dt)**2))
        return self.Q
    
    def update_F(self,dt,am_prev,wm_prev):
        # eqn (19)
        # not explained in the paper, for Rtmp approximate omega*dt as the axis-angle rotation vector,
        # then compute a rotation matrix
        # https://www.researchgate.net/post/Validity_of_Rotation_matrix_calculations_from_angular_velocity_reading
        Rtmp = (Rot.from_rotvec((wm_prev-self.xn_prev.wb)*dt)).as_matrix()
        C_prev = quat.as_rotation_matrix(self.xn_prev.q)
        self.F = np.block([[np.eye(3),     np.eye(3)*dt,   np.zeros((3,3)),  np.zeros((3,3)),    np.zeros((3,3))],
                    [np.zeros((3,3)),  np.eye(3),  -C_prev.dot(skew(am_prev-self.xn_prev.ab))*dt,
                        -C_prev*dt,     np.zeros((3,3))],
                    [np.zeros((3,3)),   np.zeros((3,3)),    Rtmp,               np.zeros((3,3)),    -np.eye(3)*dt],
                    [np.zeros((3,3)),   np.zeros((3,3)),    np.zeros((3,3)),    np.eye(3),          np.zeros((3,3))],
                    [np.zeros((3,3)),   np.zeros((3,3)),    np.zeros((3,3)),    np.zeros((3,3)),    np.eye(3)]])
        return self.F

    def predict_dx_error_state(self):
        """[DEPRECATED] shouldn't be used, the predication on dx is always zero (null mean)

        Returns
        -------
        [type]
            [description]
        """
        # eqn (17)
        # NOTE: since the mean of the error state is initialized to 0 this return always zero
        # is useful only when dx is initialized with a different mean
        self.dxapp = self.F.dot(self.dxapp)
        # update also dx to keep a match between the two
        self.dx.p = self.dxapp[0:3]
        self.dx.v = self.dxapp[3:6]
        self.dx.q = quat.from_rotation_vector(self.dxapp[6:9])
        self.dx.ab = self.dxapp[9:12]
        self.dx.wb = self.dxapp[12:15]
        
    def predict_P_error_state_covariance(self,dt):
        """Update the prediction of the error state (dx) covariance matrix, P

        Parameters
        ----------
        dt : [float]
            time from the last prediction\n
        am : [np.array((3,),float)]
            acceleration INS measurement\n
        wm : [np.array((3,),float)]
            angular speed INS measurement

        Returns
        -------
        np.array((15,15),float)
            covariance matrix P

        Warnings
        -------
        dt must match the one used to update F
        """
        self.P = self.F.dot(self.P.dot(self.F.T)) + self.Gamma_n.dot((self.Q).dot(self.Gamma_n.T))
        return self.P.copy()
    
    def update_epsilon_innovation(self, hold=False):
        """calculate the innovation, eqn(22)\n
        Remember to call UWB_measurement and predict_dx_error_state before
        
        Parameters
        ----------
        hold : [bool]
            whether to change the last value in place, set to True when calling outlier correction, by default False
        """
        epsilon_new = np.reshape((self.z - self.H.dot(self.dxapp)),(1,6))
        self.epsilon = np.append(self.epsilon, epsilon_new, axis=0) # append new value as last
        if not hold:
            self.epsilon = np.delete(self.epsilon, 0, 0) # remove first value
        else:
            self.epsilon = np.delete(self.epsilon, -2, 0) # remove outlier value
        return self.epsilon.copy()
    
    def set_sliding_window_properties(self,a,l):
        # parameters for computing sigma in eqn (36)
        self.a_par = a
        self.l_par = l
        self.sigma = np.zeros((self.l_par,))
        for i in range(self.l_par):
            self.sigma[i] = self.a_par**((self.l_par-1)-i) * (1-self.a_par)/(1-self.a_par**self.l_par)

    def update_S_theoretical_innovation_covariance(self):
        # eqn (23)
        self.S = self.H.dot(self.P.dot(self.H.T)) + self.R
        return self.Sn.copy()

    def update_Sn_estimated_innovation_covariance(self):
        # eqn (36)
        for i in range(self.l_par):
            self.Sn = self.sigma[i]*((np.reshape(self.epsilon[i,:],(1,6))).T).dot(np.reshape(self.epsilon[i,:],(1,6)))
        return self.Sn.copy()

    def update_Rn_theoretical_estimated_MNC(self):
        # eqn (25)
        epsilon_k = np.reshape(self.epsilon[-1,:],(1,6))
        self.Rn = (
            self.Sn - self.H.dot(self.P.dot(self.H.T))
        )
        # self.Rn = (
        #     (epsilon_k.T).dot(epsilon_k) - self.H.dot(self.P.dot(self.H.T))
        # )
        return self.Rn.copy()
    
    def fuzzy_setup(self):
        self.rfuzzy = ctrl.Antecedent(np.arange(0,0.8,0.1),'rfuzzy')
        self.sfuzzy = ctrl.Consequent(np.arange(0.8,2.0,0.2),'sfuzzy')
        self.rfuzzy['less'] = fuzz.trimf(self.rfuzzy.universe, [0,0,0.3])
        self.rfuzzy['equal'] = fuzz.trimf(self.rfuzzy.universe, [0.1,0.4,0.7])
        self.rfuzzy['more'] = fuzz.trimf(self.rfuzzy.universe, [0.5,0.8,0.8])
        self.sfuzzy['less'] = fuzz.trimf(self.sfuzzy.universe, [0.8,0.8,1.2])
        self.sfuzzy['equal'] = fuzz.trimf(self.sfuzzy.universe, [1.,1.4,1.8])
        self.sfuzzy['more'] = fuzz.trimf(self.sfuzzy.universe, [1.6,2.0,2.0])
        fuzzy_rule1 = ctrl.Rule(self.rfuzzy['less'],self.sfuzzy['less'])
        fuzzy_rule2 = ctrl.Rule(self.rfuzzy['equal'],self.sfuzzy['equal'])
        fuzzy_rule3 = ctrl.Rule(self.rfuzzy['more'],self.sfuzzy['more'])
        self.fuzzy_filter_system = ctrl.ControlSystem(rules=[fuzzy_rule1, fuzzy_rule2, fuzzy_rule3])
        self.fuzzy_filter = ctrl.ControlSystemSimulation(self.fuzzy_filter_system)

    def apply_fuzzy_filter(self):
        self.rfz = abs(np.trace(self.Sn)/np.trace(self.S) - 1)
        if self.iterations < 20:
            self.sfz = 1.
            return self.sfz
        self.fuzzy_filter.input['rfuzzy'] = self.rfz
        self.fuzzy_filter.compute()
        self.sfz = self.fuzzy_filter.output['sfuzzy']
        return self.sfz

    def set_R_innovation_weight_par(self, lamb, b, alpha):
        self.lamb_par = lamb
        self.b_par = b
        self.alpha_par = alpha
    
    def update_R_estimated_MNC(self):
        # eqns(26 & 27)
        # update Rn before this and remeber to update iterations at the end of each cycle
        self.dk = (self.lamb_par-self.b_par)/(self.lamb_par - self.b_par**(self.iterations+1))
        a = (self.sfz **self.alpha_par) * self.dk
        self.R = (1-a)*self.R + a*self.Rn
        return self.R.copy()

    def update_D_theoretical_zzT_expectation(self):
        # eqns (33 and 32)
        dxapp_column = np.reshape(self.dxapp, (self.dxapp.shape[0],1))
        self.D = (self.H.dot(self.P.dot(self.H.T)) + self.R
            + self.H.dot(dxapp_column.dot(dxapp_column.T.dot(self.H.T)))
        )
        self.E_zkzk = self.D + self.Sn
        return self.D.copy()
    
    def check_for_outlier(self):
        #eqns (34 & 35)
        outlier = False
        Mk = np.abs(np.diagonal(self.E_zkzk)/np.diagonal(self.D))
        Mk_sqrt = np.sqrt(Mk)
        Fk = np.eye(Mk.shape[0])
        for i in range(Fk.shape[0]):
            if Mk[i] > self.zeta:
                outlier = True
                Fk[i,i] = 1/Mk_sqrt[i]
        self.z = Fk.dot(self.z)
        return outlier

    def update_Kalman_gain(self):
        self.K = self.P.dot( self.H.T.dot( sp.linalg.pinv(self.H.dot(self.P.dot(self.H.T)) + self.R) ))
        return self.K
    
    def update_dx_and_P(self):
        # this update consider only position and velocity of the robot
        self.dxapp = self.K.dot(self.epsilon[-1])
        # update also dx to keep a match between the two
        self.dx.p = self.dxapp[0:3]
        self.dx.v = self.dxapp[3:6]
        self.dx.q = quat.from_rotation_vector(self.dxapp[6:9])
        self.dx.ab = self.dxapp[9:12]
        self.dx.wb = self.dxapp[12:15]
        #
        self.P = (np.eye(15) - self.K.dot(self.H)).dot(self.P)

    def update_xn(self):
        self.xn.p += self.dx.p
        self.xn.v += self.dx.v
        self.xn.q = (self.xn.q * self.dx.q).normalized()
        self.xn.ab += self.dx.ab
        self.xn.wb += self.dx.wb
    
    def reset_dx_error_state(self):
        self.dxapp = np.zeros((15,))
        # update also dx to keep a match between the two
        self.dx.p = self.dxapp[0:3]
        self.dx.v = self.dxapp[3:6]
        self.dx.q = quat.from_rotation_vector(self.dxapp[6:9])
        self.dx.ab = self.dxapp[9:12]
        self.dx.wb = self.dxapp[12:15]
    
    def generate_INS_measurement(self,a_real,w_real):
        #add real noise to the real state, that should be passed as an input
        self.am_prev = self.am.copy()
        self.wm_prev = self.wm.copy()
        self.am = a_real + np.random.normal(loc=0.0,scale=self.sigma_INS[0],size=(3,))
        self.wm = w_real + np.random.normal(loc=0.0,scale=self.sigma_INS[1],size=(3,))
        return self.am.copy(),self.wm.copy()
    
    def generate_UWB_measurement(self,pos,outlier=False):
        # pos = real position, passed as a numpy array
        pos = np.reshape(pos, (3,))
        exact_mes = np.linalg.norm((self.UWB_anchors_pos - pos),axis=1)
        noisy_mes = exact_mes + np.random.normal(loc=0.0,scale=self.sigma_UWB,size=(self.UWB_anchors_pos.shape[0],))
        if outlier:
            # outliers random distributed between [-0.2,0.2], as in the paper
            # here they are added to the measurement from every tag, is it correct?
            out_mes = -0.2 + 0.4*np.random.random((self.UWB_anchors_pos.shape[0],))
            noisy_mes = noisy_mes + out_mes
        return noisy_mes

        
    # def compute_dxd(self,dtheta,am,an,aw,wm,wn,ww):
    #     """derivative of the error state, eq.(5) of the paper

    #     Parameters
    #     ----------
    #     dtheta : np.array(3,float)
    #         rotation error of the quaternion\n
    #     am : np.array(3,float)
    #         measured acceleration in body-frame (output of INS accelerometer)\n
    #     an : np.array(3,float)
    #         noise on acceleration measurement\n
    #     aw : np.array(3,float)
    #         random walk noise of bias of acceleration measurement\n
    #     wm : np.array(3,float)
    #         measured angular speed (output of INS gyroscope)\n
    #     wn : np.array(3,float)
    #         noise on angular speed measurement\n
    #     ww : np.array(3,float)
    #         random walk noise of bias of angular speed measurement

    #     Returns
    #     -------
    #     (dvd,dTHETAD,dabd,dwbd)
    #         WARNING, dthetad is not dqd?
    #         derivatives, to be passed when calling self.dx.backward_euler()
    #     """
    #     dvd = -rotate_vector(self.x.q, ((np.cross((am-self.xn.ab),dtheta)) + self.dx.ab + an)) 
    #     dthetad = - (np.cross((wm-self.xn.wb),dtheta) + ww + wn)
    #     dabd = aw
    #     dwbd = ww
    #     return (dvd,dthetad,dabd,dwbd)
#############################################################################


class Unicycle_State:
    def __init__(self):
        # state as p,v,q,w
        # position, velocity, orientation quaternion, angular speed
        # all in global reference frame
        self.p = np.array((0.,0.))
        self.v = 0.
        self.theta = 0.
        self.w = 0.
        self.vd = 0.
        self.wd = 0.
    def set_state(self,p,v,theta,w,vd,wd):
        self.p = (np.array(p)).copy()
        self.v = v
        self.theta = theta
        self.w = w
        self.vd = vd
        self.wd = wd
    def update_state_Euler(self,dt,vd,wd):
        self.p += self.v * dt * np.array((np.cos(self.theta), np.sin(self.theta)))
        self.v += self.vd * dt
        self.theta = (self.theta + (self.w * dt))%(2*pi)
        self.w += wd * dt
        self.vd = vd
        self.wd = wd
    def update_state_Euler_KIN(self,dt,v,w):
        self.vd = (v-self.v)/dt
        self.wd = (w-self.w)/dt
        self.v = v
        self.w = w
        self.p += self.v * dt * np.array((np.cos(self.theta), np.sin(self.theta)))
        self.theta = (self.theta + (self.w * dt))%(2*pi)
    def return_as_3D_with_quat(self):
        p = np.append(self.p, 0.) # append z position
        v = self.v * np.array((np.cos(self.theta), np.sin(self.theta), 0.))
        q = quat.from_rotation_vector(self.theta * np.array((0,0,1))) # rotation is only along the z-axis (2D model)
        w = np.array((0.,0.,self.w))
        vd = self.vd * np.array((np.cos(self.theta), np.sin(self.theta), 0.)) + skew(w).dot(v) #Poisson relation
        wd = np.array((0.,0.,self.wd))
        return p,v,q,w,vd,wd


        
class Unicycle(Unicycle_State):
    # based on the equations reported in the notes by Prof. A.Bicchi, Nonlinear Systems, pag.86
    def __init__(self,sigma_INS, sigma_UWB,UWB_anchors_pos, a, l, lamb, b, alpha, zeta):
        super().__init__()
        self.navig = Navigator(sigma_INS, sigma_UWB, UWB_anchors_pos, a, l, lamb, b, alpha, zeta)
        self.m = 2.
        self.Iz = 0.005 #like a 0.1m radius disk
        self.tau_v = 0.
        self.tau_th = 0.
        self.set_backstepping_gains(2.,1.,1.,10.,10.)
        self.back1 = np.array((0.,0.)) #vc and wc, since the derivatives are needed
        self.back1d = np.array((0.,0.)) #derivative of self.back1
        self.back2 = np.array((0.,0.)) #tau_v and tau_th, actual input to apply
        self.e = np.array((0.,0.,0.)) #error on trajectory tracking or desired p and theta value (ex,ey,etheta)
    
    def step_simulation(self,dt,tau_v,tau_th):
        self.tau_v = tau_v
        self.tau_th = tau_th
        vd = tau_v /self.m
        wd = tau_th/self.Iz
        self.update_state_Euler(dt,vd,wd)
    
    def step_simulation_KIN(self,dt,vc,wc):
        self.update_state_Euler_KIN(dt,vc,wc)
    
    def set_backstepping_gains(self, k1, lamb1,lamb2,kbv,kbw):
        self.k1 = k1 #k of eqn. (3.67)
        self.lamb1 = lamb1
        self.lamb2 = lamb2
        self.kbv = kbv
        self.kbw = kbw
    
    def trajectory_tracking_backstepping_step1(self,dt, p_des, v_des, theta_des, w_des):
        """[summary]

        Parameters
        ----------
        dt : [float]
            time from the previous step
        p_des : [np.array((2,)float)]
            desired position\n
        v_des : [float]
            desired forward speed\n
        theta_des : [float]
            desired orientation\n
        w_des : [float]
            desired angular speed

        Warnings
        -------
        trajectory must be consistent, v_des and w_des must be the derivatives of ||p_des|| and theta_des
        """
        # pt and vt are the desired position and velocity
        # eqn (3.68) of the notes, pag.91
        e1 = -(p_des[0] - self.p[0])
        e2 = -(p_des[1] - self.p[1])
        e3 = -(theta_des - self.theta)
        vc = v_des*np.cos(e3) - self.lamb1 * e1
        wc = w_des - e2*v_des/self.k1 - self.lamb2 * np.sin(e3)
        self.back1d = (np.array((vc, wc))-self.back1) / dt
        self.back1 = np.array((vc, wc))
        self.e = np.array((e1,e2,e3))
    
    def trajectory_tracking_backstepping_step2(self):
        #eqns(3.73 and next ones)
        tau_v = self.m * (self.back1d[0] - self.kbv*(self.v-self.back1[0]) -self.e[0])
        tau_th = self.Iz * (self.back1d[1] - self.kbw*(self.w-self.back1[1]) - self.k1* np.sin(self.e[2]))
        return tau_v, tau_th

    def draw_artists(self,fig,ax):
        radius = 0.1
        radius_extra = 0.12
        #patches
        circle_patch = matplotlib.patches.CirclePolygon(
            list(self.p), radius=radius, linewidth=1,
            figure=fig, ec="xkcd:black", fc="xkcd:light teal", fill=True)
        patch_list = [circle_patch]
        #lines
        end_line = self.p + radius_extra*np.array((np.cos(self.theta), np.sin(self.theta)))
        indicator_line = matplotlib.lines.Line2D(
            [self.p[0],end_line[0]],[self.p[1],end_line[1]],
            linewidth=2, figure=fig, color="xkcd:salmon")
        line_list = [indicator_line]
        for patch in patch_list:
            ax.add_patch(patch)
        for line in line_list:
            ax.add_line(line)



        