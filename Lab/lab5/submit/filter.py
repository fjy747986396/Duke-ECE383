from klampt import *
from klampt.math import vectorops,so3,se3
from common import *
import random
#this may be useful...
import numpy as np

##################### SETTINGS ########################

# random_seed = 1
random_seed = random.seed()
verbose = True
GRAVITY = 9.81
BALL_RADIUS = 0.1

########### GIVEN KALMAN IMPLEMENTATION ###############
# def kalman_filter_update(prior_mean,prior_cov,F,g,SigmaX,H,j,SigmaZ,z):
#     if isinstance(SigmaX,(int,float)):
#         SigmaX = np.eye(len(prior_mean))*SigmaX
#     if isinstance(SigmaZ,(int,float)):
#         SigmaZ = np.eye(len(z))*SigmaZ
#     muprime = np.dot(F,prior_mean)+g 
#     covprime = np.dot(F,np.dot(prior_cov,F.T))+SigmaX 
#     C = np.dot(H,np.dot(covprime,H.T))+SigmaZ    
#     zpred = np.dot(H,muprime)+j   
#     K = np.dot(covprime,np.dot(H.T,np.linalg.pinv(C))) 
#     mu = muprime + np.dot(K,z-zpred)  
#     # print(z, zpred)
#     cov = np.dot(np.eye(covprime.shape[0])-np.dot(K,H),covprime)  
#     return (mu,cov) 
################ STATE ESTIMATION #####################

def VelEst(pos_xyz, delta_t, past_num_xy, past_num_z):
    # xy vel estimate via line fitting  
    # p = vt + p_o
    pos_xy = np.matrix(pos_xyz)[-past_num_xy:,:2]
    pos_z = np.matrix(pos_xyz)[-past_num_z:,2]
    A_1 = np.hstack((np.matrix(range(pos_xy.shape[0])).T * delta_t, np.ones((pos_xy.shape[0],1))))
    xy_vels = (np.linalg.pinv(A_1) * pos_xy)[0,:]
    # z vel estimate via 2nd order poly fitting 
    # p = vt + kt**2.0 + p_o 
    A_2 = np.hstack((np.matrix(range(pos_z.shape[0])).T * delta_t,
    np.square(np.matrix(range(pos_z.shape[0])).T * delta_t),
    np.ones((pos_z.shape[0],1))))
    z_vel = -(np.linalg.pinv(A_2) * np.flipud(pos_z))[0,:]
    return xy_vels.tolist()[0] + [z_vel.tolist()[0][0] - GRAVITY * delta_t]

class OmniscientStateEstimator:
    """A hack state estimator that gives perfect state information from
    OmniscientObjectOutput readings."""
    def __init__(self):
        self.reset()
        return
    def reset(self):
        pass
    def update(self,o):
        """Produces an updated MultiObjectStateEstimate given an OmniscientObjectOutput
        sensor reading."""
        assert isinstance(o,OmniscientObjectOutput),"OmniscientStateEstimator only works with an omniscientObjectOutput object"
        estimates = [ObjectStateEstimate(n,p+v) for n,p,v in zip(o.names,o.positions,o.velocities)]
        return MultiObjectStateEstimate(estimates)

class PositionStateEstimator:
    def __init__(self):
        self.dt = 0.02
        self.reset()
        self.previous_estimates = None
        self.previous_observations = dict() 
        self.thrown = dict()
        # set the kalman filter parameters 
        self.A = np.vstack((np.hstack((np.zeros((3,3)),np.eye(3))), np.zeros((3,6))))
        self.dyn_noise = np.vstack((np.hstack((np.eye(3) * (0.5 * self.dt**2 * 0.008)**2, np.zeros((3,3)))),
                                    np.hstack((np.zeros((3,3)), np.eye(3) * 0.008)))) # covariance 
        self.F = np.eye(self.A.shape[0]) + self.A * self.dt 
        self.H = np.hstack((np.eye(3), np.zeros((3,3))))
        self.ob_noise = np.eye(3) * (0.1/2.33)**2.0 # covariance
        self.vel_est_point_num = [7,7]
        return
    
    def reset(self):
        pass
    def update(self,o):
        """Produces an updated MultiObjectStateEstimate given an ObjectPositionOutput
        sensor reading."""
        global counter, cur_name 
        assert isinstance(o,ObjectPositionOutput),"PositionStateEstimator only works with an ObjectPositoinOutput object"
        #TODO: Fill me in for Problem 1
        estimates = []
        for n,p in zip(o.names,o.positions):
            if self.previous_estimates is not None:
                total_z_acc = -GRAVITY
                if not self.thrown[n]:
                    # check if the ball is thrown 
                    if (p[0]**2.0 + p[1]**2.0 + p[2] ** 2.0)**0.5 < 0.5 or p[2] < 0.4:
                        total_z_acc = 0.0
                        initial_cov = np.vstack((np.zeros((3,6)),np.hstack((np.zeros((3,3)), np.eye(3) * (0.1/2.33) ** 2))))
                        estimates.append(ObjectStateEstimate(n, p + [0,0,0], initial_cov))
                    else:
                        self.thrown[n] = True
                        initial_cov = np.vstack((np.zeros((3,6)),np.hstack((np.zeros((3,3)), np.eye(3) * (0.1/2.33) ** 2))))
                        estimates.append(ObjectStateEstimate(n,
                        p + VelEst(self.previous_observations[n], self.dt, self.vel_est_point_num[0], self.vel_est_point_num[1]),
                        initial_cov))
                    self.previous_observations[n].append(p)    
                   
                else:
                    cur_g = np.vstack((np.zeros((5,1)), np.array([total_z_acc]))) * self.dt
                    cur_name = n
                    previous_estimate = np.array(self.previous_estimates.get(n).x).reshape(cur_g.shape)
                    previous_covariance = self.previous_estimates.get(n).cov 
                    current_obs = np.array(p).reshape((self.H.shape[0],1))
                    updated_states, updated_cov = kalman_filter_update(previous_estimate, previous_covariance,
                        self.F, cur_g, self.dyn_noise,
                        self.H, 0.0, self.ob_noise, 
                        current_obs)
                    estimates.append(ObjectStateEstimate(n,list(updated_states[:,0]), updated_cov))
            else:
                self.previous_observations[n] = []
                self.thrown[n] = False
                initial_cov = np.vstack((np.zeros((3,6)),np.hstack((np.zeros((3,3)), np.eye(3) * (0.1/2.33) ** 2))))
                estimates.append(ObjectStateEstimate(n, p + [0,0,0], initial_cov))
        self.previous_estimates = MultiObjectStateEstimate(estimates)
        return MultiObjectStateEstimate(estimates)

class BlobStateEstimator:
    def __init__(self):
        cameraRot = [0,-1,0,0,0,-1,1,0,0]
        #at goal post, pointing a bit up and to the left
        self.Tsensor = (so3.mul(so3.rotation([0,0,1],0.20),so3.mul(so3.rotation([0,-1,0],0.25),cameraRot)),[-2.55,-1.1,0.25])
        self.T_r = np.matrix(self.Tsensor[0]).reshape((3,3)).T
        self.T_t = np.matrix(self.Tsensor[1]).reshape((3,1))
        self.fov = 90
        self.w,self.h = 320,240
        self.dmax = 5
        self.dt = 0.02
        self.f = (0.5 * self.w) / np.tan(self.fov/2.0/180 * np.pi)
        # Kalman Filter parameters 
        self.A = np.vstack((np.hstack((np.zeros((3,3)),np.eye(3))), np.zeros((3,6))))
        self.dyn_noise = np.vstack((np.hstack((np.eye(3) * (0.5 * self.dt**2 * 0.008)**2, np.zeros((3,3)))),
                                    np.hstack((np.zeros((3,3)), np.eye(3) * 0.008)))) # covariance
        self.F = np.eye(self.A.shape[0]) + self.A * self.dt
        self.g = np.vstack((np.zeros((5,1)), np.matrix([-GRAVITY]))) * self.dt
        self.H_jaccobian = None # use calJaccobian to calculate this when there is some estimated state to use
        self.obs_noise = np.diag([0.25, 0.25, 0.5, 0.5])
        # prior 
        self.previous_estimates = dict() 
        self.previous_cov = dict()
        return
    
    def reset(self):
        pass
    
    def calJaccobian(self, in_states, r_o):
        ''' in_states is a list'''
        f_x, f_y = self.f, self.f
        x_c, y_c, z_c = in_states[0], in_states[1], in_states[2]
        dh_1 = np.matrix([f_x/z_c, 0, -f_x * x_c * z_c ** (-2.0)])
        dh_2 = np.matrix([0, f_y/z_c, -f_y * y_c * z_c ** (-2.0)])
        dh_3 = np.matrix([0, 0, - 2 * f_x * r_o * z_c ** (-2.0)])
        dh_4 = np.matrix([0, 0, - 2 * f_y * r_o * z_c ** (-2.0)])
        dh = np.vstack((dh_1, dh_2, dh_3, dh_4))
        dh_dp = dh * np.linalg.pinv(self.T_r)
        dh_dv = np.zeros((dh_dp.shape[0],3))
        return np.hstack((dh_dp, dh_dv))
        
    def hTransfer(self, p_c, r_o, x_m, y_m):
        ''' transfer the states in the world to the states in image '''
        f_x, f_y = self.f, self.f 
        x_c, y_c, z_c = p_c[0], p_c[1], p_c[2]
        x_im = f_x * x_c/z_c + x_m
        y_im = f_y * y_c/z_c + y_m
        w = f_x * 2 * r_o/z_c 
        h = f_y * 2 * r_o/z_c
        return np.matrix([x_im, y_im, w, h]).T
    
    def hInvPosSolver(self, first_obs, r_o, x_m, y_m):
        ''' 
        The ball radius is given, z_c can be solved first with w and h
        then subsititute it back to x and y relations of H can solve for x and y 
        '''
        f_x, f_y = self.f, self.f 
        x_im, y_im = first_obs[0], first_obs[1]
        # solve for z_c 
        A_zc = np.matrix([2*f_x * r_o, 2 * f_y * r_o]).T
        b_zc = np.matrix(first_obs[-2:]).T
        z_c = 1/(np.linalg.pinv(A_zc) * b_zc)[0,0]
        A_xy = np.diag([f_x/z_c, f_y/z_c])
        b_xy = np.matrix([x_im - x_m, y_im - y_m]).T
        return np.vstack((np.linalg.pinv(A_xy) * b_xy, np.matrix([z_c])))
    
    def velEst(self, pos_xyz, delta_t, past_num_xy, past_num_z):
        # xy vel estimate via line fitting  
        # p = vt + p_o
        pos_xy = np.matrix(pos_xyz)[-past_num_xy:,:2]
        pos_z = np.matrix(pos_xyz)[-past_num_z:,2]
        A_1 = np.hstack((np.matrix(range(pos_xy.shape[0])).T * delta_t, np.ones((pos_xy.shape[0],1))))
        xy_vels = (np.linalg.pinv(A_1) * pos_xy)[0,:]
        if xy_vels[0,0] > 0:
            xy_vels[0,0] = - xy_vels[0,0]
        # z vel estimate via 2nd order poly fitting 
        # p = vt + kt**2.0 + p_o 
        A_2 = np.hstack((np.matrix(range(pos_z.shape[0])).T * delta_t,
        np.square(np.matrix(range(pos_z.shape[0])).T * delta_t),
        np.ones((pos_z.shape[0],1))))
        z_vel = -(np.linalg.pinv(A_2) * np.flipud(pos_z))[0,:]
        return xy_vels.tolist()[0] + [z_vel.tolist()[0][0] - GRAVITY * delta_t]
    
    def statesCollection(self, each_blob):
        ''' 
        input: blob object
        output: states and covariance(None)
        Use invh to caculate the position of the ball, the vels are assumed to be zeros
        '''
        cur_obs = [each_blob.x, each_blob.y, each_blob.w, each_blob.h]
        cur_p_c = self.hInvPosSolver(cur_obs, BALL_RADIUS, self.w/2.0, self.h/2.0)
        cur_states = se3.apply(self.Tsensor,cur_p_c.T.tolist()[0]) + [0,0,0]
        cur_cov = None
        return cur_states, cur_cov 
    
    def statesInitialization(self, each_blob, pre_xyz, num_pre_points):
        '''
        input: blob object from observation, previous collected positions, number of points for line fitting
        output: position by hInvPosSolver, and vels approximated by line fitting
        '''
        cur_obs = [each_blob.x, each_blob.y, each_blob.w, each_blob.h]
        cur_p_c = self.hInvPosSolver(cur_obs, BALL_RADIUS, self.w/2.0, self.h/2.0)
        cur_p_w = se3.apply(self.Tsensor,cur_p_c.T.tolist()[0])
        cur_p_vel = self.velEst(pre_xyz, self.dt, num_pre_points, num_pre_points)
        cur_states = cur_p_w + cur_p_vel 
        cur_cov = np.vstack((np.zeros((3,6)),np.hstack((np.zeros((3,3)), np.eye(3) * (0.1/2.33) ** 2))))
        return cur_states, cur_cov
        
    def statesEstimation(self, each_blob, previous_estimate, previous_covariance):
        '''
        input: previous_estimate(np.matrix)
        output: states and covariance estimated by kalman filter
        '''
        pre_pos_w = previous_estimate[:3,0].T.tolist()[0]
        previous_p_c = se3.apply(se3.inv(self.Tsensor), pre_pos_w)
        cur_obs = np.matrix([each_blob.x, each_blob.y, each_blob.w, each_blob.h]).T
        cur_H = self.calJaccobian(previous_p_c , BALL_RADIUS)
        cur_J = self.hTransfer(previous_p_c, BALL_RADIUS, 160, 120) - cur_H * previous_estimate 
        updated_states, cur_cov = kalman_filter_update(previous_estimate, previous_covariance,
                self.F, self.g, self.dyn_noise,
                cur_H, cur_J, self.obs_noise, 
                cur_obs)
        cur_states = updated_states.T.tolist()[0]
        return cur_states, cur_cov
    
    def update(self,o):
        """Produces an updated MultiObjectStateEstimate given an CameraColorDetectorOutput
        sensor reading."""
        assert isinstance(o,CameraColorDetectorOutput),"BlobStateEstimator only works with an CameraColorDetectorOutput object"
        #TODO: Fill me in for Problem 2
        num_pre_points = 10
        estimates = []
        for each_blob in o.blobs:
            cur_color = each_blob.color
            # initialize the states record
            if cur_color not in self.previous_estimates.keys():
                self.previous_estimates[cur_color] = []
                self.previous_cov[cur_color] = []
            # collect data until there are enough points for vel estimation
            if len(self.previous_estimates[cur_color]) < num_pre_points:
                cur_states, cur_cov = self.statesCollection(each_blob)
            # estimate the vel for initialization 
            elif len(self.previous_estimates[cur_color]) == num_pre_points: 
                pre_xyz = np.matrix(self.previous_estimates[cur_color])[:,:-3]
                cur_states, cur_cov = self.statesInitialization(each_blob, pre_xyz, num_pre_points)
            elif len(self.previous_estimates[cur_color]) > num_pre_points: # use kalman filter 
                previous_estimate = np.matrix(self.previous_estimates[cur_color][-1]).T
                previous_covariance = self.previous_cov[cur_color][-1]
                cur_states, cur_cov = self.statesEstimation(each_blob, previous_estimate, previous_covariance)
            self.previous_estimates[cur_color].append(cur_states)
            self.previous_cov[cur_color].append(cur_cov)
            estimates.append(ObjectStateEstimate(cur_color, cur_states))
        return MultiObjectStateEstimate(estimates)


positionStateEstimator = PositionStateEstimator()
blobStateEstimator = BlobStateEstimator()


def reset():
    initVis()
    positionStateEstimator.reset()
    blobStateEstimator.reset()

def update(sensorType,observation):
    """Produces an updated MultiObjectStateEstimate given sensor input.

    Input:
    - sensorType: "omniscient", "position", or "blobdetector".
    - observation: one of the types of sensor input.

    Return value: MultiObjectStateEstimate giving the estimated state of all
    observed objects.
    """
    if sensorType == 'omniscient':
        omniscientObjectState = OmniscientStateEstimator().update(observation)
        updateVis(omniscientObjectState)
        return omniscientObjectState
    elif sensorType == 'position':
        positionObjectState = positionStateEstimator.update(observation)
        updateVis(positionObjectState)
        return positionObjectState            
    elif sensorType == 'blobdetector':
        blobObjectState = blobStateEstimator.update(observation)
        updateVis(blobObjectState)
        return blobObjectState
    return MultiObjectStateEstimate([])
        

def initVis():
    """If you want to do some visualization, initialize it here.
        TODO: You may consider visually debugging some of your code here, along with updateVis().
    """
    pass
    
def updateVis(objectEstimates):
    """This gets called by update()
    TODO: You may consider visually debugging some of your code here, along with initVis().

    The current code draws gravity-inflenced arcs leading from all the
    object position / velocity estimates from your state estimator. 
    """
    gravity = GRAVITY
    if objectEstimates:
        for o in objectEstimates.objects:
            #draw a point
            kviz.update_sphere("object_est"+str(o.name),o.x[0],o.x[1],o.x[2],0.03)
            kviz.set_color("object_est"+str(o.name),o.name[0],o.name[1],o.name[2])
            #draw an arc
            trace = []
            x = [o.x[0],o.x[1],o.x[2]]
            v = [o.x[3],o.x[4],o.x[5]]
            for i in range(20):
                t = i*0.05
                trace.append(vectorops.sub(vectorops.madd(x,v,t),[0,0,0.5*gravity*t*t]))
                if trace[-1][2] < 0: break
            kviz.remove("object_trace"+str(o.name))
            kviz.add_polyline("object_trace"+str(o.name),trace);
            kviz.set_color("object_trace"+str(o.name),o.name[0],o.name[1],o.name[2])

    
