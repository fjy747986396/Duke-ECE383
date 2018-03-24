from klampt import *
from klampt.model import ik
from klampt.math import vectorops,so3,se3
from common import *
import math
import random
#this may be useful...
#import numpy as np

##################### SETTINGS ########################
event = 'A'

#difficulty 
# difficulty = 'easy'
# difficulty = 'medium'
difficulty = 'hard'

omniscient_sensor = False

# random_seed = 123345
random_seed = random.seed()
WALL_Y = 1
WALL_Z = 1.5
WALL_XS = [-1.3, -1.5]
PREPARE_LOCATIONS = {'upper right outer': (-1.3, 0.8, 1.3), # u
                     'upper left outer':(-1.3, -0.8, 1.3), #u
                     'center right outer': (-1.3, 0.8, 1), # u
                     'center left outer': (-1.3, -0.8, 1), #u
                     'lower right outer':(-1.3, 0.8, 0.5), # u
                     'lower left outer':(-1.3, -0.8, 0.5), # u 
                     'upper right inner': (-1.3, 0.4, 1.3), # u
                     'upper left inner': (-1.3, -0.4, 1.3), # u
                     'center right inner': (-1.3, 0.4, 1), # u 
                     'center left inner': (-1.3, -0.4, 1), # u
                     'lower right inner': (-1.3, 0.4, 0.5), # u
                     'lower left inner': (-1.3, -0.4, 0.5), # u
                     'bottom right outer': (-1.3, 0.8, 0.25),
                     'bottom left outer': (-1.3, -0.8, 0.25),
                     'bottom right inner': (-1.3, 0.4, 0.25),
                     'bottom left inner': (-1.3, -0.4, 0.25),
                     'top center': (-1.3, 0, 1.3), # u
                     'center center': (-1.3, 0, 0.5)} # u 

PRE_CAL_CONFIG = {(-1.3, 0.8, 1.3): [0.0, 0.8203047484373349, 0.9756390518648302, 0.15009831567151233, -0.026179938779914945, -0.2181661564992912, 0.0],
                  (-1.3, -0.8, 1.3):[0.0, -0.8744099552491591, 0.9756390518648302, 0.15009831567151233, -0.026179938779914945, -0.2181661564992912, 0.0],
                  (-1.3, 0.8, 1):[0.0, 0.8203047484373349, 1.181587903600161, 0.15009831567151233, -0.026179938779914945, -0.2181661564992912, 0.0],
                  (-1.3, -0.8, 1):[0.0, -0.9267698328089891, 1.181587903600161, 0.15009831567151233, -0.026179938779914945, -0.2181661564992912, 0.0],
                  (-1.3, 0.8, 0.5):[0.0, 0.767944870877505, 1.5917402778188285, 0.15009831567151233, -0.026179938779914945, 1.3840460968315034, 0.0],
                  (-1.3, -0.8, 0.5):[0.0, -0.8203047484373349, 1.5917402778188285, 0.15009831567151233, -0.026179938779914945, 1.3840460968315034, 0.0],
                  (-1.3, -0.4, 1.3):[0.0, -0.609119908946021, 0.5253441048502933, -0.32114058236695664, -0.026179938779914945, -0.2181661564992912, 0.0],
                  (-1.3, 0.4, 1.3):[0.0, 0.5026548245743669, 0.5253441048502933, -0.32114058236695664, -0.026179938779914945, -0.2181661564992912, 0.0],
                  (-1.3, 0.4, 1):[0.0, 0.39793506945470714, 1.5917402778188285, 0.15009831567151233, -0.026179938779914945, 1.3840460968315034, 0.0],
                  (-1.3, -0.4, 1):[0.0, -0.5567600313861911, 1.5917402778188285, 0.15009831567151233, -0.026179938779914945, 1.3840460968315034, 0.0],
                  (-1.3, 0.8, 0.25):[0.0, 0.8744099552491591, 1.879719604397893, 0.15009831567151233, -0.026179938779914945, 1.3840460968315034, 0.0],
                  (-1.3, -0.8, 0.25):[0.0, -0.8744099552491591, 1.879719604397893, 0.15009831567151233, -0.026179938779914945, 1.3840460968315034, 0.0],
                  (-1.3, 0.4, 0.25):[0.0, 0.39793506945470714, 1.879719604397893, 0.15009831567151233, -0.026179938779914945, 1.3840460968315034, 0.0],
                  (-1.3, -0.4, 0.25):[0.0, -0.5567600313861911, 1.879719604397893, 0.15009831567151233, -0.026179938779914945, 1.3840460968315034, 0.0],
                  (-1.3, 0, 1.3):[0.0, -0.13264502315156904, 0.4834562028024293, -0.32114058236695664, -0.026179938779914945, -0.2181661564992912, 0.0],
                  (-1.3, 0, 0.5):[0.0, -0.13264502315156904, 0.7295476273336297, -1.5655603390389137, -0.026179938779914945, -0.2181661564992912, 0.0]}
                  
                  
verbose = True
GRAVITY = 9.81
BALL_RADIUS = 0.1
################ STATE ESTIMATION #####################
class MyObjectStateEstimator:
    """Your own state estimator that will provide a state estimate given
    CameraColorDetectorOutput readings."""
    def __init__(self):
        self.reset()
        #TODO: fill this in with your own camera model, if you wish
        self.Tsensor = None
        cameraRot = [0,-1,0,0,0,-1,1,0,0]
        self.w,self.h = 320,240
        self.fov = 90
        self.dmax = 5
        self.dt = 0.02
        if event == 'A':
            #at goal post, pointing a bit up and to the left
            self.Tsensor = (so3.mul(so3.rotation([0,0,1],0.20),so3.mul(so3.rotation([0,-1,0],0.25),cameraRot)),[-2.55,-1.1,0.25])
            # Kalman Filter parameters 
            self.A = np.vstack((np.hstack((np.zeros((3,3)),np.eye(3))), np.zeros((3,6))))
            self.dyn_noise = np.vstack((np.hstack((np.eye(3) * (0.5 * self.dt**2 * 0.008)**2, np.zeros((3,3)))),
                                        np.hstack((np.zeros((3,3)), np.eye(3) * 0.008)))) # covariance
            self.F = np.eye(self.A.shape[0]) + self.A * self.dt
            self.g = np.vstack((np.zeros((5,1)), np.matrix([-GRAVITY]))) * self.dt
            self.H_jaccobian = None # use calJaccobian to calculate this when there is some estimated state to use
            self.obs_noise = np.diag([0.25, 0.25, 0.5, 0.5])
        elif event == 'B':
            #on ground near robot, pointing up and slightly to the left
            self.Tsensor = (so3.mul(so3.rotation([1,0,0],-0.10),so3.mul(so3.rotation([0,-1,0],math.radians(90)),cameraRot)),[-1.5,-0.5,0.25])
            self.w = 640
            self.h = 480
            self.dmax = 10
        else:
            #on ground near robot, pointing to the right
            self.Tsensor = (cameraRot,[-1.5,-0.5,0.25])
        self.T_r = np.matrix(self.Tsensor[0]).reshape((3,3)).T
        self.T_t = np.matrix(self.Tsensor[1]).reshape((3,1))
        self.f = (0.5 * self.w) / np.tan(self.fov/2.0/180 * np.pi)
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
        z_vel = (np.linalg.pinv(A_2) * np.flipud(pos_z))[0,:]
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
            # approximate the vel for initialization 
            elif len(self.previous_estimates[cur_color]) == num_pre_points: 
                pre_xyz = np.matrix(self.previous_estimates[cur_color])[:,:-3]
                cur_states, cur_cov = self.statesInitialization(each_blob, pre_xyz, num_pre_points)
            # use kalman filter
            elif len(self.previous_estimates[cur_color]) > num_pre_points:  
                previous_estimate = np.matrix(self.previous_estimates[cur_color][-1]).T
                previous_covariance = self.previous_cov[cur_color][-1]
                cur_states, cur_cov = self.statesEstimation(each_blob, previous_estimate, previous_covariance)
            self.previous_estimates[cur_color].append(cur_states)
            self.previous_cov[cur_color].append(cur_cov)
            estimates.append(ObjectStateEstimate(cur_color, cur_states))
        return MultiObjectStateEstimate(estimates)

############## CATCH LOCATION ESTIMATION ##############
class CatchLocationEstimator:
    def __init__(self):
        self.estimates_record = dict()
        self.wall_xs = WALL_XS
        self.avg_factor = 0.5
        self.bounce_factor = [0.98, 0.98, 0.6]

    def addRecord(self, new_record):
        cur_name = new_record.name
        cur_pos = new_record.meanPosition()
        # ball not hit the imagined wall yet
        if (cur_pos[0] + BALL_RADIUS) > self.wall_xs[0]:
            cur_vel = new_record.meanVelocity()
            cur_catch = self.calCatchPos(cur_pos, cur_vel, self.wall_xs, x_limit = -1)
            if cur_name not in self.estimates_record.keys():
                if cur_catch is not None:
                    self.estimates_record[cur_name] = [cur_catch]
            else:
                self.estimates_record[cur_name].append(cur_catch)
        elif (cur_pos[0] + BALL_RADIUS) < self.wall_xs[1]:
            self.estimates_record[cur_name] = []

    def findOptPrepareLoc(self, hit_xyz, prep_loc_dict):
        best_loc = None
        min_dist = float('inf')
        for each_key in prep_loc_dict.keys():
            cur_prep = prep_loc_dict[each_key]
            cur_dist = vectorops.norm(vectorops.sub(hit_xyz, cur_prep))
            if cur_dist < min_dist:
                best_loc = each_key
                min_dist = cur_dist
        return best_loc

    def calCatchPos(self, p_xyz, v_xyz, wall_xs, x_limit = 0):
        '''
        find the land postion  wall_xs used to find time to hit each wall 
        and uses those time to find the catch location 
        '''
        catch_xyz = np.zeros((len(wall_xs),3))
        if v_xyz[0] == 0 and v_xyz[1] == 0 and v_xyz[2] == 0:
            if p_xyz[1] < 0:
                catch_loc =  np.matrix(PREPARE_LOCATIONS['center left inner'])
            else:
                catch_loc = np.matrix(PREPARE_LOCATIONS['center right inner'])
            for idx in range(catch_xyz.shape[0]):
                catch_xyz[idx, :] = catch_loc
            return catch_xyz
        
        for idx in range(len(wall_xs)):
            x = wall_xs[idx]    
            total_t = (x - p_xyz[0])/v_xyz[0]
            pred_y = p_xyz[1] + total_t * v_xyz[1]
            pred_z = p_xyz[2] + total_t * v_xyz[2] - 0.5 * GRAVITY * total_t ** 2.0
            # No hitting ground use dynamic equation to predict 
            if pred_z > 0:
                catch_xyz[idx, :] = [x, pred_y, pred_z]
                p_y, p_z = pred_y, pred_z
            # Hitting the ground using iterative method for prediction 
            else:
                p_x, p_y, p_z = tuple([p_xyz[p_idx] for p_idx in range(len(p_xyz))])
                v_x, v_y, v_z = tuple([v_xyz[v_idx] for v_idx in range(len(v_xyz))])
                dt = 0.01
                counter = 0
                while p_x > x:
                    p_x = p_x + dt * v_x
                    p_y = p_y + dt * v_y
                    p_z = p_z + dt * v_z - 0.5 * GRAVITY * dt ** 2.0
                    v_z = v_z - dt * GRAVITY
                    if p_z < 0: 
                        p_z = -p_z
                        v_z = - self.bounce_factor[2] * v_z 
                        v_x = self.bounce_factor[0] * v_x
                        v_y = self.bounce_factor[1] * v_y
                    counter += 1
                    if counter > 500: break
                # decide if it is preparing phase or accurate blocking phase 
            # consider the depth of the ground 
            if p_z < 0.15: p_z = 0.15
            if (p_xyz[0] - BALL_RADIUS) > x_limit:
                best_next_state = self.findOptPrepareLoc([x, p_y, p_z], PREPARE_LOCATIONS)
                catch_xyz[idx, :] = list(PREPARE_LOCATIONS[best_next_state])
            else:
                catch_xyz[idx, :] = [x, p_y, p_z]
        return catch_xyz

    def getRecord(self,name):
        if name in self.estimates_record.keys():
            return self.estimates_record[name]
        else:
            return []

CATCH_EST = CatchLocationEstimator()
################### CONTROLLER ########################
class MyController:
    """Attributes:
    - world: the WorldModel instance used for planning.
    - objectStateEstimator: a StateEstimator instance, which you may set up.
    - state: a string indicating the state of the state machine. TODO:
      decide what states you want in your state machine and how you want
      them to be named.
    """
    def __init__(self,world,robotController):
        self.world = world
        self.objectStateEstimator = None
        self.state = None
        self.robotController = robotController
        self.ini_config = PRE_CAL_CONFIG[PREPARE_LOCATIONS['center center']]
        self.dt = 0.02
        self.vel_limits = robotController.model().getVelocityLimits()
        self.reset(robotController)
        self.ready = False
    
    def reset(self,robotController):
        """Called on initialization, and when the simulator is reset.
        TODO: You may wish to fill this in with custom initialization code.
        """
        self.objectStateEstimator = MyObjectStateEstimator()
        self.objectEstimates = None
        #TODO: you may want to do more here to set up your
        #state machine and other initial settings of your controller.
        #The 'waiting' state is just a placeholder and you are free to
        #change it as you see fit.
        self.qdes = self.ini_config
        robotController.setMilestone(self.qdes)
        self.initVis()

    def configSolver(self, robot, catch_location):
        ''' 
        solve for all the possibilities and find the one with lowest cost compare with the current config
        '''
        # Pre-calculated waiting and prepare configs 
        if tuple(catch_location) in PRE_CAL_CONFIG.keys():
            return [PRE_CAL_CONFIG[tuple(catch_location)], 0.0]
        current_config = robot.getConfig()
        opt_sol = [current_config, float('inf')]
        # norm solver 
        for link_idx in range(4,robot.numLinks()-1):
            cur_end_link = robot.link(link_idx)
            for link_pos in [0.02 * mul for mul in range(1)]:
                cur_goal = ik.objective(cur_end_link, local = [(link_pos,0,0)], world = [catch_location])
                if ik.solve(cur_goal):
                    cur_sol = robot.getConfig()
                    cur_dist = robot.distance(cur_sol, current_config)
                    if cur_dist < opt_sol[1]: opt_sol[0], opt_sol[1] = cur_sol, cur_dist
        return opt_sol

    def calOptCatch(self, robot, catch_locations):
        '''
        when there are several possible catch on different wall levels and for different balls
        choose the one with lowest cost
        '''
        best_catch = [robot.getConfig(), float('inf')]
        for loc_idx in range(len(catch_locations)):
            cur_loc = catch_locations[loc_idx]
            cur_opt = self.configSolver(robot, cur_loc)
            if cur_opt[1] < best_catch[1]: best_catch[0], best_catch[1] = cur_opt[0], cur_opt[1]
        return best_catch

    def myPlayerLogic(self,
                      dt,
                      sensorReadings,
                      objectStateEstimate,
                      robotController):
        """
        TODO: fill this out to updates the robot's low level controller
        in response to a new time step.  This is allowed to set any
        attributes of MyController that you wish, such as self.state.
        
        Arguments:
        - dt: the simulation time elapsed since the last call
        - sensorReadings: the sensor readings given on the current time step.
          this will be a dictionary mapping sensor names to sensor data.
          The name "blobdetector" indicates a sensor reading coming from the
          blob detector.  The name "omniscient" indicates a sensor reading
          coming from the omniscient object sensor.  You will not need to
          use raw sensor data directly, if you have a working state estimator.
        - objectStateEstimate: a MultiObjectStateEstimate class (see
          stateestimation.py) produced by the state estimator.
        - robotController: a SimRobotController instance giving access
          to the robot's low-level controller.  You can call any of the
          methods.  At the end of this call, you can either compute some
          PID command via robotController.setPIDCommand(), or compute a
          trajectory to execute via robotController.set/addMilestone().
          (if you are into masochism you can use robotController.setTorque())
        """
        robot = robotController.model()
        # Solver 
        possible_catch = []
        # if no result return to the center prepare location 
        num_obs = len(objectStateEstimate.objects)
        if num_obs == 0: 
            self.qdes = self.ini_config
        else:
            for o in objectStateEstimate.objects:
                if (o.meanVelocity()[0] < -1 
                    and abs(o.meanPosition()[1]) < WALL_Y 
                    and o.meanPosition()[0] > WALL_XS[1] 
                    and o.meanPosition()[0] < 2) or vectorops.norm(vectorops.sub(o.meanVelocity(), [0]*3)) == 0:
                    CATCH_EST.addRecord(o)
                    catch_array = CATCH_EST.getRecord(o.name)
                    if catch_array != []:
                        last_record = catch_array[-1]
                        for idx in range(last_record.shape[0]):
                            possible_catch.append(last_record[idx,:].tolist())
                opt_result = self.calOptCatch(robot, possible_catch)
                self.qdes = opt_result[0]
        robotController.setMilestone(self.qdes)
        return
        
    def loop(self,dt,robotController,sensorReadings):
        """Called every control loop (every dt seconds).
        Input:
        - dt: the simulation time elapsed since the last call
        - robotController: a SimRobotController instance. Use this to get
          sensor data, like the commanded and sensed configurations.
        - sensorReadings: a dictionary mapping sensor names to sensor data.
          The name "blobdetector" indicates a sensor reading coming from the
          blob detector.  The name "omniscient" indicates a sensor reading coming
          from the omniscient object sensor.
        Output: None.  However, you should produce a command sent to
          robotController, e.g., robotController.setPIDCommand(qdesired).
        """
        multiObjectStateEstimate = None
        if self.objectStateEstimator and 'blobdetector' in sensorReadings:
            multiObjectStateEstimate = self.objectStateEstimator.update(sensorReadings['blobdetector'])
            self.objectEstimates = multiObjectStateEstimate
            #multiObjectStateEstimate is now a MultiObjectStateEstimate (see common.py)
        if 'omniscient' in sensorReadings:
            omniscientObjectState = OmniscientStateEstimator().update(sensorReadings['omniscient'])
            #omniscientObjectStateEstimate is now a MultiObjectStateEstimate (see common.py)
            multiObjectStateEstimate  = omniscientObjectState
            #if you want to visualize the traces, you can uncomment this
            #self.objectEstimates = multiObjectStateEstimate

        self.myPlayerLogic(dt,
                           sensorReadings,multiObjectStateEstimate,
                           robotController)
        self.updateVis()
        return

    def initVis(self):
        """If you want to do some visualization, initialize it here.
            TODO: You may consider visually debugging some of your code here, along with updateVis().
        """
        ### Uncomment the following to see the visualization
        # goal_y = 1
        # goal_z = 1.5
        # for wall_idx in range(len(WALL_XS)):
        #     cur_wall_x = WALL_XS[wall_idx]

        #     kviz.add_line('top' + str(wall_idx), cur_wall_x, goal_y, goal_z,
        #               cur_wall_x, -goal_y, goal_z)
        #     kviz.add_line('right' + str(wall_idx), cur_wall_x, goal_y, goal_z,
        #                       cur_wall_x, goal_y, 0)
        #     kviz.add_line('left' + str(wall_idx), cur_wall_x, -goal_y, goal_z,
        #                       cur_wall_x, -goal_y, 0)
        #     kviz.set_color('top' + str(wall_idx), 1, 0.8, 0.3)      
        #     kviz.set_color('right' + str(wall_idx), 1, 0.8, 0.3) 
        #     kviz.set_color('left' + str(wall_idx), 1, 0.8, 0.3)         
        # for each_loc in PREPARE_LOCATIONS.keys():
        #     cur_loc = PREPARE_LOCATIONS[each_loc]
        #     kviz.add_sphere('test' + each_loc, cur_loc[0], cur_loc[1], cur_loc[2], 0.1)
        #     kviz.set_color('test' + each_loc, 1,1,1)
        pass
        
    def updateVis(self):
        """This gets called every control loop.
        TODO: You may consider visually debugging some of your code here, along with initVis().

        For example, to draw a ghost robot at a given configuration q, you can call:
          kviz.add_ghost()  (in initVis)
          kviz.set_ghost(q) (in updateVis)

        The current code draws gravity-inflenced arcs leading from all the
        object position / velocity estimates from your state estimator.  Event C
        folks should set gravity=0 in the following code.
        """
        ### Uncomment the following to see the visualization
        # if self.objectEstimates:
        #     for o in self.objectEstimates.objects:
        #         #draw a point
        #         kviz.update_sphere("object_est"+str(o.name),o.x[0],o.x[1],o.x[2],0.03)
        #         kviz.set_color("object_est"+str(o.name),o.name[0],o.name[1],o.name[2])
        #         #draw an arc
        #         trace = []
        #         x = [o.x[0],o.x[1],o.x[2]]
        #         v = [o.x[3],o.x[4],o.x[5]]
        #         if event=='C': gravity = 0
        #         else: gravity = GRAVITY
        #         for i in range(20):
        #             t = i*0.05
        #             trace.append(vectorops.sub(vectorops.madd(x,v,t),[0,0,0.5*gravity*t*t]))
        #         kviz.update_polyline("object_trace"+str(o.name),trace);
        #         kviz.set_color("object_trace"+str(o.name),o.name[0],o.name[1],o.name[2])
        #         cur_records = CATCH_EST.getRecord(o.name)
        #         if cur_records != []:
        #             last_record = cur_records[-1]
        #             for idx in range(last_record.shape[0]):
        #                 cur_name = 'Test' + str(idx)
        #                 kviz.update_sphere(cur_name, last_record[idx,0], last_record[idx,1], last_record[idx,2], 0.1)
        #                 kviz.set_color(cur_name, o.name[0],o.name[1],o.name[2])
