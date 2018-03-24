import math
from klampt.math import vectorops
from klampt.model import ik

robot = None
dt = 0.01

#for I terms
pid_integrators = [0,0,0,0]

#higher level controller status... you don't need to use these if you wish
status = "pen up"
current_stroke = 0
current_stroke_progress = 0
stroke_list = [[(0.2,0.05),(0.2,-0.05)]]

def init(world):
    global robot,stroke_list
    robot = world.robot(0)
    print(robot.link(3).getWorldPosition((0,0,0.1)))
    stroke_list = curves()

def getPIDTorqueAndAdvance(q,dq,
                        qdes,dqdes,
                        kP,kI,kD,
                        pid_integrators,dt):
    """ TODO: implement me
    Returns the torques resulting from a set of PID controllers, given:
    - q: current configuration (list of floats)
    - dq: current joint velocities 
    - qdes: desired configuration
    - dqdes: desired joint velocities
    - kP: list of P constants, one per joint
    - kI: list of I constants, one per joint
    - kD: list of D constants, one per joint
    - pid_integrators: list of error integrators, one per joint
    - dt: time step since the last call of this function

    The pid_integrators list should also be updated according to the time step.
    """
    for joint_idx in range(len(pid_integrators)):
        pid_integrators[joint_idx] += (q[joint_idx] - qdes[joint_idx])*dt
    torques = [0 for _ in range(len(q))]
    for i in range(len(torques)):
        #only the P term is computed here...
        torques[i] = (-kP[i]*(q[i] - qdes[i]) 
            - kD[i] * (dq[i] - dqdes[i]) 
            - kI[i] * pid_integrators[i])
        if i == 3: # the pen 
            torques[i] -= robot.link(i).getMass().getMass()*9.8
    return torques

def getTorque(t,q,dq):
    """ TODO: implement me
    Returns a 4-element torque vector given the current time t, the configuration q, and the joint velocities dq to drive
    the robot so that it traces out the desired curves.
    
    Recommended order of operations:
    1. Monitor and update the current status and stroke
    2. Update the stroke_progress, making sure to perform smooth interpolating trajectories
    3. Determine a desired configuration given current state using IK
    4. Send qdes, dqdes to PID controller
    """
    global robot,status,current_stroke,current_stroke_progress,stroke_list
    # ### PID Implementation Arc Test  
    # qdes = [0,0,0,0]
    # dqdes = [0,0,0,0]
    # if t > 0.5:
    #     #move up
    #     qdes[1] = 0.3
    # if t > 1.5:
    #     #drop the last link down
    #     qdes[3] = 0.04
    #     # print(q, qdes)
    # if t > 2.0:
    #     #move down
    #     qdes[1] = -0.3
    # kP = [20,6,10,4]
    # kI = [9,4,5,3]
    # kD = [15,2,0.4,2]
    # ### 
    
    ### Draw Strokes 
    kP = [20,6,10,4]
    kI = [9,4,5,3]
    kD = [15,2,0.4,2]
    joint_max_vel = 0.1
    # interpolation
    interp_frac = dt * t
    cur_target = (0.2,0.05,0.02)
    # move the pen to the first location 
    # loop over each stroke 
    # from one stroke to another stroke lift the pen up 
    # print(isArrive())
    qdes = configSolver(cur_target)
    dqdes = calVel(joint_max_vel, q, qdes)
    ###

    # ### Test Print 
    # print(q)
    print(qdes)
    # print(dqdes)
    # print(cur_target)
    # print(q)
    # print(ee_local_position)
    # ###
    return getPIDTorqueAndAdvance(q,dq,
    qdes,dqdes,
    kP,kI,kD,
    pid_integrators,dt)
    
    #return [0,1,0,0]

def configSolver(cur_target):
    global robot
    robot.setConfig([0,0,0.5,0.01])
    end_link = robot.link(robot.numLinks()-1)
    ee_local_axis = end_link.getAxis()
    ee_local_position = (0,0,0)
    cur_target_axis = (0,0,-1)
    goal = ik.objective(end_link, 
    local=[ee_local_position, vectorops.add(ee_local_position, ee_local_axis)], # match end points 
    world=[cur_target, vectorops.add(cur_target, cur_target_axis)])
    ik.solve_global(goal)
    # ### Test Print
    # print('Local Coord: ', [ee_local_position, vectorops.add(ee_local_position, (0,0,-0.01))])
    # print('Global Coord: ', [cur_target, vectorops.add(cur_target, (0,0,0.01))])
    # ###
    return robot.getConfig()

def calVel(max_vel, current_val, target_val):
    out_len = len(current_val)
    out_vel = [0 for _ in range(out_len)]
    for idx in range(out_len):
        vel_ratio = abs(0.5 - 1/(1 + math.exp(current_val[idx] - target_val[idx])))
        cur_vel = vel_ratio * max_vel
        if cur_vel < 0.005: cur_vel = 0
        out_vel[idx] = cur_vel
    return out_vel
    
def isArrive(cur_q, target_q, error_limit):
    return vectorops.norm(vectorops.sub(cur_q, target_q)) < error_limit
        

def curves():
    # K = [[(0.2,0.05),(0.2,-0.05)],[(0.25,0.05),(0.2,0.0),(0.25,-0.05)]]
    # H = [[(0.28,0.05),(0.28,-0.05)],[(0.33,0.05),(0.33,-0.05)],[(0.28,0),(0.33,0)]]
    # initials = K + H    
    return [[(0.2,0.05),(0.2,-0.05)]]

#####################################################################
# Place your written answers here
# Ki term does not have to wait for a long time to eliminate the sse
#
#
#
#
#
#
#
#
