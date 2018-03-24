# To provide better result the error limit (under getTorque) are set to be very small and it takes minutes to finish 
# tracking all the initials.
# if you just want to see the state machine, increase those error_limits to the suggested values 
# Thank you.

import math
from klampt.math import vectorops
from klampt.model import ik

robot = None
dt = 0.01

#for I terms
pid_integrators = [0,0,0,0]

#higher level controller status... you don't need to use these if you wish
status = "move to above"
current_stroke = 0
current_stroke_progress = 1
current_frac = 0
stroke_list = []

def init(world):
    global robot,stroke_list
    robot = world.robot(0)
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
    torque_cap= 1.525
    for joint_idx in range(len(pid_integrators)):
        pid_integrators[joint_idx] += (q[joint_idx] - qdes[joint_idx])*dt
    torques = [0 for _ in range(len(q))]
    for i in range(len(torques)):
        #only the P term is computed here...
        cur_torque = (-kP[i]*(q[i] - qdes[i]) 
            - kD[i] * (dq[i] - dqdes[i]) 
            - kI[i] * pid_integrators[i])
        if i == 0 or i == 1:
            cur_torque = math.copysign(1, cur_torque) * min(torque_cap, abs(cur_torque))
        elif i == 3: # the pen 
            cur_torque -= robot.link(i).getMass().getMass()*9.8
        torques[i] = cur_torque
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
    global robot,status,current_stroke,current_stroke_progress,current_frac, stroke_list
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
    # print(q)
    # ### 
    
    ### Draw Strokes 
    kP = [25,6,10,9]
    kI = [10,3.5,5,2]
    kD = [15,2,0.4,2]
    joint_max_vel = 0.0
    small_error_limit = 10**-2.2 # To see fast process set to 10**-1.0
    large_error_limit = 10**-1.75 # To see fast process set to 10**-0.85
    lift_h = 0.01
    write_h = -0.0001
    num_frac = 50.0
    dqdes = [0,0,0,0]
    # State Machine 
    if status == 'lift up':
        cur_target = addDim(stroke_list[current_stroke][-1], lift_h)
        qdes = configSolver(cur_target)
        if isArrive(q, qdes, dq, dqdes, large_error_limit, large_error_limit):
            current_stroke += 1
            status = 'move to above'
    elif status == 'move to above':
        cur_target = addDim(stroke_list[current_stroke][0], lift_h)
        qdes = configSolver(cur_target)
        if isArrive(q, qdes, dq, dqdes, large_error_limit, large_error_limit):
            status = 'push down'
    elif status == 'push down':
        cur_target = addDim(stroke_list[current_stroke][0], write_h)
        qdes = configSolver(cur_target)
        if isArrive(q, qdes, dq, dqdes, small_error_limit, small_error_limit):
            status = 'writting stroke'
    elif status == 'writting stroke':
        start_point = addDim(stroke_list[current_stroke][current_stroke_progress - 1], write_h)
        end_point = addDim(stroke_list[current_stroke][current_stroke_progress], write_h)
        cur_target = vectorops.interpolate(start_point, end_point, 1/num_frac * current_frac)
        qdes = configSolver(cur_target)
        if isArrive(q, qdes, dq, dqdes, small_error_limit,small_error_limit):
            current_frac += 1 
            if current_frac > num_frac:
                current_frac = 0
                if current_stroke_progress < len(stroke_list[current_stroke])-1:
                    current_stroke_progress += 1
                else:
                    current_stroke_progress = 1
                    if current_stroke == len(stroke_list) - 1:
                        status = 'finish'
                        print('finish')
                    else:
                        status = 'lift up'
    elif status == 'finish':
        hold_point = addDim(stroke_list[current_stroke][-1], lift_h)
        qdes = configSolver(hold_point)
    ###
    
    ### Test Print 
    # print('Status: ' +  status + " ")
    # print('Current Stroke: ',current_stroke)
    # print('Stroke Progress: ', current_stroke_progress)
    # print('Current frac: ', current_frac)
    # print('Current pen tip ', q[3])
    # print('Target pen tip  ', dqdes[3])
    # print('Current pen vel ', dq)
    # print('Current target pen vel ', dqdes)
    # print('---------------------------')
    ###
    return getPIDTorqueAndAdvance(q,dq,
    qdes,dqdes,
    kP,kI,kD,
    pid_integrators,dt)
    
    #return [0,1,0,0]

def configSolver(cur_target):
    global robot
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
    
def isArrive(cur_q, target_q, cur_dq, target_dq, q_error_limit, dq_error_limit):
    config_len = len(cur_dq)
    for idx in range(config_len):
        cur_q_dist = abs(cur_q[idx] - target_q[idx])
        cur_dq_dist = abs(cur_dq[idx] - target_dq[idx])
        if cur_q_dist > q_error_limit or cur_dq_dist > dq_error_limit:
            return False
    return True

def addDim(in_coord, new_d_val):
    out_coord = list(in_coord)
    out_coord.append(new_d_val)
    return out_coord


def curves():
    Z_1 = [[(0.2,0.05),(0.25,0.05),(0.2,-0.05),(0.25,-0.05)]]
    Z_2 = [[(0.27,0.05),(0.32,0.05),(0.27,-0.05),(0.32,-0.05)]]
    return Z_1 + Z_2

#####################################################################
# Place your written answers here
# Question 1: Why adding a compensator term helps?
# Adding the compensator term in the controller will get rid off the known offset created by the gravity and balance
# t = mg, so that it will not require the Ki term to eliminate this known offset and move the pen tip to the desired location
# To tune the gains the following code is used to plot the configuration:
# print(q) on the website 
# copy paste the result into a DataFile.txt in the same folder as the following python file
# 
# import numpy as np
# import matplotlib.pyplot as plt
# 
# 
# DATAFILE =  open('DataFile.txt','r')
# config_record = []
# for each_line in DATAFILE:
#     if "[" in each_line:
#         if ">" in each_line:
#             each_line = each_line[3:-2].split(",")
#         else:
#             print("illegal Config" + str(each_line))
#             each_line = each_line[1:-2].split(",")
#         config_record.append([float(each) for each in each_line])
# config_record = np.array(config_record)
# 
# for joint_idx in range(config_record.shape[1]):
#     plt.figure(figsize=(10,5))
#     plt.title("joint idx: " + str(joint_idx))
#     plt.plot(config_record[:,joint_idx])
# After the change of the configuration are ploted, the following rules are applied.
# To reduce steady state error, increase Ki, but settling time will increase 
# To reduce settling time, increase Kp or Kd, but might make the system overdamped 
# 
# Question 2: Description of the method 
#   To make the pen track through the given letters, the actions are seperate into four difference states, lift up, 
# move to above, push down and writting strok. At the beginning the pen is set to lift up state and then the pen is moved
# to the location above the starting point of the first stroke. The pen is then pushed down to the starting point of the 
# first stroke. Then each stroke is interpolated into many points and each point is set to be a target that the pen 
# is going to move to. Only when the pen reaches the current target point(within error_limit), will the target point be updated.
# After one stroke is finished, the pen is lifted up again to move to the above of the next stroke. 
# When there is no stroke left the state is set to finish and the pen is lifted to stay above the end point of the final stroke.
#   To obtain the configeration for each target point as the pen tracks, IK solver is used. To set the goal for the IK solver, the 
# target point and target point + [0,0,-1] are matched with the the [0,0,0] and [0,0,1] in the end link local coordinates. 
