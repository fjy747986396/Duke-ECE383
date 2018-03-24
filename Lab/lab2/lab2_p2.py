from klampt import *
from klampt.math import vectorops,so3,se3
import math

def lab2b(l1,l2,l3,point):
    """
    Compute all IK solutions of a 3R manipulator.  The joint axes in the reference configuration
    are ZYY, with each link's origin displaced by a given amount on the X axis.
    In:
    - L1, L2, L3: the link lengths
    - point: the target position (x,y,z)
    Out:
    - A pair (n,solutionList) where n is the number of solutions (either 0,
      1, 2, 4, or float('inf')) and solutionList is a list of all solutions.
      In the n=0 case, it should be an empty list [], and in the n=inf case
      it should give one example solution.

      Each solution is a 3-tuple giving the joint angles, in radians.
    """
    a1_1 = math.atan2(point[1], point[0])
    a1_2 = a1_1 + math.pi
    a1 = [a1_1,a1_2]
    solution_num = 0
    solution_list = []
    # starting from a1_1
    for idx in range(len(a1)):
        each_a1 = a1[idx]
        j1 = (l1 * math.cos(each_a1), l1 * math.sin(each_a1), 0)
        dist = vectorops.distance(point, j1) # from j1 to target 
        if dist > (l2 + l3):
            pass # no solution
        elif dist < abs(l2 - l3):
            pass # no solution
        elif dist == abs(l2 - l3) and l2 != l3:
            solution_num += 1
            solution_list.append((each_a1, math.pi,0))
        elif dist == (l2 + l3):
            solution_num += 1
            solution_list.append((each_a1, math.pi,0))
        elif dist == 0: # point at the joint 2
            return (float('inf'), [(each_a1, math.pi,0)])
        else:
            cos_angle = (dist**2.0 - l2 ** 2.0 - l3 ** 2.0)/(-2.0 * l2 * l3)
            a3_1 = (math.pi - math.acos(cos_angle))
            a3_2 = (-a3_1)
            a3 = [a3_1, a3_2]
            for each_a3 in a3:
                h = l3 * math.sin(each_a3)
                ratio = h/dist
                delta_a = math.asin(ratio)
                total_a = math.asin(point[2]/dist)
                if idx == 0:
                    a2 = -(delta_a + total_a)
                if idx == 1:
                    a2 = math.pi - delta_a + total_a
                solution_num += 1
                solution_list.append((each_a1, a2, each_a3))
    return (solution_num, solution_list)
            
    
def ik_goal_motion(t):
    """Returns a point describing where the goal should be at time t"""
    return (math.sin(t)*1.5+0.3,1.0*math.cos(t/2+0.5), abs((t % 3)*0.2-0.5 ) )

def removeError(s_value):
    return abs(s_value)/s_value * min(1, abs(s_value))
