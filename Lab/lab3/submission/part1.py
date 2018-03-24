from klampt import *
from klampt.math import vectorops

#attractive force constant
attractiveConstant = 100
#repulsive distance
repulsiveDistance = 0.1
#time step to limit the distance traveled
timeStep = 0.01

class Circle:
    def __init__(self,x=0,y=0,radius=1):
        self.center = (x,y)
        self.radius = radius
        
    def contains(self,point):
        return (vectorops.distance(point,self.center) <= self.radius)
    def distance(self,point):
        return (vectorops.distance(point,self.center) - self.radius)
    def getCenter(self):
        return self.center
    def getRadius(self):
        return self.radius

def force(q,target,obstacles):
    """Returns the potential field force for a robot at configuration q,
    trying to reach the given target, with the specified obstacles.

    Input:
    - q: a 2D point giving the robot's configuration
    - robotRadius: the radius of the robot
    - target: a 2D point giving the robot's target
    - obstacles: a list of Circle's giving the obstacles
    """
    #basic target-tracking potential field implemented here
    #TODO: implement your own potential field
    att_force = vectorops.mul(vectorops.sub(target,q),attractiveConstant)
    total_rep_force = [0,0]
    for each_ob in obstacles:
        cur_rep_force = repForce(q, each_ob, repulsiveDistance)
        total_rep_force = vectorops.add(total_rep_force, cur_rep_force)
    total_force = vectorops.add(att_force, total_rep_force)
        #limit the norm of f
    if vectorops.norm(total_force) > 1:
        total_force = vectorops.unit(
            vectorops.add(total_force, (1e-10, 1e-10)))
    return total_force

def repForce(point_loc, in_obstacle, max_detect_dist):
    '''
    return the replusive force 
    according to the distance between the point and the obstacle
    if it is too far away just return 0:
    point_loc: (x,y)
    in_obstacle: circle object 
    '''
    obs_center = in_obstacle.getCenter()
    dist = in_obstacle.distance(point_loc)
    # if dist <= 0: 
    #     return(0,0)
    #     #assert("Check current point location, overlaps the obstacle")
    if dist > max_detect_dist:
        return (0,0)
    else:
        # direction to the obstacle center 
        dir_center = vectorops.sub(point_loc, obs_center)
        # normalize vector 
        dir_center = vectorops.unit(dir_center)
        # to avoid local min add a small tangential vector 
        # dir_around = (dir_center[1],dir_center[0])
        return vectorops.mul(dir_center, 1/abs(dist))

def start():
    return (-1,0.5)

def target():
    return (1,0.5)

def obstacles():
    return [Circle(0.3,0.25,0.2),
        Circle(0.1,0.3,0.2),
        Circle(-0.4,0.6,0.2)]
    
    ''' LOCAL MIN CASE
    [Circle(0.5,0.25,0.2),
        Circle(0.5,0.75,0.2),
        Circle(0.1,0.5,0.2),
        Circle(-0.4,0.5,0.2)]
    '''



