from klampt import *
from klampt.math import vectorops,so2,so3
from klampt.plan.cspace import CSpace,MotionPlan
import math

#turn this to False if redrawing becomes too slow
draw_roadmap = False
#how much planning time to take every frame, in seconds
max_plan_time = 0.5
#how many iterations to take per frame
max_plan_iters = 100

class SE2ObstacleCSpace(CSpace):
    """The configuration space being used in Lab 3D.
    Consists of a RobotModel robot and TerrainModel obstacles.

    Note: compared to Lab 3B, the translation domain is larger!  This requires
    some of the parameters (collision checking resolution, for example) to be
    set higher as well.
    """
    def __init__(self,world):
        CSpace.__init__(self)
        #set bounds
        self.bound = [(-5,5),(-5,5),(0,math.pi*2)]
        #set collision checking resolution
        self.eps = 1e-1
        #get the robot (a RobotModel instance)
        self.robot = world.robot(0)
        #get obstacles here, these will be TerrainModel instances
        self.obstacles = [world.terrain(i) for i in xrange(world.numTerrains())]
    
    def feasible(self,q):
        """TODO: Implement this feasibility test.  It is used by the motion planner to
        determine whether the robot at configuration q is feasible."""
        #modified bounds test: we don't care about angle
        if not CSpace.feasible(self,(q[0],q[1],0)): return False
        self.robot.setConfig(q)
        base = self.robot.link(2)
        for o in self.obstacles:
            if o.geometry().collides(base.geometry()): return False
        return True

    def interpolate(self,a,b,u):
        """TODO: Implement this interpolation function.
        Inputs:
        - a: the start configuration
        - b: the end configuration
        - u: the interpolation parameter in the range from 0 to 1 
        Out: the interpolated configuration
        """
        interpolate_result = vectorops.interpolate(a[0:2], b[0:2], u)
        interpolate_angle = interpolateAngle(a[2], b[2], u)
        interpolate_result.append(interpolate_angle)
        return interpolate_result

    def distance(self,a,b):
        """TODO: Implement this interpolation function.
        Inputs:
        - a: the start configuration
        - b: the end configuration
        """
        sq_x = (a[0] - b[0])**2.0
        sq_y = (a[1] - b[1])**2.0
        sq_angle =(min([abs(a[2] - b[2]),
                    2*math.pi - abs(a[2] - b[2])])) ** 2.0 
        return math.sqrt(sq_x + sq_y + sq_angle)
# helper functions 
def interpolateAngle(a_1, a_2, u):
	'''
	interpoalte angle through the shortest distance direction 
	normalize the result angle to 0 - 2pi range 
	'''
	if a_1 < (a_2 + math.pi) and a_1 > a_2:
		out_angle = a_1 - (a_1 - a_2) * u
	elif a_2 > (a_1 + math.pi) and a_2 > a_1:
		out_angle = a_1 - (2 * math.pi - a_2 + a_1) * u
	else:
		out_angle = a_1 + (a_2 - a_1) * u
	return normalizeAngle(out_angle)

def normalizeAngle(in_a):
	'''
	normalize angle to 0 - 2 pi range 
	'''
	while in_a < 0:
		in_a += 2 * math.pi
	return in_a


def makePlanner(space, start, goal):
    """Creates a MotionPlan object for the given space, start, and goal.
    Returns (planner,optimizing) where optimizing is True if the planner should
    continue be run after the first solution path has been found"""
    #This sets a Probabilistic Road Map (PRM) planner that connects
    #a random point to its 10 nearest neighbors. If knn is set to 0,
    #the points are connected as long as they lie
    #within distance 0.1 of one another
    # MotionPlan.setOptions(type="prm",knn=10,connectionThreshold=1)
    #This line sets a Rapidly-exploring Random Tree (RRT) planner that
    #repeatedly extends the tree toward a random point at maximum
    #distance 0.25.  It uses the bidirectional=True option, which grows
    #trees from both the start and the goal
    #MotionPlan.setOptions(type="rrt",connectionThreshold=2.0,perturbationRadius=2.5,bidirectional=True)
    #MotionPlan.setOptions(type="sbl",connectionThreshold=5.0,gridResolution=1.0,perturbationRadius=1.5,bidirectional=True)
    optimizing = False

    #Optimizing planners.  Make sure to uncomment optimizing = True below.
    #This sets the PRM algorithm with shortcutting
    # MotionPlan.setOptions(type="prm",knn=10,connectionThreshold=1.0,shortcut=True)
    #This sets the RRT* algorithm
    MotionPlan.setOptions(type="rrt*",connectionThreshold=2.0,perturbationRadius=2.5)
    #This sets a fast-marching method algorithm (Note: this does not work properly with rotations)
    #MotionPlan.setOptions(type="fmm*")
    #This sets a random-restart + shortcutting RRT
    # MotionPlan.setOptions(type="rrt",connectionThreshold=2.0,perturbationRadius=2.5,bidirectional=True,restart=True,shortcut=True,restartTermCond="{foundSolution:1,maxIters:1000}")
    optimizing = True

    #create the planner, and return it along with the termination criterion
    planner = MotionPlan(space)
    return planner,optimizing

def start():
    return (-3.5,-3,math.pi/4)

def target():
    return (3.5,3,math.pi*7/4)

