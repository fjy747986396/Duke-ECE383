#Lab 1b stub code
import math

def lab1b(point,angle):
    #TODO: Return the coordinates of the point after
    #rotating by the given angle about the origin.
    #In:
    # - point: a tuple (px,py)
    # - angle: an angle, in degrees
    #Out:
    # - a tuple (qx,qy) indicating the rotated point
    c_theta = math.cos(angle*math.pi/180)
    s_theta = math.sin(angle*math.pi/180)
    new_x = point[0] * c_theta - point[1] * s_theta
    new_y = point[0] * s_theta + point[1] * c_theta
    point = (new_x, new_y)
    return point


def fuzzy_eq(a,b,eps=1e-8):
    """Returns true if a is within +/- eps of b."""
    return abs(a-b)<=eps

def fuzzy_veq(a,b,eps=1e-8):
    return all(fuzzy_eq(ai,bi,eps) for ai,bi in zip(a,b)) 

def selfTest():
    """You may use this function to make sure your values are correct"""
    assert fuzzy_veq(lab1b((1,0),0),(1,0))
    assert fuzzy_veq(lab1b((1,10),0),(1,10))
    assert fuzzy_veq(lab1b((10,0),90),(0,10))
    assert fuzzy_veq(lab1b((10,5),180),(-10,-5))
    assert fuzzy_veq(lab1b((10,5),270),(5,-10))
    return

#uncomment this line to run the self test
selfTest()