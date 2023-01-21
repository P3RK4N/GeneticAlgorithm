import os
import sys
import cv2
import numpy as np
import torch
from render import ij2xy, resolution, draw_body
import threading

'''
left click bone
right click muscle
double left click point
'''

DELAY = 100

points = np.zeros((0, 2))
bones = []
muscles = []
rod_colors = []
timers = [None]
img = np.zeros(resolution, dtype=np.uint8)
was_dbl = False
last_point = None

def distance(a, b):
    return np.sqrt((a**2 - b**2).sum())

# function that gets called when something with the mouse happens on the screen
def mouse_callback(event, x, y, flags, params):
    curr_point = ij2xy(np.array([[y, x]]))
    global was_dbl, last_point, timers, points
    # if left button is clicked we are not still sure is it a double or single click
    # make a timer for 0.5sec and if it gets executed than it was single
    # if it gets caceled by the double click event then it was not a single click
    if event == cv2.EVENT_LBUTTONDOWN:
        # if this is the first click find the point and save it
    	# alse add a bone between last saved point and closes one
        if was_dbl:
            was_dbl = False
        else:
            def lclck():
                print("L Mouse position: ({}, {})".format(x, y))
                global last_point
                if last_point is not None:
                    other_point = find_closest_point(curr_point)
                    print(other_point)
                    bones.append([last_point, other_point])
                    last_point = None
                else:
                    last_point = find_closest_point(curr_point)
            timer = threading.Timer(0.5, lclck)  # create a timer that will call the function after a 1 second delay
            timer.start()  # start the timer    
            timers[0] = timer

    if event == cv2.EVENT_RBUTTONDOWN:
    	# if this is the first click find the point and save it
    	# alse add a muscle between last saved point and closes one
        print("R Mouse position: ({}, {})".format(x, y))
        if last_point is not None:
            other_point = find_closest_point(curr_point)
            print(other_point, 'pther')
            muscles.append([last_point, other_point])
            last_point = None
        else:
            last_point = find_closest_point(curr_point)


    if event == cv2.EVENT_LBUTTONDBLCLK:
    	# create a point
    	# cancel the timer (left button click created this event)
        print("D Mouse position: ({}, {})".format(x, y))
        was_dbl = True
        timers[0].cancel()
        points = np.concatenate(
            (points, curr_point), axis=0
        )
        last_point = None

# find the closest point in eucledean distance
def find_closest_point(a):
    dists = np.sqrt(((a- points)**2).sum(axis=1))
    return np.argmin(dists)

cv2.namedWindow("hello")
cv2.setMouseCallback("hello", mouse_callback)

# the main loop
while True:
    img = draw_body(points, bones+muscles, [255]*len(bones) + [50]*len(muscles))
    cv2.imshow('hello', img)
    if cv2.waitKey(DELAY) == ord('q'):
        break

cv2.destroyAllWindows()

# make the specified dir if it does not exist already
dir_name = sys.argv[1]
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

# save everything
for var, svar in [(points, 'points'), (muscles, 'muscles'), (bones, 'bones')]:
    with open(os.path.join(dir_name, svar+'.txt'), 'w') as f:
        for x, y in var:
            f.write(f"{x} {y}\n")


