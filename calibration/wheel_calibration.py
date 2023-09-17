# for computing the wheel calibration parameters
import numpy as np
import os
import sys
sys.path.insert(0, "../util")
from pibot import PenguinPi

def calibrateWheelRadius(): 
    # Compute the robot scale parameter using a range of wheel velocities.
    # For each wheel velocity, the robot scale parameter can be computed
    # by comparing the time and distance driven to the input wheel velocities.

    '''
        The radius of the penguinpi wheel can be related to the linear velocity of the 
        robot by v = rw. 

        The calibrateWheelRadius() function will involve the robot driving a distance of 1 
        m for each defined wheel velocity. Here we assume that wheel velocity is "w". The 
        time it takes for the robot to travel the 1m will be used to determine the constant 
        linear velocity of the robot. The radius ("scale") of the robot can then be determined
        by averaging these results.

        From this, we arrive at: 
             d/t (m/s) = v = rw (ticks/s) 
             r = d/(t*w) (m/tick)
             scale = mean(r)
    '''

    ##########################################
    # Feel free to change the range / step
    ##########################################
    wheel_velocities_range = range(20, 80, 15)
    delta_times = []

    for wheel_vel in wheel_velocities_range:
        print("Driving at {} ticks/s.".format(wheel_vel))
        # Repeat the test until the correct time is found.
        while True:
            delta_time = input("Input the time to drive in seconds: ")
            try:
                delta_time = float(delta_time)
            except ValueError:
                print("Time must be a number.")
                continue

            # Drive the robot at the given speed for the given time
            ppi.set_velocity([1, 0], tick=wheel_vel, time=delta_time)

            uInput = input("Did the robot travel 1m?[y/N]")
            if uInput == 'y':
                delta_times.append(delta_time)
                print("Recording that the robot drove 1m in {:.2f} seconds at wheel speed {}.\n".format(delta_time,
                                                                                                        wheel_vel))
                break

    # Once finished driving, compute the scale parameter by averaging
    num = len(wheel_velocities_range)
    scale = 0
    for delta_time, wheel_vel in zip(delta_times, wheel_velocities_range):
        scale += 1/(delta_time*wheel_vel)
    scale = scale/num

    print("The scale parameter is estimated as {:.6f} m/ticks.".format(scale))

    return scale


def calibrateBaseline(scale): # theta/t = w = r/l*(wl-wr) but wr = 0 so l = r/(2pi/t)*wr
    # Compute the robot basline parameter using a range of wheel velocities.
    # For each wheel velocity, the robot baseline parameter can be computed by
    # comparing the time elapsed and rotation completed to the input wheel
    # velocities to find out the distance between the wheels.

    '''
        The length of the penguinpi base can be related to the angular velocity of the 
        robot by w_robot = r/L * (wr - wl)

        The calibrateBaseline() function will involve the robot rotating a total of 360 degrees 
        for each defined wheel velocity. Here we assume that wheel velocity is the angular velocity
        for both wheels (!). The time it takes for the robot to rotate 360 degrees will be used to 
        determine the constant angular velocity of the robot. The length ("baseline") of the robot can 
        then be determined by averaging these results.

        From this, we arrive at: 
             theta/t (rad/s) = r/L * (wr - wl)
                 - theta = 2pi
                 - wr - wl = 2*wheel_vel (for the robot to pivot along its central axis, wl = -wr or vice versa)
             L = wrt/pi
             baseline = mean(L)
    '''


    ##########################################
    # Feel free to change the range / step
    ##########################################
    wheel_velocities_range = range(30, 60, 10)
    delta_times = []

    for wheel_vel in wheel_velocities_range:
        print("Driving at {} ticks/s.".format(wheel_vel))
        # Repeat the test until the correct time is found.
        while True:
            delta_time = input("Input the time to drive in seconds: ")
            try:
                delta_time = float(delta_time)
            except ValueError:
                print("Time must be a number.")
                continue

            # Spin the robot at the given speed for the given time
            ppi.set_velocity([0, 1], tick=20,turning_tick=wheel_vel, time = delta_time)

            uInput = input("Did the robot spin 360deg?[y/N]")
            if uInput == 'y':
                delta_times.append(delta_time)
                print("Recording that the robot spun 360deg in {:.2f} seconds at wheel speed {}.\n".format(delta_time,
                                                                                                           wheel_vel))
                break

    # Once finished driving, compute the basline parameter by averaging
    # l = r/(2pi/t)*wr
    num = len(wheel_velocities_range)
    baseline = 0
    for delta_time, wheel_vel in zip(delta_times, wheel_velocities_range):
        #baseline += scale*wheel_vel*delta_time/(2*np.pi)
        baseline += (scale*delta_time*wheel_vel)/(np.pi)
        # pass # TODO: replace with your code to compute the baseline parameter using scale, wheel_vel, and delta_time

    baseline = baseline/num
    print("The baseline parameter is estimated as {:.6f} m.".format(baseline))

    return baseline


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", metavar='', type=str, default='localhost')
    parser.add_argument("--port", metavar='', type=int, default=40000)
    args, _ = parser.parse_known_args()

    ppi = PenguinPi(args.ip,args.port)

    # calibrate pibot scale and baseline
    dataDir = "{}/param/".format(os.getcwd())

    print('Calibrating PiBot scale...\n')
    scale = calibrateWheelRadius()
    fileNameS = "{}scale.txt".format(dataDir)
    np.savetxt(fileNameS, np.array([scale]), delimiter=',')

    print('Calibrating PiBot baseline...\n')
    baseline = calibrateBaseline(scale)
    fileNameB = "{}baseline.txt".format(dataDir)
    np.savetxt(fileNameB, np.array([baseline]), delimiter=',')

    print('Finished calibration')
