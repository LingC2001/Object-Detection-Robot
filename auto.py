# teleoperate the robot, perform SLAM and object detection

# basic python packages
import numpy as np
import cv2 
import os, sys
import time
import json
from copy import deepcopy
import math
from auto_RRTC import *
from obstacle import *
import matplotlib.pyplot as plt
import TargetPoseEst


# import utility functions
sys.path.insert(0, "{}/utility".format(os.getcwd()))
from util.pibot import PenguinPi # access the robot
import util.DatasetHandler as dh # save/load functions
import util.measure as measure # measurements
import pygame # python package for GUI
import shutil # python package for file operations

# import SLAM components you developed in M2
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

# import CV components
sys.path.insert(0,"{}/network/".format(os.getcwd()))
sys.path.insert(0,"{}/network/scripts".format(os.getcwd()))
from network.scripts.detector_yolo import Detector

#! ################ Calista
def check_for_target(robot_pose):
    '''
       parser.add_argument("--runtype", type=str, default='sim')
    '''
    operate.command['inference'] = True
    operate.detect_target()
    #print('detector output', operate.detector_output)
    bboxs = operate.detector_output[:,0:4]
    labels = operate.detector_output[:,5]
    confs = operate.detector_output[:,4]
    out = {
                "pose" : robot_pose,
                "bbox" : bboxs.tolist(),
                "confs" : confs.tolist(),
                "labels" : labels.tolist()
            }
    #image_poses = out
    completed_img_dict = TargetPoseEst.get_image_info(out)
    #image_poses.append(out)
    #print('image dict', completed_img_dict)
    
    target_map, depth = TargetPoseEst.estimate_pose(operate.folder, operate.camera_matrix, completed_img_dict, args.runtype)
    print('target map', target_map)
    print('depth', depth)
    #target_est = TargetPoseEst.merge_estimations(target_map)
    #print('Target est', target_est)
    operate.command['inference'] = False

    return target_map, depth

#!########################################

class Operate:
    def __init__(self, args):
        self.level = 0
        self.camera_matrix = None
        self.scale = None
        self.baseline = None
        self.dist_coeffs = None
        self.folder = 'pibot_dataset/'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        else:
            shutil.rmtree(self.folder)
            os.makedirs(self.folder)
        
        # initialise data parameters
        if args.play_data:
            self.pibot = dh.DatasetPlayer("record")
        else:
            self.pibot = PenguinPi(args.ip, args.port)

        # initialise SLAM parameters
        self.ekf = self.init_ekf(args.calib_dir, args.ip)
        self.aruco_det = aruco.aruco_detector(
            self.ekf.robot, marker_length = 0.07) # size of the ARUCO markers

        if args.save_data:
            self.data = dh.DatasetWriter('record')
        else:
            self.data = None
        self.output = dh.OutputWriter('lab_output')
        self.command = {'motion':[0, 0], 
                        'inference': False,
                        'output': False,
                        'save_inference': False,
                        'save_image': False}
        self.quit = False
        self.pred_fname = ''
        self.request_recover_robot = False
        self.file_output = None
        self.ekf_on = False
        self.double_reset_comfirm = 0
        self.image_id = 0
        self.notification = 'Press ENTER to start SLAM'
        # a 5min timer
        self.count_down = 3000
        self.start_time = time.time()
        self.control_clock = time.time()
        # initialise images
        self.img = np.zeros([240,320,3], dtype=np.uint8)
        self.aruco_img = np.zeros([240,320,3], dtype=np.uint8)
        self.detector_output = np.zeros([240,320], dtype=np.uint8)
        
        if args.ckpt == "":
            self.detector = None
            self.network_vis = cv2.imread('pics/8bit/detector_splash.png')
        else:
            # y = input("Use yolo? (y/n): ")
            y = "y"
            if y == "y":
                self.detector = Detector(use_gpu=False)
                self.detector.load_weights(sim=False)
                self.network_vis = np.ones((240, 320,3))* 100
                with open('lab_output/output.txt', 'w') as f:
                        f.write("")
                print("Emptied lab_output/output.txt file")
            else:
                self.detector = None
                self.network_vis = cv2.imread('pics/8bit/detector_splash.png')

        self.bg = pygame.image.load('pics/gui_mask.jpg')
        self.last_keys_pressed = [False, False, False, False, False]
        self.vis_id = 0
        self.fruit_search = False
        self.fruit_lists = None
        self.fruits_true_pos = None
        self.aruco_true_pos = None

    # wheel control
    def control(self, motion, drive_time):
        if not self.fruit_search:       
            if args.play_data:
                lv, rv = self.pibot.set_velocity()            
            else:
                lv, rv = self.pibot.set_velocity(
                    self.command['motion'])
            if not self.data is None:
                self.data.write_keyboard(lv, rv)
            dt = time.time() - self.control_clock
            drive_meas = measure.Drive(lv, rv, dt)
            self.control_clock = time.time()
            return drive_meas
        else:
            lv, rv = self.pibot.set_velocity(motion, time = drive_time)
            drive_meas = measure.Drive(lv, rv, drive_time)
            self.control_clock = time.time()
            return drive_meas
    # camera control
    def take_pic(self):
        self.img = self.pibot.get_image()
        if not self.data is None:
            self.data.write_image(self.img)

    # SLAM with ARUCO markers       
    def update_slam(self, drive_meas):
        lms, self.aruco_img = self.aruco_det.detect_marker_positions(self.img)
        # cv2.imshow("img", self.aruco_img)
        # cv2.waitKey(0)
        if self.request_recover_robot:
            is_success = self.ekf.recover_from_pause(lms)
            if is_success:
                self.notification = 'Robot pose is successfuly recovered'
                self.ekf_on = True
            else:
                self.notification = 'Recover failed, need >2 landmarks!'
                self.ekf_on = False
            self.request_recover_robot = False
        elif self.ekf_on: # and not self.debug_flag:
            self.ekf.predict(drive_meas)
            self.ekf.add_landmarks(lms)
            self.ekf.update(lms)

    # using computer vision to detect targets
    def detect_target(self):
        if self.command['inference'] and self.detector is not None:
            image = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)
            output, output_img = self.detector.detect_single_image(image)
            self.detector_output = output[0].numpy()
            print(self.detector_output)
            self.network_vis = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
            self.command['inference'] = False
            self.file_output = (self.detector_output, deepcopy(self.ekf.robot.state[:,0].tolist()))
            self.notification = f'{self.detector_output.shape[0]} target(s) detected'

    # save raw images taken by the camera
    def save_image(self):
        f_ = os.path.join(self.folder, f'img_{self.image_id}.png')
        if self.command['save_image']:
            image = self.pibot.get_image()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f_, image)
            self.image_id += 1
            self.command['save_image'] = False
            self.notification = f'{f_} is saved'

    # wheel and camera calibration for SLAM
    def init_ekf(self, datadir, ip):
        fileK = "{}intrinsic.txt".format(datadir)
        self.camera_matrix = np.loadtxt(fileK, delimiter=',')
        fileD = "{}distCoeffs.txt".format(datadir)
        self.dist_coeffs = np.loadtxt(fileD, delimiter=',')
        fileS = "{}scale.txt".format(datadir)
        self.scale = np.loadtxt(fileS, delimiter=',')
        # if ip == 'localhost':
        #     self.scale /= 2
        fileB = "{}baseline.txt".format(datadir)  
        self.baseline = np.loadtxt(fileB, delimiter=',')
        # self.baseline = 0.12
        robot = Robot(self.baseline, self.scale, self.camera_matrix, self.dist_coeffs)
        return EKF(robot)

    # save SLAM map
    def record_data(self):
        if self.command['output']:
            self.output.write_map(self.ekf)
            self.notification = 'Map is saved'
            self.command['output'] = False
        # save inference with the matching robot pose and detector labels
        if self.command['save_inference']:
            if self.file_output is not None:
                #image = cv2.cvtColor(self.file_output[0], cv2.COLOR_RGB2BGR)

                bboxs = self.file_output[0][:,0:4]
                labels = self.file_output[0][:,5]
                confs = self.file_output[0][:,4]

                out = {
                    "pose" : self.file_output[1],
                    "bbox" : bboxs.tolist(),
                    "confs" : confs.tolist(),
                    "labels" : labels.tolist()
                }
                cv2.imwrite('lab_output/pred_img'+str(self.vis_id)+'.png', cv2.cvtColor(self.network_vis, cv2.COLOR_RGB2BGR))
                self.vis_id += 1
                self.notification = f'Prediction is saved to lab_output/output.txt'
                with open('lab_output/output.txt', 'a') as f:
                    f.write(json.dumps(out))
                    f.write('\n')


            else:
                self.notification = f'No prediction in buffer, save ignored'
            self.command['save_inference'] = False

    # paint the GUI            
    def draw(self, canvas):
        canvas.blit(self.bg, (0, 0))
        text_colour = (220, 220, 220)
        v_pad = 40
        h_pad = 20

        # paint SLAM outputs
        ekf_view = self.ekf.draw_slam_state(res=(320, 480+v_pad),
            not_pause = self.ekf_on)
        canvas.blit(ekf_view, (2*h_pad+320, v_pad))
        robot_view = cv2.resize(self.aruco_img, (320, 240))
        self.draw_pygame_window(canvas, robot_view, 
                                position=(h_pad, v_pad)
                                )

        # for target detector (M3)
        detector_view = cv2.resize(self.network_vis,
                                   (320, 240), cv2.INTER_NEAREST)
        self.draw_pygame_window(canvas, detector_view, 
                                position=(h_pad, 240+2*v_pad)
                                )

        # canvas.blit(self.gui_mask, (0, 0))
        self.put_caption(canvas, caption='SLAM', position=(2*h_pad+320, v_pad))
        self.put_caption(canvas, caption='Detector',
                         position=(h_pad, 240+2*v_pad))
        self.put_caption(canvas, caption='PiBot Cam', position=(h_pad, v_pad))

        notifiation = TEXT_FONT.render(self.notification,
                                          False, text_colour)
        canvas.blit(notifiation, (h_pad+10, 596))

        time_remain = self.count_down - time.time() + self.start_time
        if time_remain > 0:
            time_remain = f'Count Down: {time_remain:03.0f}s'
        elif int(time_remain)%2 == 0:
            time_remain = "Time Is Up !!!"
        else:
            time_remain = ""
        count_down_surface = TEXT_FONT.render(time_remain, False, (50, 50, 50))
        canvas.blit(count_down_surface, (2*h_pad+320+5, 530))
        return canvas

    @staticmethod
    def draw_pygame_window(canvas, cv2_img, position):
        cv2_img = np.rot90(cv2_img)
        view = pygame.surfarray.make_surface(cv2_img)
        view = pygame.transform.flip(view, True, False)
        canvas.blit(view, position)
    
    @staticmethod
    def put_caption(canvas, caption, position, text_colour=(200, 200, 200)):
        caption_surface = TITLE_FONT.render(caption,
                                          False, text_colour)
        canvas.blit(caption_surface, (position[0], position[1]-25))

    def scale_speed(self):
        keys_pressed = pygame.key.get_pressed()
        shift_pressed = keys_pressed[pygame.K_LSHIFT] or keys_pressed[pygame.K_RSHIFT]
        if shift_pressed == True:
            speedscale = 3
        else:
            speedscale = 1
        return speedscale 

    # keyboard teleoperation        
    def update_keyboard(self):
        for event in pygame.event.get():

            if not self.fruit_search:
    ########### replace with your M1 codes ###########
                keys = pygame.key.get_pressed()
                up = keys[pygame.K_UP]
                down = keys[pygame.K_DOWN]
                left = keys[pygame.K_LEFT]
                right = keys[pygame.K_RIGHT]
                shift = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
                
            ############### add your codes below ###############
                v = 1
                keys_pressed = [up, down, left, right, shift]
                if keys_pressed != self.last_keys_pressed:
                    if up:
                        if (not left) and (not right): # up only
                            self.command['motion'] = [self.scale_speed()*v,0]
                            print("Moving Forward")
                        elif left and (not right): # up left
                            self.command['motion'] = [self.scale_speed()*v,self.scale_speed()*v]
                            print("Moving Forward-Left")
                        elif (not left) and right: # up right
                            self.command['motion'] = [self.scale_speed()*v,-self.scale_speed()*v]
                            print("Moving Forward-Right")
                        else:
                            self.command['motion'] = [0, 0]
                            print("Moving Forward")
                    elif down:
                        if (not left) and (not right): # down only
                            self.command['motion'] = [-self.scale_speed()*v,0]
                            print("Moving Backward")
                        elif left and (not right): # down left
                            self.command['motion'] = [-self.scale_speed()*v,-self.scale_speed()*v]
                            print("Moving Backward-Left")
                        elif (not left) and right: # down right
                            self.command['motion'] = [-self.scale_speed()*v,self.scale_speed()*v]
                            print("Moving Backword-Right")
                        else:
                            self.command['motion'] = [0, 0]
                            print("Moving Backward")
                    elif left: 
                        if not right: # left only
                            self.command['motion'] = [0,self.scale_speed()*v]
                            print("Spinning Left")
                        else:
                            self.command['motion'] = [0, 0]
                            print("Stopping")
                    elif right: # right only
                        self.command['motion'] = [0,-self.scale_speed()*v]
                        print("Spinning Right")
                    else:
                        self.command['motion'] = [0, 0]
                        print("Stopping")
                    
                    self.last_keys_pressed = [up, down, left, right, shift]


            ####################################################
            # stop
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.command['motion'] = [0, 0]
            # save image
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_i:
                self.command['save_image'] = True
            # save SLAM map
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                self.command['output'] = True
            # reset SLAM map
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                if self.double_reset_comfirm == 0:
                    self.notification = 'Press again to confirm CLEAR MAP'
                    self.double_reset_comfirm +=1
                elif self.double_reset_comfirm == 1:
                    self.notification = 'SLAM Map is cleared'
                    self.double_reset_comfirm = 0
                    self.ekf.reset()
            # run SLAM
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                n_observed_markers = len(self.ekf.taglist)
                if n_observed_markers == 0:
                    if not self.ekf_on:
                        self.notification = 'SLAM is running'
                        self.ekf_on = True

                #     else:
                #         self.notification = '> 2 landmarks is required for pausing'
                # elif n_observed_markers < 3:
                #     self.notification = '> 2 landmarks is required for pausing'
                # else:
                #     if not self.ekf_on:
                #         self.request_recover_robot = True
                #     self.ekf_on = not self.ekf_on
                #     if self.ekf_on:
                #         self.notification = 'SLAM is running'
                #     else:
                #         self.notification = 'SLAM is paused'
            # run object detector
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                self.command['inference'] = True
            # save object detection outputs
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                self.command['save_inference'] = True
            # quit
            elif event.type == pygame.QUIT:
                self.quit = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.quit = True
        if self.quit:
            pygame.quit()
            sys.exit()


####################################################### FRUIT SEARCH SCRIPTS ##################################################

def read_true_map(fname):
    """Read the ground truth map and output the pose of the ArUco markers and 3 types of target fruit to search

    @param fname: filename of the map
    @return:
        1) list of target fruits, e.g. ['apple', 'pear', 'lemon']
        2) locations of the target fruits, [[x1, y1], ..... [xn, yn]]
        3) locations of ArUco markers in order, i.e. pos[9, :] = position of the aruco10_0 marker
    """
    with open(fname, 'r') as fd:
        gt_dict = json.load(fd)
        fruit_list = []
        fruit_true_pos = []
        aruco_true_pos = np.empty([10, 2])

        # remove unique id of targets of the same type
        for key in gt_dict:
            x = np.round(gt_dict[key]['x'], 1)
            y = np.round(gt_dict[key]['y'], 1)

            if key.startswith('aruco'):
                if key.startswith('aruco10'):
                    aruco_true_pos[9][0] = x
                    aruco_true_pos[9][1] = y
                else:
                    marker_id = int(key[5])
                    aruco_true_pos[marker_id-1][0] = x
                    aruco_true_pos[marker_id-1][1] = y
            else:
                fruit_list.append(key[:-2])
                if len(fruit_true_pos) == 0:
                    fruit_true_pos = np.array([[x, y]])
                else:
                    fruit_true_pos = np.append(fruit_true_pos, [[x, y]], axis=0)

        return fruit_list, fruit_true_pos, aruco_true_pos


def read_search_list():
    """Read the search order of the target fruits

    @return: search order of the target fruits
    """
    search_list = []
    with open('search_list.txt', 'r') as fd:
        fruits = fd.readlines()

        for fruit in fruits:
            search_list.append(fruit.strip())

    return search_list


def print_target_fruits_pos(search_list, fruit_list, fruit_true_pos):
    """Print out the target fruits' pos in the search order

    @param search_list: search order of the fruits
    @param fruit_list: list of target fruits
    @param fruit_true_pos: positions of the target fruits
    """

    print("Search order:")
    n_fruit = 1
    for fruit in search_list:
        for i in range(3):
            if fruit == fruit_list[i]:
                print('{}) {} at [{}, {}]'.format(n_fruit,
                                                  fruit,
                                                  np.round(fruit_true_pos[i][0], 1),
                                                  np.round(fruit_true_pos[i][1], 1)))
        n_fruit += 1
    

def get_distance_to_goal(waypoint, robot_pose):
    # calc distance to waypoint
    x_goal, y_goal = waypoint
    try:
        x, y, theta = robot_pose
    except:
        x, y = robot_pose

    x_diff = x_goal - x
    y_diff = y_goal - y
    distance_to_goal = np.hypot(x_diff, y_diff)
    return distance_to_goal

def get_desired_heading(waypoint, robot_pose):
    # calc desired heading to waypoint (we will clamp between -np.pi and np.pi)
    x_goal, y_goal = waypoint
    x, y, theta = robot_pose
    x_diff = x_goal - x
    y_diff = y_goal - y
    desired_heading = np.arctan2(y_diff,x_diff)
    if desired_heading < 0:
        desired_heading += 2*np.pi
    # desired_heading = (angle_to_goal+np.pi) % (2*np.pi) -np.pi
    return desired_heading

def call_umeyama(detected_lms, aruco_true_pos):
    if (len(detected_lms)) > 1:
        detected_lms_pos = np.zeros([len(detected_lms), 2])
        aruco_poses = np.zeros([len(detected_lms), 2])
        for i in range(len(detected_lms_pos)):
            #print(detected_lms_pos)
            detected_lms_pos[i] = [detected_lms[i].position[0],detected_lms[i].position[1]]
            tag = detected_lms[i].tag-1
            aruco_poses[i] = aruco_true_pos[tag]
        R, t = operate.ekf.umeyama(detected_lms_pos.T, aruco_poses.T)
        robot_state = operate.ekf.get_state_vector()
        x, y, theta = robot_state[0:3]
        new_robot_pose = R @ np.array([x,y]).reshape(2,1) + t
        robot_state[0:3] = [new_robot_pose[0], new_robot_pose[1], theta]
        print(robot_state[0:3])
        operate.ekf.set_state_vector(robot_state)

def drive_to_point(waypoint, robot_pose, aruco_true_pos):
    calc_new_path = False
    # imports camera / wheel calibration parameters 
    scale = operate.scale
    baseline = operate.baseline
    ####################################################
    # PARAMETERS
    wheel_vel = 20 # tick
    turning_vel = 5
    K_pv = 1
    K_pw = 1.5

    rot_threshold = 0.06 # might want to decrease for better acc
    dist_threshold = 0.04
    dt = 0.05
    
    ####################################################################
    # ROTATION TO FACE WAYPOINT
    # get desired heading
    desired_heading = get_desired_heading(waypoint, robot_pose)
    # print("Desired_heading:")
    # print(desired_heading)
    orientation_diff = desired_heading - robot_pose[2]

    while abs(orientation_diff) > rot_threshold:
        # Calculating control vel and drive time
        if orientation_diff > np.pi or orientation_diff < -np.pi:
            turn_amount = 2*np.pi - abs(orientation_diff)
        else:
            turn_amount = abs(orientation_diff)
        w_k = K_pw*abs(turn_amount)        
        if  0<=orientation_diff <= np.pi or orientation_diff <= -np.pi:
            w_k = int(np.ceil(w_k))
        else:
            w_k = -int(np.ceil(w_k))
        # print("w_k:")
        # print(w_k)
        motion = [0, w_k]
        turn_time = max(abs(baseline*np.pi/(scale*w_k*turning_vel)*((turn_amount)/(2*np.pi)))/5,0.01)
        turn_time = dt

        # ===================================================================================

        operate.update_keyboard()
        operate.take_pic()
        drive_meas = operate.control(motion, turn_time)
        operate.update_slam(drive_meas)
        operate.record_data()
        operate.save_image()
        operate.detect_target()
        # visualise
        operate.draw(canvas)
        pygame.display.update()
        # drive_meas = operate.control([0,0], 0.1)
        # time.sleep(0.1)

        operate.ekf.robot.state[2] = (operate.ekf.robot.state[2]+2*np.pi)%(2*np.pi)
        
        # ===================================================================================

        # robot_pose = get_robot_pose(aruco_true_pos)
        robot_pose = operate.ekf.get_state_vector()[0:3,0]
        # print("Robot_pose:")
        # print(robot_pose)
        #desired_heading = get_desired_heading(waypoint, robot_pose)

        desired_heading = get_desired_heading(waypoint, robot_pose)
        orientation_diff = desired_heading - robot_pose[2]
        # print("Orientation Difference")
        # print(orientation_diff)

        
    time.sleep(0.5)
    #####################################################################
    # LINEAR DRIVE TO WAYPOINT
    # get distance to waypoint
    distance_to_goal = get_distance_to_goal(waypoint, robot_pose)
    min_dist = 9e9
    while distance_to_goal > dist_threshold and (-np.pi/2<orientation_diff < np.pi/2):
    #for i in range(1):
        # calculate vel and drive time
        min_dist = distance_to_goal
        v_k = K_pv*distance_to_goal
        v_k = int(np.ceil(v_k))
        motion = [v_k, 0]
        motion = [3, 0]
        drive_time = distance_to_goal/(motion[0]*scale*wheel_vel)
        drive_time = dt
        # ======================================================================================

        operate.update_keyboard()
        operate.take_pic()
        drive_meas = operate.control(motion, drive_time)
        operate.update_slam(drive_meas)
        operate.record_data()
        operate.save_image()
        operate.detect_target()
        # visualise
        operate.draw(canvas)
        pygame.display.update()
        
        # ===================================================================================== 
        # robot_pose = get_robot_pose(aruco_true_pos)
        robot_pose = operate.ekf.get_state_vector()[0:3,0]#get_robot_pose(aruco_true_pos)
        # print("Robot_pose:")
        # print(robot_pose)
        desired_heading = get_desired_heading(waypoint, robot_pose)
        orientation_diff = desired_heading - robot_pose[2]
        distance_to_goal = get_distance_to_goal(waypoint, robot_pose)
        # print("Distance_to_goal:")
        # print(distance_to_goal)

        # aruco_arr = operate.ekf.get_state_vector()[3:,0]
        # for i in range(len(aruco_arr)//2):
        #     dist = np.sqrt((robot_pose[0]-aruco_arr[2*i])**2 + (robot_pose[0]-aruco_arr[2*i+1])**2)
        #     if dist < 0.1:
        #         motion = [-1, 0]
        #         drive_time = 0.1/(abs(motion[0])*scale*wheel_vel)
        #         operate.update_keyboard()
        #         operate.take_pic()
        #         drive_meas = operate.control(motion, drive_time)
        #         operate.update_slam(drive_meas)
        #         operate.record_data()
        #         operate.save_image()
        #         operate.detect_target()
        #         # visualise
        #         operate.draw(canvas)
        #         pygame.display.update()
        #         print("Too close to obstacle, Moving backwards")
        #         calc_new_path = True
        
        if calc_new_path:
            break
    if calc_new_path:
        return robot_pose

    print("Arrived at [{}, {}]".format(robot_pose[0], robot_pose[1]))

    return robot_pose

def spin_360(robot_pose):

    # First 180 deg turn
    desired_heading = (robot_pose[2] + np.pi )%(2*np.pi)
    orientation_diff = desired_heading - robot_pose[2]
    rot_threshold = 0.1
    dt = 0.05
    while abs(orientation_diff) > rot_threshold:
              
        if orientation_diff > np.pi or orientation_diff < -np.pi:
            turn_amount = 2*np.pi - abs(orientation_diff)
        else:
            turn_amount = abs(orientation_diff)
        w_k = 0.7*abs(turn_amount)        
        if  0<=orientation_diff <= np.pi or orientation_diff <= -np.pi:
            w_k = int(np.ceil(w_k))
        else:
            w_k = -int(np.ceil(w_k))
        turning_vel = 5
        # print("w_k:")
        # print(w_k)
        motion = [0, w_k]
        turn_time = abs(operate.baseline*np.pi/(operate.scale*w_k*turning_vel)*((turn_amount)/(2*np.pi)))
        turn_time = dt
        # ===================================================================================

        operate.update_keyboard()
        operate.take_pic()
        drive_meas = operate.control(motion, turn_time)
        operate.update_slam(drive_meas)
        operate.record_data()
        operate.save_image()
        operate.detect_target()
        # visualise
        operate.draw(canvas)
        pygame.display.update()

        operate.ekf.robot.state[2] = (operate.ekf.robot.state[2]+2*np.pi)%(2*np.pi)
        
        # ===================================================================================
        robot_pose = operate.ekf.get_state_vector()[0:3,0]
        orientation_diff = desired_heading - robot_pose[2]
        print(orientation_diff)
        drive_meas = operate.control([0,0], turn_time)
        
        
    time.sleep(1)


    # Second 180 deg turn
    desired_heading = (robot_pose[2] + np.pi )%(2*np.pi)
    orientation_diff = desired_heading - robot_pose[2]
    rot_threshold = 0.1
    while abs(orientation_diff) > rot_threshold:
            
        if orientation_diff > np.pi or orientation_diff < -np.pi:
            turn_amount = abs(orientation_diff)
        else:
            turn_amount = 2*np.pi -  abs(orientation_diff)
        w_k = 0.7*abs(turn_amount)        
        if  0<orientation_diff <= np.pi or orientation_diff <= -np.pi:
            w_k = int(np.ceil(w_k))
        else:
            w_k = -int(np.ceil(w_k))
        turning_vel = 5
        # print("w_k:")
        # print(w_k)
        motion = [0, w_k]
        turn_time = abs(operate.baseline*np.pi/(operate.scale*w_k*turning_vel)*((turn_amount)/(2*np.pi)))
        turn_time = dt
        # ===================================================================================

        operate.update_keyboard()
        operate.take_pic()
        drive_meas = operate.control(motion, turn_time)
        operate.update_slam(drive_meas)
        operate.record_data()
        operate.save_image()
        operate.detect_target()
        # visualise
        operate.draw(canvas)
        pygame.display.update()

        operate.ekf.robot.state[2] = (operate.ekf.robot.state[2]+2*np.pi)%(2*np.pi)


        robot_pose = operate.ekf.get_state_vector()[0:3,0]
        orientation_diff = desired_heading - robot_pose[2]
        # print(orientation_diff)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--ip", metavar='', type=str, default='localhost')
    parser.add_argument("--port", metavar='', type=int, default=40000)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--save_data", action='store_true')
    parser.add_argument("--play_data", action='store_true')
    parser.add_argument("--ckpt", default='network/scripts/model/model.best.pth')
    parser.add_argument("--map", type=str, default='M4_true_map.txt')
    parser.add_argument("--order_list", type=str, default='search_list.txt')
    parser.add_argument("--level", type=int, default=2)
    parser.add_argument("--manual", action='store_false')
    
    args, _ = parser.parse_known_args()
    
    pygame.font.init() 
    TITLE_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 35)
    TEXT_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 40)
    
    width, height = 700, 660
    canvas = pygame.display.set_mode((width, height))
    pygame.display.set_caption('ECE4078 2021 Lab')
    pygame.display.set_icon(pygame.image.load('pics/8bit/pibot5.png'))
    canvas.fill((0, 0, 0))
    splash = pygame.image.load('pics/loading.png')
    pibot_animate = [pygame.image.load('pics/8bit/pibot1.png'),
                     pygame.image.load('pics/8bit/pibot2.png'),
                     pygame.image.load('pics/8bit/pibot3.png'),
                    pygame.image.load('pics/8bit/pibot4.png'),
                     pygame.image.load('pics/8bit/pibot5.png')]
    pygame.display.update()


    start = True

    counter = 40
    while not start:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                start = True
        canvas.blit(splash, (0, 0))
        x_ = min(counter, 600)
        if x_ < 600:
            canvas.blit(pibot_animate[counter%10//2], (x_, 565))
            pygame.display.update()
            counter += 2

    operate = Operate(args)
    operate.fruit_search = args.manual
    print('manual mode:', operate.fruit_search)
    
    operate.level = args.level

    if operate.fruit_search:
        if int(operate.level) == 1:
            
            # read in the true map
            fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
            operate.fruit_lists = fruits_list
            operate.fruits_true_pos = fruits_true_pos
            operate.aruco_true_pos = aruco_true_pos

            search_list = read_search_list()
            print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)

            lms = []
            for i in range(len(aruco_true_pos)):
                new_marker = measure.Marker(aruco_true_pos[i].reshape(2,1),i+1)
                lms.append(new_marker)
            operate.ekf.add_landmarks(lms)
            print("Landmarks initialized:")
            print(operate.ekf.get_state_vector()[0:3,0])



            while start:
                operate.update_keyboard()
                operate.draw(canvas)
                pygame.display.update()
                if operate.ekf_on:
                    # enter the waypoints
                    # instead of manually enter waypoints, you can get coordinates by clicking on a map, see camera_calibration.py
                    x,y = 0.0,0.0
                    x = input("X coordinate of the waypoint: ")
                    try:
                        x = float(x)
                    except ValueError:
                        print("Please enter a number.")
                        continue
                    y = input("Y coordinate of the waypoint: ")
                    try:
                        y = float(y)
                    except ValueError:
                        print("Please enter a number.")
                        continue
                    time.sleep(1)
                    # estimate the robot's pose
                    robot_pose = operate.ekf.get_state_vector()[0:3,0]
                    # robot drives to the waypoint
                    waypoint = [x,y]
                    robot_pose = drive_to_point(waypoint,robot_pose, operate.aruco_true_pos)

                    # Final Robot pose after attempting to drive to waypoint
                    robot_pose = operate.ekf.get_state_vector()[0:3,0]
                    print("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint,robot_pose))

                    # exit
                    uInput = input("Add a new waypoint? [Y/N]")
                    if uInput == 'N' or uInput == 'n':
                        break
        elif int(operate.level) ==2:
            operate.ekf_on = True
            # read in the true map
            fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
            operate.fruit_lists = fruits_list
            operate.fruits_true_pos = fruits_true_pos
            operate.aruco_true_pos = aruco_true_pos

            search_list = read_search_list()
            print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)

            lms = []
            for i in range(len(aruco_true_pos)):
                new_marker = measure.Marker(aruco_true_pos[i].reshape(2,1),i+1)
                lms.append(new_marker)
            operate.ekf.add_landmarks(lms)
            print("Landmarks initialized:")
            print(operate.ekf.get_state_vector()[0:3,0])
            
            time.sleep(1)
            while start:
                operate.update_keyboard()
                operate.draw(canvas)
                pygame.display.update()
                if operate.ekf_on:
                    
                    with open(args.map) as f:
                        data = f.read()
                    # reconstructing the data as a dictionary
                    js = json.loads(data)

                    with open(args.order_list) as f:
                        data = f.read()
                    # reconstructing the data as a dictionary
                    fruits_order = data.splitlines()

                    # print(js)
                    goal_list = [0]*len(fruits_order)
                    obstacle_list = []
                    obs_list = []
                    apple_list = []
                    lemon_list =[]
                    orange_list = []
                    pear_list = []
                    strawberry_list = []
                    
                    # print(js)
                    for key,value in js.items():
                        if 'apple' in key:
                            try:
                                f_idx = fruits_order.index('apple')
                            except ValueError:
                                obstacle_list.append(value)
                                obs_list.append([value['x'],value['y']])
                            else:
                                goal_list[f_idx] = value# a dictionary with x,y
                                apple_list.append([value['x'],value['y']])
                        elif 'lemon' in key:
                            try:
                                f_idx = fruits_order.index('lemon')
                            except ValueError:
                                obstacle_list.append(value)
                                obs_list.append([value['x'],value['y']])
                            else:
                                goal_list[f_idx] = value# a dictionary with x,y
                                lemon_list.append([value['x'],value['y']])
                            
                        elif 'orange' in key:
                            try:
                                f_idx = fruits_order.index('orange')
                            except ValueError:
                                obstacle_list.append(value)
                                obs_list.append([value['x'],value['y']])
                            else:
                                goal_list[f_idx] = value# a dictionary with x,y
                                orange_list.append([value['x'],value['y']])
                            
                        elif 'pear' in key:
                            try:
                                f_idx = fruits_order.index('pear')
                            except ValueError:
                                obstacle_list.append(value)
                                obs_list.append([value['x'],value['y']])
                            else:
                                goal_list[f_idx] = value# a dictionary with x,y
                                pear_list.append([value['x'],value['y']])
                            
                        elif 'strawberry' in key :
                            try:
                                f_idx = fruits_order.index('strawberry')
                            except ValueError:
                                obstacle_list.append(value)
                                obs_list.append([value['x'],value['y']])
                            else:
                                goal_list[f_idx] = value# a dictionary with x,y
                                strawberry_list.append([value['x'],value['y']])
                        else: # not including the fruits    
                            obstacle_list.append(value) # a dictionary with x,y
                            obs_list.append([value['x'],value['y']])
                    # print("Obs List")
                    # print(obstacle_list)
                    # print("Goal List")
                    # print(goal_list)

                    robot_pose = operate.ekf.get_state_vector()[0:3,0]
                    # spin_360(robot_pose)

                    goal_idx = 0
                    while goal_idx < len(goal_list):
                        

                        goal_obj = goal_list[goal_idx]

                        goal = np.array([goal_obj['x'], goal_obj['y']])
                        start = operate.ekf.get_state_vector()[0:2,0]
                        
                        
                        # testing
                        # start = [1.1992355075676007, -1.0007954680562197]
                        # goal = [1.2,1]
                        obs_rad = 0.25
                        print('starting coord', start)
                        print('current goal:', goal)
                        
                        all_obstacles_wrt_current_goal,move_back_flag = find_obs_wrt_goal(goal, goal_list, obstacle_list, obs_rad) # last number is the radius of obstacle
                        # print('obs without goal:', all_obstacles_wrt_current_goal)
                        
                        d = get_distance_to_goal(start,goal)
                        original_d = d
                        print('distance from start to goal:', d)
                        d = d/16
                        rrtc = RRTC(start=start, goal=goal, width=d, height=d, obstacle_list=all_obstacles_wrt_current_goal,
                                    expand_dis=0.15, path_resolution=0.01)


                        path_robot_to_goal = rrtc.planning() # this is a list of coordinates [x,y]
                        
                        while path_robot_to_goal == None:
                            d=d*2
                            rrtc = RRTC(start=start, goal=goal, width=d, height=d, obstacle_list=all_obstacles_wrt_current_goal,
                                    expand_dis=0.15, path_resolution=0.01)
                            if d/original_d>16:
                                break
                        print(f'using window size {d} the is {d/original_d} times of distance to goal')
                        
                        while path_robot_to_goal == None:
                            print('REDO all obs removing stuck')
                            print('goal:', goal)
                            
                            print('goal_list:')
                            print(goal_list)
                            
                            pre_obj = goal_list[goal_idx-1]
                            pre = np.array([pre_obj['x'], pre_obj['y']])
                            print('pre:', pre)
                            all_obstacles_wrt_current_goal,move_back_flag = find_obs_wrt_goal(goal, goal_list, obstacle_list, obs_rad, pre,operate.ekf.get_state_vector()[0:2,0])
                            
                            divide_2 = True
                            if move_back_flag:
                                print('MOVING BACK from')
                                # imports camera / wheel calibration parameters 
                                scale = operate.scale
                                baseline = operate.baseline
                                ####################################################
                                # PARAMETERS
                                wheel_vel = 20 # tick
                                K_pv = 1
                                K_pw = 1.5
                                dt = 0.05
                                
                                # v_k = K_pv*distance_to_goal
                                # v_k = int(np.ceil(v_k))
                                # motion = [v_k, 0]
                                
                                motion = [-3, 0]
                                # drive_time = distance_to_goal/(motion[0]*scale*wheel_vel)
                                drive_time = dt
                                # ======================================================================================

                                operate.update_keyboard()
                                operate.take_pic()
                                drive_meas = operate.control(motion, drive_time)
                                operate.update_slam(drive_meas)
                                operate.record_data()
                                operate.save_image()
                                operate.detect_target()
                                # visualise
                                operate.draw(canvas)
                                pygame.display.update()
                                print('ADJUSTING START')
                                start = operate.ekf.get_state_vector()[0:2,0]
                                
                            start = operate.ekf.get_state_vector()[0:2,0]
                            d = get_distance_to_goal(start,goal)
                            print('distance from start to goal:', d)
                            #!
                            # if divide_2:
                            #     d=d/2
    
                            rrtc = RRTC(start=start, goal=goal, width=d, height=d, obstacle_list=all_obstacles_wrt_current_goal,
                                    expand_dis=0.15, path_resolution=0.01)
                            
                            path_robot_to_goal = rrtc.planning()
                            
                            while path_robot_to_goal == None:
                                d=d*2
                                rrtc = RRTC(start=start, goal=goal, width=d, height=d, obstacle_list=all_obstacles_wrt_current_goal,
                                        expand_dis=0.15, path_resolution=0.01)
                                if d/original_d>16:
                                    break
                            print(f'using window size {d} the is {d/original_d} times of distance to goal')
                            
                            if path_robot_to_goal == None:
                                
                                obs_rad = obs_rad-0.025
                                print(f'Still stuck  - attempt to reduce obs radius to: {obs_rad}')
                                #! divide_2 = False
                                
                            
                        
                        if any(path_robot_to_goal[-1] !=goal):
                            path_robot_to_goal=path_robot_to_goal[::-1]

                        final_path = []
                        for i in range(len(path_robot_to_goal)):
                            if i%2 == 0:
                                final_path.append(path_robot_to_goal[i])
                        if (final_path[-1]!=path_robot_to_goal[-1]):    
                            final_path.append(path_robot_to_goal[-1])
                        path_robot_to_goal = final_path

                        plot_path(path_robot_to_goal, obs_list,'path',None,goal_idx, apple_list,lemon_list,orange_list,pear_list,strawberry_list,obs_rad)
                        plt.savefig
                        
                        print('\n#############################################\nfinish rrtc for goal ', goal_idx)


                        print('rrtc path from:',path_robot_to_goal[0],'to',path_robot_to_goal[-1])
                        
                        
                        robot_pose_list =[]

                        robot_pose = operate.ekf.get_state_vector()[0:3,0]
                        print("Robot Pose ")
                        robot_pose_list.append(robot_pose[0:2])
                        calc_new_path = False

                    
                        
                        node_idx = 1
                        no_obs_near_line = False
                        while node_idx < len(path_robot_to_goal):
                            if not no_obs_near_line:
                                waypoint = path_robot_to_goal[node_idx]
                            else:
                                waypoint = [goal_list[goal_idx]['x'], goal_list[goal_idx]['y']]
                            # print('\nwaypoint dest:',waypoint)
                            drive_to_point(waypoint,robot_pose,operate.aruco_true_pos)
                            if calc_new_path:
                                break

                            robot_pose = operate.ekf.get_state_vector()[0:3,0]
                            robot_pose_list.append(robot_pose[0:2])
                            print("reaching waypoint: {}; New robot pose: {}".format(waypoint,robot_pose))
                            # stop 
                            
                            distance_to_goal =get_distance_to_goal(waypoint, path_robot_to_goal[-1])
                            print('distance to goal remaining:', distance_to_goal)
                            if distance_to_goal < 0.35:
                                print('within stoping range to collect goal', goal_idx)
                                plot_path(path_robot_to_goal, obs_list,'path',robot_pose_list,goal_idx,apple_list,lemon_list,orange_list,pear_list,strawberry_list,obs_rad)
                                break
                            node_idx += 1

                            # if i % 10 == 0:
                            plot_path(path_robot_to_goal, obs_list,'path',robot_pose_list,goal_idx,apple_list,lemon_list,orange_list,pear_list,strawberry_list,obs_rad)
                        
                        """
                            ### Calculate Line ###
                            robot_pose = operate.ekf.get_state_vector()[0:3,0]
                            aruco_pose = operate.ekf.get_state_vector()[3:,0]
                            goal_x = goal_list[goal_idx]['x']
                            goal_y = goal_list[goal_idx]['y']
                            goal_pose = [goal_x, goal_y]
                            goal_heading = get_desired_heading(goal_pose, robot_pose) - np.pi # get normal rads in unit circle orientation
                            current_pose = robot_pose[0:2]
                            delta_x = 0.005
                            no_obs_near_line = True
                            
                            dist_to_goal = np.sqrt((goal_pose[0]-current_pose[0])**2+(goal_pose[1]-current_pose[1])**2)
                            



                            while  dist_to_goal >= 0.05:
                                for i in range(len(aruco_pose)//2):
                                    aruco_x = aruco_pose[2*i]
                                    aruco_y= aruco_pose[2*i+1]
                                    dist = np.sqrt((current_pose[0]-aruco_x)**2+(current_pose[1]-aruco_y)**2)
                                    if dist < 0.25:
                                        no_obs_near_line = False
                                        break
                                current_pose[0] += -delta_x*np.cos(goal_heading)    # because gazebo coords are flipped
                                current_pose[1] += -delta_x*np.sin(goal_heading) # because gazebo coords are flipped

                                
                                dist_to_goal = np.sqrt((goal_pose[0]-current_pose[0])**2+(goal_pose[1]-current_pose[1])**2)

                        
                            if no_obs_near_line:
                                spin_360(robot_pose)

                                robot_pose = operate.ekf.get_state_vector()[0:3,0]
                                aruco_pose = operate.ekf.get_state_vector()[3:,0]
                                goal_x = goal_list[goal_idx]['x']
                                goal_y = goal_list[goal_idx]['y']
                                goal_pose = [goal_x, goal_y]
                                goal_heading = get_desired_heading(goal_pose, robot_pose) - np.pi # get normal rads in unit circle orientation
                                current_pose = robot_pose[0:2]
                                delta_x = 0.005
                                no_obs_near_line = True
                                
                                dist_to_goal = np.sqrt((goal_pose[0]-current_pose[0])**2+(goal_pose[1]-current_pose[1])**2)
                                


                                if dist_to_goal < 0.5:
                                    while  dist_to_goal >= 0.05:
                                        for i in range(len(aruco_pose)//2):
                                            aruco_x = aruco_pose[2*i]
                                            aruco_y= aruco_pose[2*i+1]
                                            dist = np.sqrt((current_pose[0]-aruco_x)**2+(current_pose[1]-aruco_y)**2)
                                            if dist < 0.25:
                                                no_obs_near_line = False
                                                break
                                        current_pose[0] += -delta_x*np.cos(goal_heading)    # because gazebo coords are flipped
                                        current_pose[1] += -delta_x*np.sin(goal_heading) # because gazebo coords are flipped
        
                                        
                                        dist_to_goal = np.sqrt((goal_pose[0]-current_pose[0])**2+(goal_pose[1]-current_pose[1])**2)
                                else:
                                    no_obs_near_line = False
                        """

                        # Implement correct obs
                        # Update new obstacle and 
                        aruco_pose = operate.ekf.get_state_vector()[3:,0]
                        obstacle_list = []
                        for i in range(len(aruco_pose)//2):
                            # print("updating obs")
                            ob = {
                                'x': aruco_pose[2*i],
                                'y': aruco_pose[2*i+1]
                            }
                            obstacle_list.append(ob)
                        
                        ## Implement correct fruit pose based on new aruco positions?

                        if not calc_new_path:
                            print("Next Goal")
                            goal_idx +=1
                            time.sleep(3)



                    break
    else:
        while start:
            operate.update_keyboard()
            operate.take_pic()
            drive_meas = operate.control()
            operate.update_slam(drive_meas)
            operate.record_data()
            operate.save_image()
            operate.detect_target()
            
            #! Calista #######################
            pose = operate.ekf.get_state_vector()[0:3]
            robot_pose = [pose[0][0], pose[1][0], pose[2][0]] # return as a flattened array or will get errors in drive to point bc of numpy arrays and scalars
            map, depth = check_for_target(robot_pose)
            
            # visualise
            operate.draw(canvas)
            pygame.display.update()



