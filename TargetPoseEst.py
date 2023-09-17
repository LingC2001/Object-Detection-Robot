# estimate the pose of a target object detected
from re import X
import numpy as np
import json
import os
from pathlib import Path
import ast
import cv2
import math
# from machinevisiontoolbox import Image
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import PIL

# use the machinevision toolbox to get the bounding box of the detected target(s) in an image
# from network.scripts.detector import Detector
from network.scripts.detector_yolo import Detector

def detect_target(img, detector,count=0,visualise=False):
    network_vis = np.ones((240, 320,3))* 100
    if detector is not None:
        
        output, network_vis = detector.detect_single_image(img)
        #print(network_vis.shape)
        detector_output = output[0].numpy()
        #print(detector_output)
        if not os.path.exists('lab_output/detection_view'):
            os.makedirs('lab_output/detection_view')
        if visualise:
            cv2.imshow(f"out{count}", network_vis)
            cv2.waitKey(0)
        cv2.imwrite(f'lab_output/detection_view/out{count}.png',network_vis)
        return detector_output
    else:
        return "ohno"

def get_bounding_box(bbox): # bounding box is arranged [x1, y1, x2, y2] but want to convert to # [[xcenter],[ycenter],[width],[height]]
    bbox_height = bbox[3]-bbox[1]
    bbox_width = bbox[2]-bbox[0]
    bbox_x = bbox[0]+bbox_width/2
    bbox_y = bbox[1]+bbox_height/2
    #print([[bbox_x],[bbox_y],[bbox_width],[bbox_height]])
    return [[bbox_x],[bbox_y],[bbox_width],[bbox_height]]

# read in the list of detection results with bounding boxes and their matching robot pose info
def get_image_info(image):
    # there are at most five types of targets in each image
    target_lst_box = [[], [], [], [], []]
    target_lst_pose = [[], [], [], [], []]
    completed_img_dict = {}

    bboxes = image['bbox'] # image is a dictionary containing robot pose + lists for bounding boxes and their corresponding labels
    labels = image['labels']
    poses = image['pose']
    i = 0
    for label in labels:
        try: 
            box = get_bounding_box(bboxes[i]) # [[xcenter],[ycenter],[width],[height]]
            pose = [poses]
            target_lst_box[int(label)].append(box) # append current bounding box to list of bounding boxes in image
            target_lst_pose[int(label)].append(np.array(pose).reshape(3,)) # append corresponding robot pose for image for each bounding box/detection
        except:
            pass
        i = i + 1
  

    # if there are more than one objects of the same type, combine them
    for i in range(5):
        if len(target_lst_box[i])>0:
            box = np.stack(target_lst_box[i], axis=1)
            pose = np.stack(target_lst_pose[i], axis=1)
            completed_img_dict[i] = {'target': box, 'robot': pose}

    return completed_img_dict

# estimate the pose of a target based on size and location of its bounding box in the robot's camera view and the robot's pose
def estimate_pose(base_dir, camera_matrix, completed_img_dict, runtype):
    camera_matrix = camera_matrix
    focal_length_x = camera_matrix[0][0]
    focal_length_y = camera_matrix[1][1]
    # actual sizes of targets [For the simulation models]
    # You need to replace these values for the real world objects
    target_dimensions = []
    
    if runtype == 'sim':
        apple_dimensions = [0.075448, 0.074871, 0.071889]
        target_dimensions.append(apple_dimensions)
        lemon_dimensions = [0.060588, 0.059299, 0.053017]
        target_dimensions.append(lemon_dimensions)
        orange_dimensions = [0.0721, 0.0771, 0.0739]
        target_dimensions.append(orange_dimensions)
        pear_dimensions = [0.0946, 0.0948, 0.135]
        target_dimensions.append(pear_dimensions)
        strawberry_dimensions = [0.052, 0.0346, 0.0376]
        target_dimensions.append(strawberry_dimensions)
    elif runtype == 'physical':
        apple_dimensions = [0.075, 0.075, 0.073]
        target_dimensions.append(apple_dimensions)
        lemon_dimensions = [0.048, 0.069, 0.053]
        target_dimensions.append(lemon_dimensions)
        orange_dimensions = [0.07, 0.07, 0.0767]
        target_dimensions.append(orange_dimensions)
        pear_dimensions = [0.065, 0.073, 0.116]
        target_dimensions.append(pear_dimensions)
        strawberry_dimensions = [0.037, 0.048, 0.067]
        target_dimensions.append(strawberry_dimensions)

    target_list = ['apple', 'lemon', 'orange', 'pear', 'strawberry']
    target_pose_dict = {}
    # for each target in each detection output, estimate its pose
    for target_num in completed_img_dict.keys():
        box = completed_img_dict[target_num]['target'] # [[xcenter],[ycenter],[width],[height]]

        robot_pose = completed_img_dict[target_num]['robot'] # [[x], [y], [theta]]
        true_height = target_dimensions[target_num][2] # z dimension of target_dimension corresponds to height
        # print("True height", true_height)
        box = np.moveaxis(box, 0,2)
        box = box.reshape(-1,4)
        # print("box",box)
        # print("robot_pose",robot_pose)
        ######### Replace with your codes #########
        # TODO: compute pose of the target based on bounding box info and robot's pose
        # physical robot 320, 240
        # for camera dimension 640,480 (0,0) at image center


        # DON'T DETECT WRONG OBJ OR WRONG HEIGHT
        for i in range(box.shape[0]):

            
            x_c = (camera_matrix[0][2]-box[i][0]) # get coordinates of center of bbox relative to center of image/principal axis
            y_c = (camera_matrix[1][2]-box[i][1])
            # print(x_c)
            # print('focal_length x',focal_length_x, 'focal_length y:', focal_length_y)
        

            box_height = box[i][3]
            depth = (focal_length_y/box_height)*true_height # similar triangles


            y_robotf = (x_c/focal_length_x)*depth
            x_robotf = depth
            

            #print( x_robotf, y_robotf)
            pose_in_robot_frame = np.array([[x_robotf],[y_robotf]])
            robotp = np.array([[robot_pose[0][i]],[robot_pose[1][i]]])

            theta = robot_pose[2][i]

            rot_mat = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta),np.cos(theta)]])
            world_pose = rot_mat@pose_in_robot_frame + robotp

            target_pose = {'y': world_pose[1][0], 'x': world_pose[0][0]} # make sure they are in correct order y,x
            
            
            target_pose_dict[target_list[target_num]+str(i)] = target_pose
            #print(target_list[target_num])
            #print(target_pose)

            ###########################################
    
    return target_pose_dict

def plot_cluster(est, name):
    if not os.path.exists('lab_output/cluster_view'):
        os.makedirs('lab_output/cluster_view')
    plt.figure()
    a = [x[0] for x in est]
    a1 = [x[1] for x in est]
    plt.plot(a,a1,'o')

    plt.savefig(f'lab_output/cluster_view/{name}_cluster.png')

# merge the estimations of the targets so that there are at most 3 estimations of each target type
def merge_estimations(target_pose_dict):
    target_map = target_pose_dict
    apple_est, lemon_est, pear_est, orange_est, strawberry_est = [], [], [], [], []
    target_est = {}

    # combine the estimations from multiple detector outputs
    for f in target_map:
        for key in target_map[f]:
            if key.startswith('apple'):
                apple_est.append(
                    np.array(list(target_map[f][key].values()), dtype=float))
            elif key.startswith('lemon'):
                lemon_est.append(
                    np.array(list(target_map[f][key].values()), dtype=float))
            elif key.startswith('pear'):
                pear_est.append(
                    np.array(list(target_map[f][key].values()), dtype=float))
            elif key.startswith('orange'):
                orange_est.append(
                    np.array(list(target_map[f][key].values()), dtype=float))
            elif key.startswith('strawberry'):
                strawberry_est.append(
                    np.array(list(target_map[f][key].values()), dtype=float))
                
    plot_cluster(apple_est,'apple')
    plot_cluster(lemon_est,'lemon')
    plot_cluster(orange_est,'orange')
    plot_cluster(pear_est,'pear')
    plot_cluster(strawberry_est,'straw')
    

    ######### Replace with your codes #########
    # TODO: the operation below takes the first three estimations of each target type, replace it with a better merge solution
    if len(apple_est) > 1:
        kmeans2 = KMeans(n_clusters=2, random_state=0).fit(apple_est)
        kmeans1 = KMeans(n_clusters=1, random_state=0).fit(apple_est)
        #print('apple ', kmeans1.inertia_ - kmeans2.inertia_)
        if kmeans1.inertia_ - kmeans2.inertia_ < 0.2:  # ! 1.5 is arbitrary, inertia is the sum of square distance
            apple_est = kmeans1.cluster_centers_
        else:
            apple_est = kmeans2.cluster_centers_

    if len(lemon_est) > 1:
        kmeans2 = KMeans(n_clusters=2, random_state=0).fit(lemon_est)
        kmeans1 = KMeans(n_clusters=1, random_state=0).fit(lemon_est)
        #print('lemon ', kmeans1.inertia_ - kmeans2.inertia_)
        if kmeans1.inertia_ - kmeans2.inertia_ < 0.2:  # ! 1.5 is arbitrary, inertia is the sum of square distance
            lemon_est = kmeans1.cluster_centers_
        else:
            lemon_est = kmeans2.cluster_centers_

    if len(pear_est) > 1:
        kmeans2 = KMeans(n_clusters=2, random_state=0).fit(pear_est)
        kmeans1 = KMeans(n_clusters=1, random_state=0).fit(pear_est)
        #print('pear ', kmeans1.inertia_ - kmeans2.inertia_)
        if kmeans1.inertia_ - kmeans2.inertia_ < 0.2:  # ! 1.5 is arbitrary, inertia is the sum of square distance
            pear_est = kmeans1.cluster_centers_
        else:
            pear_est = kmeans2.cluster_centers_

    if len(orange_est) > 1:
        kmeans2 = KMeans(n_clusters=2, random_state=0).fit(orange_est)
        kmeans1 = KMeans(n_clusters=1, random_state=0).fit(orange_est)
        #print('orange ', kmeans1.inertia_ - kmeans2.inertia_)
        if kmeans1.inertia_ - kmeans2.inertia_ < 0.2:  # ! 1.5 is arbitrary, inertia is the sum of square distance
            orange_est = kmeans1.cluster_centers_
        else:
            orange_est = kmeans2.cluster_centers_

    if len(strawberry_est) > 1:
        kmeans2 = KMeans(n_clusters=2, random_state=0).fit(strawberry_est)
        kmeans1 = KMeans(n_clusters=1, random_state=0).fit(strawberry_est)
        #print('strawberry ', kmeans1.inertia_ - kmeans2.inertia_)
        if kmeans1.inertia_ - kmeans2.inertia_ < 0.2:  # ! 1.5 is arbitrary, inertia is the sum of square distance
            strawberry_est = kmeans1.cluster_centers_
        else:
            strawberry_est = kmeans2.cluster_centers_


    for i in range(3):
        # except here is to deal with list with lenght of 1 ( out of indices problem)
        try:
            target_est['apple_' +
                       str(i)] = {'y': apple_est[i][0], 'x': apple_est[i][1]}
        except:
            pass
        try:
            target_est['lemon_' +
                       str(i)] = {'y': lemon_est[i][0], 'x': lemon_est[i][1]}
        except:
            pass
        try:
            target_est['pear_' +
                       str(i)] = {'y': pear_est[i][0], 'x': pear_est[i][1]}
        except:
            pass
        try:
            target_est['orange_' +
                       str(i)] = {'y': orange_est[i][0], 'x': orange_est[i][1]}
        except:
            pass
        try:
            target_est['strawberry_' +
                       str(i)] = {'y': strawberry_est[i][0], 'x': strawberry_est[i][1]}
        except:
            pass
    ###########################################

    return target_est

# def merge_class(class_est):
# 
#   kmeans = KMeans(n_clusters=2, random_state=0).fit(class_est)
#   return kmeans.cluster_centers_
        

if __name__ == "__main__":

    # camera_matrix = np.ones((3,3))/2
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtype", type=str, default='sim')
    args, _ = parser.parse_known_args()
    fileK = "{}intrinsic.txt".format('./calibration/param/')
    camera_matrix = np.loadtxt(fileK, delimiter=',')
    base_dir = Path('./')
    
    print(args.runtype)
    if args.runtype == "sim":
        USE_SIMULATION_DETECTOR = True
    else:
        USE_SIMULATION_DETECTOR = False

    print(USE_SIMULATION_DETECTOR)

    detector = Detector(use_gpu=False)
    detector.load_weights(sim=USE_SIMULATION_DETECTOR)

    # a dictionary of all the saved detector outputs
    image_poses = []
    with open(base_dir/'lab_output/images.txt') as fp:
        img_count = 0
        for line in fp.readlines():
            pose_dict = ast.literal_eval(line) # [["pose"],["bbox"],["label"]]
            
            pose = pose_dict["pose"]
            file_path = pose_dict["imgfname"]
            print(file_path)
            # Read img
            image = cv2.imread(file_path)
            
            # Pass into yolov7
            #print(image.shape)
            detector_output = detect_target(image, detector,img_count)
            img_count+=1 
            #print(detector_output.shape)
            # reconstructing output dict
            bboxs = detector_output[:,0:4]
            labels = detector_output[:,5]
            confs = detector_output[:,4]
            out = {
                "pose" : pose,
                "bbox" : bboxs.tolist(),
                "confs" : confs.tolist(),
                "labels" : labels.tolist()
            }

            # combine pose from images.txt with output of yolo into a dict
            image_poses.append(out) # list of dict
    
    # estimate pose of targets in each detector output
    target_map = {} 
    key = 0       
    for image in image_poses:
        completed_img_dict = get_image_info(image) # image is a dict
        target_map[key] = estimate_pose(base_dir, camera_matrix, completed_img_dict, args.runtype)
        key = key+1
    # print(target_map)

    # merge the estimations of the targets so that there are at most 3 estimations of each target type
    target_est = merge_estimations(target_map)
    #print(target_est)
                     
    # save target pose estimations
    with open(base_dir/'lab_output/targets.txt', 'w') as fo:
        json.dump(target_est, fo)
    
    print('Estimations saved!')
    print('view lab_output folder for cluster and detection bounding box')
