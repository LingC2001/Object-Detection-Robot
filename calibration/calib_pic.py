# for taking a photo of the calibration rig
import numpy as np
import cv2
from cv2 import aruco
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, "../util")
from pibot import PenguinPi
import pygame

class calibration:
    def __init__(self,args):
        self.pibot = PenguinPi(args.ip, args.port)
        self.img = np.zeros([240,320,3], dtype=np.uint8)
        self.command = {'motion':[0, 0], 'image': False}
        self.finish = False
        self.last_keys_pressed = [False, False, False, False, False]

    def image_collection(self, dataDir, images_to_collect):
        if self.command['image']:
            for i in range(images_to_collect):
                image = self.pibot.get_image()
                filename = "calib_{}.png".format(i)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(filename, image)
            self.finish = True

    def scale_speed(self):
        keys_pressed = pygame.key.get_pressed()
        shift_pressed = keys_pressed[pygame.K_LSHIFT] or keys_pressed[pygame.K_RSHIFT]
        if shift_pressed == True:
            speedscale = 3
        else:
            speedscale = 1
        return speedscale 
        
    def update_keyboard(self):
        for event in pygame.event.get():
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
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.command['motion'] = [0, 0]
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                self.command['image'] = True


    def control(self):
        motion_command = self.command['motion']
        lv, rv = self.pibot.set_velocity(motion_command)

    def take_pic(self):
        self.img = self.pibot.get_image()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", metavar='', type=str, default='localhost')
    parser.add_argument("--port", metavar='', type=int, default=40000)
    args, _ = parser.parse_known_args()

    currentDir = os.getcwd()
    dataDir = "{}/param/".format(currentDir)
    if not os.path.exists(dataDir):
        os.makedirs(dataDir)
    
    images_to_collect = 1

    calib = calibration(args)

    width, height = 640, 480
    canvas = pygame.display.set_mode((width, height))
    pygame.display.set_caption('Calibration')
    canvas.fill((0, 0, 0))
    pygame.display.update()
    
    # collect data
    print('Collecting {} images for camera calibration.'.format(images_to_collect))
    print('Press ENTER to capture image.')
    while not calib.finish:
        
        calib.update_keyboard()
        calib.control()
        calib.take_pic()
        calib.image_collection(dataDir, images_to_collect)
        img_surface = pygame.surfarray.make_surface(calib.img)
        img_surface = pygame.transform.flip(img_surface, True, False)
        img_surface = pygame.transform.rotozoom(img_surface, 90, 1)
        canvas.blit(img_surface, (0, 0))
        pygame.display.update()
    print('Finished image collection.\n')


