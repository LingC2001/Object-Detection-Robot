import os
from shutil import move
from turtle import window_width
import cv2
import math
from obstacle import *
# import matplotlib.pyplot as plt
#! New
from matplotlib import pyplot as plt, patches

def find_obs_wrt_goal(goal_next, goal_list, obstacle_list, obs_radius, pre = None, stuck_start=None, wall_as_obs = False): # this is to ensure the goal is not in the obstacle list
    checking = True
    all_obstacles = []
    all_obs_coord = []
    # print('goalnext:',goal_next)
    goal_x = goal_next[0]
    goal_y = goal_next[1]
    print('pre:', pre)
    print('stuck_start:',stuck_start)
    move_back_flag =0
    

    for obs in obstacle_list:
        obs_x = obs['x'] 
        obs_y = obs['y'] 
        # print(obs_x,obs_y)
        
        if np.all(stuck_start) != None:
            if abs(stuck_start[0] - obs_x) < 0.15 and abs(stuck_start[1] - obs_y) < 0.15:
                    print(f'ignoring the obs at [{obs_x},{obs_y}] as obstacle because it is too near the robot start')
                    move_back_flag =1
                    continue
                
        all_obstacles.append(Circle(obs_x, obs_y, obs_radius))
        all_obs_coord.append([obs_x, obs_y])

    if wall_as_obs:
        wall_coord = [(0,1.5),(0,-1.5),(1.5,0),(-1.5,0)] # x,,y
        wall_pos = ['u','d','r','l']
        for w_coord, w_pose in zip(wall_coord,wall_pos):
            if w_pose == 'u' or w_pose == 'd':
                w_width = 1.5
                w_height = 0.1
            else:
                w_width = 0.1
                w_height = 1.5
            all_obs_coord.append(Rectangle(w_coord,w_width,w_height,w_pose)) 
            
            if np.all(stuck_start) != None:
                if abs(stuck_start[0]) > 1.5 or  abs(stuck_start[1] ) > 1.5:
                        print(f'ignoring the obs at [{obs_x},{obs_y}] as obstacle because it is too near the robot start')
                        # move_back_flag =1
                        continue
        
    for obs in goal_list:
        obs_x = obs['x'] 
        obs_y = obs['y'] 
        
        if np.all(stuck_start) != None:
            if abs(stuck_start[0] - obs_x) < 0.15 and abs(stuck_start[1] - obs_y) < 0.15:
                    print(f'ignoring this goal at [{obs_x},{obs_y}] as obstacle because it is too near the robot start')
                    move_back_flag =1
                    continue
        
        # if np.all(pre) != None:
        #     pre_x = pre[0]
        #     pre_y = pre[1]
        #     if abs(pre_x - obs_x) < 0.05 and abs(pre_y - obs_y) < 0.05:
        #         print('ignoring the pre goal as obstacle')
        #         continue        
            
        if abs(goal_x - obs_x) < 0.05 and abs(goal_y - obs_y) < 0.05:
            continue
        else: 
            all_obstacles.append(Circle(obs_x, obs_y, obs_radius))
            all_obs_coord.append([obs_x, obs_y])
            # print('goal:',[obs_x, obs_y])
    if checking:
        print('all_obs_coord:\n', all_obs_coord)
    return all_obstacles,move_back_flag # with respect to the next goal

def scatter_plot (est, color):
    if est == None:
        return 0
    a = [x[0] for x in est]
    a1 = [x[1] for x in est]
    plt.scatter(a, a1, s=100, color = color, marker='*')
    
    
def plot_path(est,obs, name, pose=None, count = None, apple_list = None, lemon_list = None, orange_list = None, pear_list = None, strawberry_list = None, obs_radius =0.25):
    if not os.path.exists('path_vis'):
        os.makedirs('path_vis')
    plot = plt.figure()
    a = [x[0] for x in est]
    a1 = [x[1] for x in est]
    plt.plot(a, a1,'-o')
    b = [x[0] for x in obs]
    b1 = [x[1] for x in obs] 
    plt.scatter(b, b1,c='r')
    
    ax = plot.add_subplot()
    circle_list = [patches.Circle((x[0], x[1]), radius=obs_radius, color='red', fill=False) for x in obs]
    for circle in circle_list:
        ax.add_patch(circle)
    
    # fruit
    scatter_plot(apple_list,'red')
    scatter_plot(orange_list,'orange')
    scatter_plot(lemon_list,'yellow')
    scatter_plot(pear_list,'green')
    scatter_plot(strawberry_list,'pink')
    
    
    if pose != None:
        # print('plotting pose')
        c = [x[0] for x in pose]
        c1 = [x[1] for x in pose]
        plt.plot(c,c1,'-x',c='y')
    ax = plt.gca()
    ax.invert_yaxis()
    ax.invert_xaxis()
    
    # plt.show()
    plt.savefig(f'path_vis/{name}{count}.png') 

# This is an adapted version of the RRT implementation done by Atsushi Sakai (@Atsushi_twi)
class RRTC:
    """
    Class for RRT planning
    """
    class Node:
        """
        RRT Node
        """
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.path_x = []
            self.path_y = []
            self.parent = None

    def __init__(self, start=np.zeros(2),
                 goal=np.array([120,90]),
                 obstacle_list=None,
                 width = 160,
                 height=100,
                 expand_dis=3.0, 
                 path_resolution=0.5, 
                 max_points=200):
        """
        Setting Parameter
        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacle_list: list of obstacle objects
        width, height: search area
        expand_dis: min distance between random node and closest node in rrt to it
        path_resolion: step size to considered when looking for node to expand
        """
        self.start = self.Node(start[0], start[1])
        self.end = self.Node(goal[0], goal[1])
        self.width = width
        self.height = height
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.max_nodes = max_points
        self.obstacle_list = obstacle_list
        self.start_node_list = [] # Tree from start
        self.end_node_list = [] # Tree from end
        
    def planning(self):
        """
        rrt path planning
        """
        self.start_node_list = [self.start]
        self.end_node_list = [self.end]
        
        count = 0
        
        while len(self.start_node_list) + len(self.end_node_list) <= self.max_nodes:

        #TODO: Complete the planning method ----------------------------------------------------------------

            ## 1 - Sample random node and add to "start" tree
            rnd_node = self.get_random_node() # sample random node

            expansion_ind = self.get_nearest_node_index(self.start_node_list, rnd_node) # get nearest node in start tree to random node
            expansion_node = self.start_node_list[expansion_ind]

            new_node = self.steer(expansion_node, rnd_node, self.expand_dis) # find node closer to random node - MAKE SURE TO SPECIFY EXPANSION DISTANCE

            #! ######
            # print('found new node:', [new_node.x,new_node.y])
            # print(len(self.start_node_list))
            # print(len(self.end_node_list))
            
            if count > 5000:
                print('\nINFINITE LOOP')
                return None
            count +=1
            
            
            
            if self.is_collision_free(new_node) or len(self.start_node_list)<2:
                # print("collision free ")
                self.start_node_list.append(new_node) # if collision free add to start list
                
                ## 2 - Using the new node added to start tree, add a node to "end" tree in this direction (follow pseudocode)
                end_exp_ind = self.get_nearest_node_index(self.end_node_list, new_node) 
                end_exp_node = self.end_node_list[end_exp_ind]

                new_end_node = self.steer(end_exp_node, new_node,self.expand_dis)

                if self.is_collision_free(new_end_node):
                    self.end_node_list.append(new_end_node)

                ## 3 - Check if we can merge
                closest_ind = self.get_nearest_node_index(self.end_node_list, new_node)
                closest_node = self.end_node_list[closest_ind]
                d, _ = self.calc_distance_and_angle(closest_node, new_node)
                
                #! ######
                # print('distance:',d)
                
                if d < self.expand_dis:
                    # print('path found count:', count)
                    # print("if d < self.expand_dis: ")
                    self.end_node_list.append(new_node)
                    self.start_node_list.append(closest_node)
                    return self.generate_final_course(len(self.start_node_list) - 1, len(self.end_node_list) - 1)

                ## 4 - Merge trees
                self.start_node_list, self.end_node_list = self.end_node_list, self.start_node_list
        print('no path found count:',count)    
        return None  # cannot find path
    
    # ------------------------------DO NOT change helper methods below ----------------------------
    def steer(self, from_node, to_node, extend_length=float("inf")):
        """
        Given two nodes from_node, to_node, this method returns a node new_node such that new_node 
        is “closer” to to_node than from_node is.
        """
        
        new_node = self.Node(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)

        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]

        if extend_length > d:
            extend_length = d

        # How many intermediate positions are considered between from_node and to_node
        n_expand = math.floor(extend_length / self.path_resolution)

        # Compute all intermediate positions
        for _ in range(n_expand):
            new_node.x += self.path_resolution * cos_theta
            new_node.y += self.path_resolution * sin_theta
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)

        d, _ = self.calc_distance_and_angle(new_node, to_node)
        if d <= self.path_resolution:
            new_node.path_x.append(to_node.x)
            new_node.path_y.append(to_node.y)

        new_node.parent = from_node

        return new_node

    def is_collision_free(self, new_node):
        """
        Determine if nearby_node (new_node) is in the collision-free space.
        """
        if new_node is None:
            return True
        
        points = np.vstack((new_node.path_x, new_node.path_y)).T

        for obs in self.obstacle_list:
            in_collision = obs.is_in_collision_with_points(points)
            if in_collision:
                return False
        
        return True  # safe
    
    def generate_final_course(self, start_mid_point, end_mid_point):
        """
        Reconstruct path from start to end node
        """
        # First half
        node = self.start_node_list[start_mid_point]
        path = []
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])
        
        # Other half
        node = self.end_node_list[end_mid_point]
        path = path[::-1]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])

        return path

    def calc_dist_to_goal(self, x, y):
        dx = x - self.end.x
        dy = y - self.end.y
        return math.hypot(dx, dy)

    def get_random_node(self):
        x = self.width * np.random.random_sample()
        y = self.height * np.random.random_sample()
        rnd = self.Node(x, y)
        return rnd

    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):        
        # Compute Euclidean disteance between rnd_node and all nodes in tree
        # Return index of closest element
        dlist = [(node.x - rnd_node.x) ** 2 + (node.y - rnd_node.y)
                 ** 2 for node in node_list]
        minind = dlist.index(min(dlist))
        return minind

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta        