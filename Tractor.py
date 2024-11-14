import math
import numpy as np
import random

class Tractor:
    def __init__(self) -> None:
        self.x = 0 # x coordinate
        self.y = 0 # y coordinate
        self.th = 0 # yaw heading (theta)

        self.x_dot = 0 # x velocity
        self.y_dot = 0 # y velocity
        self.th_dot = 0 # theta ang velocity

        self.l = 3.15 # wheelbase
        self.a_fwd = 0.336 # acceleration constant
        self.v_tol = 0.01 # velocity tolerance/hysterisis
        self.d_dot = 0.4 # wheel angle angular velocity
        self.max_d = 0.4 # max wheel angle 
        self.d_tol = 0.01 # wheel angle tolerance/hysteresis

        self.v_command = 0 # commanded velocity
        self.d_command = 0 # commanded wheel angle

        self.v = 0 # actual velocity
        self.d = 0 # actual wheel angle

        self.rear = -1
        self.front = self.l + 1
        self.right = -1.5
        self.left = 1.5
        self.footprint = np.array([[self.front, self.rear, self.rear,  self.front],
                                   [self.left,  self.left, self.right, self.right]])

        self.start_x = 0
        self.start_y = 0

        self.goal_x = None
        self.goal_y = None

        self.goal_tractor = np.array([[0],[0]])
        self.prev_goal_tractor = np.array([[0],[0]])

        self.prev_goal_x = self.start_x
        self.prev_goal_y = self.start_y

        self.path = []
        self.path_goal_index = 0 # element in self.path which is the current goal
        self.path_length = 0

        self.goal_tol = 0.5
        
        self.prev_path_hdg = 2*random.randint(0,1)-1

    def distance_to_rectangle_edges(self, phi):
        #vertices = np.array([[self.front, self.rear, self.rear, self.front],
        #                [self.left, self.left, self.right, self.right]])
        R = float('inf')
        for i in range(4):
            if(i<3):
                x2 = self.footprint[0,i+1]
                y2 = self.footprint[1,i+1]
                x1 = self.footprint[0,i]
                y1 = self.footprint[1,i]
            else:
                x2 = self.footprint[0,0]
                y2 = self.footprint[1,0]
                x1 = self.footprint[0,3]
                y1 = self.footprint[1,3]

            t = (y1*math.cos(phi) - x1*math.sin(phi))/(y1*math.cos(phi) - y2*math.cos(phi) - x1*math.sin(phi) + x2*math.sin(phi))
            r = -(x1*y2 - x2*y1)/(y1*math.cos(phi) - y2*math.cos(phi) - x1*math.sin(phi) + x2*math.sin(phi))
            if((t <= 1 and t >= 0) and r < R and r > 0):
                R = r
    
        return R

    def set_goal(self,x,y):
        self.goal_x = x
        self.goal_y = y

    def set_random_goal(self):
        self.goal_x = 10*random.random()-5
        self.goal_y = 10*random.random()-5
    
    def set_random_polar_goal(self, min_radius, max_radius, min_angle, max_angle, butterfly):
        r = (max_radius - min_radius)*random.random()+min_radius
        phi = (max_angle - min_angle)*random.random()+min_angle
        b = 1
        if(random.random()>0.5 and butterfly):
            b = -1

        self.goal_x = b*r*math.cos(phi)
        self.goal_y = b*r*math.sin(phi)

    def get_goal_dist(self):
        return math.sqrt((self.goal_x-self.x)**2 + (self.goal_y-self.y)**2)
    
    def get_normalized_dist(self, max_dist):
        return self.get_goal_dist()/max_dist
    
    def get_goal_direction(self):
        return math.atan2(self.goal_y-self.y, self.goal_x-self.x)

    def get_normalized_direction(self, angle):
        return (angle+math.pi)/(2*math.pi)
    
    def get_heading_error(self):
        return self.wrap_angle(self.get_goal_direction() - self.th)
    
    def get_norm_observation(self):
        return np.array([self.get_normalized_dist(math.sqrt(2*50**2)), self.get_heading_error()/math.pi])
        #return np.array([self.get_normalized_dist(math.sqrt(2*50**2)), self.get_heading_error()/math.pi, self.v/5, self.v_command/5, self.d/0.4,self.d_command/0.4])
    
    def get_observation(self):
        R = np.array([[math.cos(self.th), math.sin(self.th)],
                      [-math.sin(self.th), math.cos(self.th)]])
        self.prev_goal_tractor = self.goal_tractor
        self.goal_tractor = R@np.array([[self.goal_x - self.x],
                                       [self.goal_y - self.y]])
        
        #return np.array([self.goal_tractor[0,0], self.goal_tractor[1,0], self.prev_goal_tractor[0,0], self.prev_goal_tractor[1,0]])
        '''print(self.get_next_goal())
        return np.array([self.goal_tractor[0,0]/30, self.goal_tractor[1,0]/30, self.get_distance_to_path()/10, self.d_command/0.4, self.d/0.4])'''
        #observation = current goal in local coordinates, next goal in local coordinates, distance to path
        next_goal = self.get_next_goal()
        next_goal_tractor = self.pose_to_tractorframe(next_goal[0],next_goal[1],0)

        r_goal = math.sqrt(self.goal_tractor[0]**2 + self.goal_tractor[1]**2)
        if(r_goal > 30):
            r_goal = 30
        th_goal = math.atan2(self.goal_tractor[1], self.goal_tractor[0])

        r_next_goal = math.sqrt(next_goal_tractor[0]**2 + next_goal_tractor[1]**2)
        if(r_next_goal > 30):
            r_next_goal = 30
        th_next_goal = math.atan2(next_goal_tractor[1], next_goal_tractor[0])

        dist_to_path = self.get_distance_to_path(cutoff=False)
        #print([r_goal, th_goal, r_next_goal, th_next_goal, dist_to_path, self.v, self.d])
        return np.array([r_goal/30, th_goal/math.pi, r_next_goal/30, th_next_goal/math.pi, dist_to_path/30, self.v/5, self.d/0.4]) #, self.th_dot/0.6712

    def get_point_on_path(self, wp_x=None, wp_y=None, prev_wp_x=None, prev_wp_y=None, cutoff=False):
        if(wp_x is None and wp_y is None and prev_wp_x is None and prev_wp_y is None):
            wp_x = self.goal_x
            wp_y = self.goal_y
            prev_wp_x = self.prev_goal_x
            prev_wp_y = self.prev_goal_y
        tractor_pos = np.array([self.x,self.y])
        start = np.array([prev_wp_x, prev_wp_y])
        goal = np.array([wp_x, wp_y])
        v = tractor_pos-start
        s = goal-start
        if(np.array_equal(start, goal)):
            cp = 0
        else:
            cp = v.dot(s)/s.dot(s)

        if(cutoff):
            if(cp < 1 and cp > 0):
                return cp*s + start
            elif(cp <= 0):
                return start
            elif(cp >= 1):
                return goal
        else:
            return cp*s + start   

    def get_distance_to_path(self, wp_x=None, wp_y=None, prev_wp_x=None, prev_wp_y=None, cutoff=True):
        if(wp_x is None and wp_y is None and prev_wp_x is None and prev_wp_y is None):
            wp_x = self.goal_x
            wp_y = self.goal_y
            prev_wp_x = self.prev_goal_x
            prev_wp_y = self.prev_goal_y

        v = self.get_point_on_path(wp_x, wp_y, prev_wp_x, prev_wp_y, cutoff=cutoff)
        tractor_pos = np.array([self.x,self.y])
        return np.linalg.norm(v-tractor_pos)

    def get_distance_to_next_path(self):
        next_wp = self.get_next_goal()
        next_wp_x = next_wp[0]
        next_wp_y = next_wp[1]
        d = self.get_distance_to_path(next_wp_x, next_wp_y, self.goal_x, self.goal_y)
        return d

    def reset(self):
        self.x = 0 # x coordinate
        self.y = 0 # y coordinate
        self.th = 0 # yaw heading (theta)
        self.x_dot = 0 # x velocity
        self.y_dot = 0 # y velocity
        self.th_dot = 0 # theta ang velocity
        self.v_command = 0 # commanded velocity
        self.d_command = 0 # commanded wheel angle
        self.v = 0 # actual velocity
        self.d = 0 # actual wheel angle
        self.goal_tractor = np.array([[0],[0]])
        self.prev_goal_tractor = np.array([[0],[0]])
        self.start_x = 0
        self.start_y = 0
        self.prev_goal_x = self.start_x
        self.prev_goal_y = self.start_y

    def set_vel(self, v):
        self.v_command = v

        #self.v_command = v
    
    def set_angle(self, d):
        self.d_command = d

    def wrap_angle(self, angle):
        wrapped_angle = (angle + math.pi) % (2 * math.pi) - math.pi
        return wrapped_angle

    def update_vel(self, dt):
        if self.v < self.v_command*math.cos(self.d):
            self.v = self.v + self.a_fwd * dt
            if(self.v > self.v_command*math.cos(self.d)):
                self.v = self.v_command*math.cos(self.d)
        elif self.v > self.v_command*math.cos(self.d):
            self.v = self.v - self.a_fwd * dt
            if(self.v < self.v_command*math.cos(self.d)):
                self.v = self.v_command*math.cos(self.d)
        else:
            self.v = self.v_command


    def update_angle(self, dt):
        if self.d < self.d_command:
            self.d = self.d + self.d_dot * dt
            if(self.d > self.d_command):
                self.d = self.d_command
        elif self.d > self.d_command:
            self.d = self.d - self.d_dot * dt
            if(self.d < self.d_command):
                self.d = self.d_command
        else:
            self.d = self.d_command

    def update(self, dt):
        self.update_vel(dt)
        self.update_angle(dt)
        self.update_goal()
        self.x_dot = self.v*math.cos(self.th)
        self.y_dot = self.v*math.sin(self.th)
        self.th_dot = self.v/self.l*math.tan(self.d)
        self.x = self.x + dt*self.x_dot
        self.y = self.y + dt*self.y_dot
        self.th = self.wrap_angle(self.th + dt*self.th_dot)

    def pose_to_tractorframe(self, x, y, angle):
        R = np.array([[math.cos(self.th), math.sin(self.th)],
                    [-math.sin(self.th), math.cos(self.th)]])
        t = np.array([[x - self.x],[y - self.y]])
        res = R@t
        
        x_tractor = res.item((0,0))
        y_tractor = res.item((1,0))
        yaw_tractor = angle - self.th

        return (x_tractor, y_tractor, yaw_tractor)
    
    def get_path_length(self, path):
        path_length = 0
        prev_wp = [0,0]
        for i,wp in enumerate(path):
            path_length += math.sqrt((prev_wp[0] - wp[0])**2 + (prev_wp[1] - wp[1])**2)
            prev_wp = wp
        return path_length

    def set_path(self, path):
        self.path = path
        self.path_goal_index = 0
        self.prev_goal_x = 0
        self.prev_goal_y = 0
        self.path_length = self.get_path_length(path)
        self.set_path_goal()
        #self.set_goal(self.path[self.path_goal_index][0],self.path[self.path_goal_index][1])

    def set_path_goal(self):
        self.set_goal(self.path[self.path_goal_index][0],self.path[self.path_goal_index][1])

    def update_goal(self):
        if(self.goal_x != None and self.goal_y != None):
            if((self.get_goal_dist() < self.goal_tol or self.get_distance_to_next_path() < self.get_distance_to_path())  and self.path_goal_index < len(self.path)-1):
                self.prev_goal_x = self.path[self.path_goal_index][0]
                self.prev_goal_y = self.path[self.path_goal_index][1]
                self.path_goal_index += 1            
                self.set_path_goal()

    def get_next_goal(self):
        if(self.path_goal_index < len(self.path)-1):
            return self.path[self.path_goal_index+1]
        else:
            return self.path[self.path_goal_index]

    def random_path(self, min_n:int, max_n:int, min_radius:float, max_radius:float, min_angle:float, max_angle:float, first_ang_min:float=None, first_ang_max:float=None):
        path = []
        n = random.randint(min_n, max_n)
        x = 0
        y = 0
        phi = 0
        for i in range(n):
            if(i == 0 and first_ang_max != None and first_ang_min != None):
                d_phi = (first_ang_max - first_ang_min)*random.random()+first_ang_min
                sign_angle = int(d_phi/abs(d_phi))
                if(sign_angle == self.prev_path_hdg):
                    d_phi = -d_phi
                self.prev_path_hdg = sign_angle
            else:
                d_phi = (max_angle - min_angle)*random.random()+min_angle
            phi += d_phi
            r = (max_radius - min_radius)*random.random()+min_radius
            x = r*math.cos(phi) + x
            y = r*math.sin(phi) + y
            path.append([x,y])

        self.set_path(path)    

    def project_vertices(self,vertices, axis):
        # Project vertices onto the axis and return the range of the projection
        projections = [np.dot(vertex, axis) for vertex in vertices]
        return min(projections), max(projections)
        
    def get_edges(self,vertices):
        edges = []
        for i in range(len(vertices)):
            edge = (vertices[i], vertices[(i + 1) % len(vertices)])
            edges.append(edge)
        return edges
        
    def get_normal(self,edge):
        # Calculate the normal vector of an edge
        p1, p2 = edge
        normal = np.array([-(p2[1] - p1[1]), p2[0] - p1[0]])
        # Ensure unit length
        normal = normal/np.linalg.norm(normal)
        return normal

    def overlap(self,poly1_np, poly2_np):
        poly1 = []
        poly2 = []
        for i in range(4):
            poly1.append((poly1_np[0,i],poly1_np[1,i]))
            poly2.append((poly2_np[0,i],poly2_np[1,i]))

        edges1 = self.get_edges(poly1)
        edges2 = self.get_edges(poly2)
        
        # Check overlap for each edge of both polygons
        for edge in edges1 + edges2:
            axis = self.get_normal(edge)
            min1, max1 = self.project_vertices(poly1, axis)
            min2, max2 = self.project_vertices(poly2, axis)
            
            # If there's no overlap on any axis, polygons do not intersect
            if max1 < min2 or max2 < min1:
                return False
        
        # If overlap exists on all axes, polygons intersect
        return True

    def check_collision(self, obstacles):
        collision = False
        for obstacle in obstacles:
            if(obstacle.w > 0 and obstacle.h > 0): # obstacle is rectangle
                if(self.overlap(self.footprint, obstacle.vertices_tractorframe)):
                    collision = True

            else:
                if(obstacle.x_tractorframe - obstacle.radius <= self.front and obstacle.x_tractorframe + obstacle.radius >= self.rear and obstacle.y_tractorframe - obstacle.radius <= self.left and obstacle.y_tractorframe + obstacle.radius >= self.right):
                    collision = True

        return collision


