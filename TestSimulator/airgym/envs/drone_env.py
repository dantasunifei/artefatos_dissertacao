import setup_path
import airsim
import numpy as np
import math
import time
from argparse import ArgumentParser
from pyquaternion import Quaternion
from PIL import Image
from numpy import linalg as LA
from scipy.spatial import distance

import gym
from gym import spaces
from airgym.envs.airsim_env import AirSimEnv


class AirSimDroneEnv(AirSimEnv):
    def __init__(self, ip_address, step_length, image_shape):
        super().__init__(image_shape)
        self.step_length = step_length
        self.image_shape = image_shape
        self.level = 0
        self.state = {
            "position": np.zeros(3),
            "collision": False,
            "prev_position": np.zeros(3),
        }

        self.drone = airsim.MultirotorClient(ip=ip_address)
        self.action_space = spaces.Discrete(3)
        self._setup_flight()

        self.image_request = airsim.ImageRequest(
            3, airsim.ImageType.Scene, False, False
        )

    def __del__(self):
        self.drone.reset()

    def _setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)
        self.drone.moveByVelocityAsync(0, 0, -10, 5).join()
        self.drone.moveByVelocityAsync(2.5, 2.5, -1, 5).join()

    def transform_obs(self, responses):
        img1d = np.array(responses[0].image_data_float, dtype=np.float)
        img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width))

        from PIL import Image

        image = Image.fromarray(img2d)
        im_final = np.array(image.resize((84, 84)).convert("L"))

        return im_final.reshape([84, 84, 1])

    def _get_obs(self):
        responses = self.drone.simGetImages([self.image_request])
        response = responses[0]
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        img2d = img1d.reshape(response.height, response.width, 3)
        image = Image.fromarray(img2d)  
        image = image.convert('L')
        image.save("view_obs.png","PNG") #write to png 
        im_final = np.array(image)
        im_final = im_final.reshape([768, 1024, 1])

        self.drone_state = self.drone.getMultirotorState()

        self.state["prev_position"] = self.state["position"]
        self.state["position"] = self.drone_state.kinematics_estimated.position
        self.state["velocity"] = self.drone_state.kinematics_estimated.linear_velocity

        collision = self.drone.simGetCollisionInfo().has_collided
        self.state["collision"] = collision

        return im_final

    def _do_action(self, action):
        quad_offset = self.interpret_action(action)
        
        if quad_offset[3] != 0:
            self.drone.rotateByYawRateAsync(quad_offset[3], 2).join()
        typeDrivetrain = airsim.DrivetrainType.MaxDegreeOfFreedom
        q             = self.drone.simGetVehiclePose().orientation 
        my_quaternion = Quaternion(w_val = q.w_val,
                                x_val = q.x_val,
                                y_val = q.y_val,
                                z_val = q.z_val)
        action = [quad_offset[0], quad_offset[1], quad_offset[2]]
        mvm           = my_quaternion.rotate(action)
        velocities = self.drone.getMultirotorState().kinematics_estimated.angular_velocity
        donre_vel_rota =[velocities.x_val , velocities.y_val]
        # Perform the movement
        self.drone.moveByVelocityZAsync(vx = donre_vel_rota[0] + mvm[0],
              vy          = donre_vel_rota[1] + mvm[1],
              z           = 0.1,
              duration    = 5, 
              drivetrain  = typeDrivetrain,
              yaw_mode    = airsim.YawMode(is_rate = True, yaw_or_rate = 0)).join()

    def distancia(self, A, B, P): 
        return LA.norm(np.cross(B-A, A-P))/LA.norm(B-A)


    def _compute_reward(self):
        thresh_dist = 5
        beta = 1
        z = -10
        pts = [
            np.array([0.1,0.1,0.1] ), 
            np.array([70.,55.,0.1] ), 
            np.array([0.1,105.,0.1]), 
            np.array([0.1,155.,0.1]), 
            np.array([-70.,255.,0.]), 
            np.array([0.1,355.,0.1]), 
            np.array([0.1,455.,0.1])
        ] #drone de cima

        quad_pt = np.array(
            list(
                (
                    self.state["position"].x_val,
                    self.state["position"].y_val,
                    self.state["position"].z_val,
                )
            )
        )

        dead = False
        if self.state["position"].y_val >= pts[self.level+1][1]:
            self.level += 1
            reward = 20 * (1 + self.level / len(pts))
        elif self.state["collision"]:
            reward = -100
            dead = True
            self.level = 0
        else:
            p1 = np.asarray(pts[self.level])
            p2 = np.asarray(pts[self.level+1])
            p3 = np.asarray(quad_pt)
            
            #dist = s/elf.distancia(p1,p2,p3)
            dist = np.linalg.norm(np.cross((quad_pt - p1), (quad_pt - p2))) / np.linalg.norm(p1 - p2)
            d = (0.9 * dist + 0.1 * distance.euclidean(p3, p2))
            if abs(dist) > 20. :
                reward = -50
                dead = True
                self.level = 0
            else:
                reward = -1 * abs(d)
        return reward, dead

    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()
        reward, done = self._compute_reward()

        return obs, reward, done, self.state

    def reset(self):
        self._setup_flight()
        return self._get_obs()

    def interpret_action(self, action):
        if action == 0: #frente
            quad_offset = (2, 0, 0, 0)
        elif action == 1: #+yaw
            quad_offset = (2, 0, 0, 10.)    
        elif action == 2: #-yaw
            quad_offset = (2, 0, 0, -10.)   
        return quad_offset