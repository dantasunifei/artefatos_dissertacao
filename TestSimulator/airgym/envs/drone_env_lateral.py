from typing import Any
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

        self.image_request = airsim.ImageRequest( 2 , airsim.ImageType.Scene, False, False)
        camera_pose = airsim.Pose(airsim.Vector3r(0, 0, 0.5), airsim.to_quaternion(0,0,-1.3))  #PRY in radians
        self.drone.simSetCameraPose(2, camera_pose)

    def __del__(self):
        self.drone.reset()

    def _setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)

        # camera de frente
        #self.drone.moveByVelocityZAsync(0,0,-2,0.5).join()
        #self.drone.moveToPositionAsync(-15, 20, 23, 10).join()
        #self.drone.rotateByYawRateAsync(-15, 10).join()
        #self.drone.moveByVelocityZAsync(0,0,0,0.1).join()

        self.drone.moveByVelocityZAsync(0,0,-2,1).join()
        self.drone.moveToPositionAsync(-15, 18, 23, 10).join()
        self.drone.moveByVelocityZAsync(0,0,0,0.1).join()
        self.drone.rotateByYawRateAsync(-13, 2).join()
        self.drone.moveByVelocityZAsync(0,0,0,0.1).join()
        
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
        
        #image_np = np.array(image.resize((1024, 768)))
        #image_end = (image_np.reshape(1, , , 1)).astype(np.float32)
        #image = image_end/ 255.0

        self.drone_state = self.drone.getMultirotorState()

        self.state["prev_position"] = self.state["position"]
        self.state["position"] = self.drone_state.kinematics_estimated.position
        self.state["velocity"] = self.drone_state.kinematics_estimated.linear_velocity

        collision = self.drone.simGetCollisionInfo().has_collided
        self.state["collision"] = collision

        return im_final

    def _do_action(self, action):
        quad_offset = self.interpret_action(action)
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
        self.drone.moveByVelocityAsync(vx = donre_vel_rota[0] + mvm[0],
              vy          = donre_vel_rota[1] + mvm[1],
              vz           = quad_offset[2],
              duration    = 1.5, 
              drivetrain  = typeDrivetrain,
              yaw_mode    = airsim.YawMode(is_rate = True,  
                                           yaw_or_rate = 0)).join()

    def distancia(self, A, B, P): 
        #Comprobamos que el punto no corresponda a los extremos del segmento. 
        #if all(A==P) or all(B==P): 
        #    return 0 
        ##Calculamos el angulo entre AB y AP, si es mayor de 90 grados retornamos la distancia enre A y P 
        #elif np.arccos(np.dot((P-A)/LA.norm(P-A), (B-A)/LA.norm(B-A))) > np.pi/2: 
        #    return LA.norm(P-A) 
        ##Calculamos el angulo entre AB y BP, si es mayor de 90 grados retornamos la distancia enre B y P. 
        #elif np.arccos(np.dot((P-B)/LA.norm(P-B), (A-B)/LA.norm(A-B))) > np.pi/2: 
        #    return LA.norm(P-B) #Como ambos angulos son menores o iguales a 90º sabemos que podemos hacer una proyección ortogonal del punto. 
        return LA.norm(np.cross(B-A, A-P))/LA.norm(B-A)


    def _compute_reward(self):
        thresh_dist = 5
        beta = 1
        z = -10
        pts = [
            np.array([0.5,0.5,20] ), 
            np.array([70.,55.,20] ), 
            np.array([0.1,105.,20]), 
            np.array([0.1,155.,20]), 
            np.array([-70.,255.,20]), 
            np.array([0.1,355.,20]), 
            np.array([0.1,455.,20])
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
        if self.state["position"].y_val+1 >= pts[self.level+1][1]:
            self.level += 1
            reward = 20 * (1 + self.level / len(pts))
            if self.level == 2 or self.level == 4 or self.level == 6:
                self.drone.rotateByYawRateAsync(-15, 4).join()
            elif self.level == 1:
                self.drone.rotateByYawRateAsync(15, 8).join()
            elif self.level == 3 or self.level == 5:
                self.drone.rotateByYawRateAsync(15, 4).join()

        elif self.state["collision"]:
            reward = -50
            dead = True
            self.level = 0
        else:
            p1 = np.asarray(pts[self.level])
            p2 = np.asarray(pts[self.level+1])
            p3 = np.asarray(quad_pt)
         
            tamanho_baloada = 10        
            pico_baloada = ((p2[2]-p1[2])/2)+tamanho_baloada
            pm = np.array([(p1[0]+(p2[0]-p1[0])/2.0),(p1[1]+(p2[1]-p1[1])/2.0),pico_baloada])

            if (p3[1] > pm[1]):
                dist = self.distancia(pm,p2,p3)
            else:
                dist = self.distancia(p1,pm,p3)

            if abs(dist) > 40. :
                dead = True
                self.level = 0
            reward = -1 * abs(dist)

        return reward, dead

    
    #def _compute_reward(self):
    #    thresh_dist = 5
    #    beta = 1
    #
    #    z = -10
    #    #pts = [
    #    #    np.array([-0.55265, -31.9786, -19.0225]),
    #    #    np.array([48.59735, -63.3286, -60.07256]),
    #    #    np.array([193.5974, -55.0786, -46.32256]),
    #    #    np.array([369.2474, 35.32137, -62.5725]),
    #    #    np.array([541.3474, 143.6714, -32.07256]),
    #    #]
    #    pts = [
    #        np.array([0.1,0.1,0.1] ), 
    #        np.array([70.,55.,0.1] ), 
    #        np.array([0.1,105.,0.1]), 
    #        np.array([0.1,155.,0.1]), 
    #        np.array([-70.,255.,0.]), 
    #        np.array([0.1,355.,0.1]), 
    #        np.array([0.1,455.,0.1])
    #    ] #drone de cima


    #    quad_pt = np.array(
    #        list(
    #            (
    #                self.state["position"].x_val,
    #                self.state["position"].y_val,
    #                self.state["position"].z_val,
    #            )
    #        )
    #    )

    #    if self.state["collision"]:
    #        reward = -100
    #    else:
    #        dist = 10000000
    #        for i in range(0, len(pts) - 1):
    #            dist = min(
    #                dist,
    #                np.linalg.norm(np.cross((quad_pt - pts[i]), (quad_pt - pts[i + 1])))
    #                / np.linalg.norm(pts[i] - pts[i + 1]),
    #            )

    #        if dist > thresh_dist:
    #            reward = -10
    #        else:
    #            reward_dist = math.exp(-beta * dist) - 0.5
    #            #reward_speed = (
    #            #    np.linalg.norm(
    #            #        [
    #            #            self.state["velocity"].x_val,
    #            #            self.state["velocity"].y_val,
    #            #            self.state["velocity"].z_val,
    #            #        ]
    #            #    )
    #            #    - 0.5
    #            #)
    #            #reward = reward_dist + reward_speed
    #            reward = reward_dist
    #    done = 0
    #    if reward <= -10:
    #        done = 1

    #    return reward, done

    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()
        reward, done = self._compute_reward()

        return obs, reward, done, self.state

    def reset(self):
        self._setup_flight()
        return self._get_obs()

    def interpret_action(self, action):
        #if action == 0: #frente
        #    quad_offset = (0, 2.5, 0, 0)
        #elif action == 1: #+z
        #    quad_offset = (0, 2.5, 1, 0)    
        #elif action == 2: #-z
        #    quad_offset = (0, 2.5, -1, 0)  
        if action == 0: #frente
            quad_offset = (2.5, 0, 0, 0)
        elif action == 1: #+z
            quad_offset = (1, 0, 1, 0)   
        elif action == 2: #-z
            quad_offset = (1, 0, -1, 0)   
        return quad_offset