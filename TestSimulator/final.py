#DL
from numpy.core.defchararray import index
import tensorflow as tf
from object_detection.utils import label_map_util
from matplotlib import pyplot as plt
from PIL import Image
import dl

#RL
import setup_path
import gym
import airgym
import time
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

#DB
import db
import cv2
TAMANHO = 10.

#Define Ambiente
env = DummyVecEnv(
    [
        lambda: Monitor(
            gym.make(
                "airgym:airsim-drone-sample-v0",
                ip_address="127.0.0.1",
                step_length=0.25,
                image_shape=(768, 1024, 1),
            )
        )
    ]
)
env = VecTransposeImage(env)

#define variáveis globais
modo =  "cima"
tecnica = ["DB","DL","RL"]
tempo_medio = [0,0,0]
soma_media = [0,0,0]
soma_distancia = [0,0,0]

def main():
    #carregar redes neurais
    model = A2C.load(modo + "\\best_model", env=env)
    graph = tf.Graph()
    index = dl.carregarGraph(graph, f'{modo}\\frozen_inference_graph.pb','cima\\label_map_pbtxt_fname.pbtxt')

    #início do ambiente em número limitado de testes
    obs = env.reset()
    i = 0
    while i < TAMANHO:
        contador = 0
        distancia = 0
        indice_tecnica = 0
        tempo = 0

        #todas técnicas
        while indice_tecnica < len(tecnica):
            image = obs[0,0,:,:]
            image_path = Image.fromarray(image)  
            image_path.save("view.png","PNG") 
            image_np = cv2.imread("view.png")
            image = np.uint8(image_path)

            try:
                if indice_tecnica == 0:
                    inicio = time.time()
                    a = db.findAngulo(image)
                    fim = time.time()
                    action = [ db.defineAcao(a) ]
                elif indice_tecnica == 1:
                    inicio = time.time()
                    output_dict = dl.run_inference_for_single_image(image_np, graph)
                    fim = time.time()
                    action = [ output_dict['detection_classes'][0] - 1 ]
                else:
                    inicio = time.time()
                    action, _ = model.predict(obs, deterministic=True)
                    fim = time.time()
            except:
                action = 0

            #obtém indicadores
            obs, rewards, dones, info = env.step(action)
            distancia = distancia + rewards
            contador = contador + 1
            tempo = tempo +  (fim-inicio) 

            #processo finalizado
            if dones:
                soma_distancia [indice_tecnica] += distancia
                soma_media [indice_tecnica] += contador
                tempo_medio[indice_tecnica] += tempo
                obs = env.reset()
                print(f"Técnica: {tecnica[indice_tecnica]} Distância: {distancia} Contador: {contador}")
                contador = 0
                distancia = 0
                tempo = 0
                indice_tecnica = indice_tecnica + 1
        i+=1

    #apresenta resultados
    i = 0
    for i in range(len(tecnica)):
        print(f"Técnica: {tecnica[i]} Distância Média: {soma_distancia[i]/TAMANHO} Contador Médio: {soma_media[i]/TAMANHO} Tempo Médio: {tempo_medio[i]/TAMANHO}")

#Chama programa principal
main()