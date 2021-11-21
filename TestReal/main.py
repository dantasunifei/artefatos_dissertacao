#!/usr/bin/env python3

import numpy as np
import os
import cv2
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import time

sys.path.insert(0, '/home/viscap/Desktop/Antonio/models/research')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import imutils
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
#from object_detection.utils import ops as utils_ops
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import glob
import asyncio
from mavsdk import start_mavlink
from mavsdk import connect as mavsdk_connect
from mavsdk import (
    Attitude,
    OffboardError,
    PositionNEDYaw,
    VelocityBodyYawspeed,
    VelocityNEDYaw,
)

async def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {
                output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(
                    tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(
                    tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(
                    tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [
                                           real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [
                                           real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(
                output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict
    
async def run_offb_ctrl_velocity_body(drone,x,y,z):
    """ Does Offboard control using velocity body co-ordinates. """
    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(x, y, z, 0.0))
    await asyncio.sleep(1)
    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
    await asyncio.sleep(0.1)

async def inicializar_drone(drone):
    print("-- Arming")
    await drone.action.arm()

    print("-- Setting initial setpoint")
    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))

    print("-- Starting offboard")
    try:
    	await drone.offboard.start()
    except OffboardError as error:
    	print(f"Starting offboard mode failed with error code: {error._result.result}")
    	print("-- Disarming")
    	await drone.action.disarm()

async def finalizar_drone(drone):
    print("-- Stopping offboard")
    try:
        await drone.offboard.stop()
    except OffboardError as error:
        print(f"Stopping offboard mode failed with error code: {error._result.result}")

async def print_battery(drone):
    async for battery in drone.telemetry.battery():
        print(f"Battery: {battery.remaining_percent}")


async def print_gps_info(drone):
    async for gps_info in drone.telemetry.gps_info():
        print(f"GPS info: {gps_info}")


async def print_in_air(drone):
    async for in_air in drone.telemetry.in_air():
        print(f"In air: {in_air}")


async def print_position(drone):
    async for position in drone.telemetry.position():
        print(position)


async def setup_tasks(drone):
    asyncio.ensure_future(print_battery(drone))
    asyncio.ensure_future(print_gps_info(drone))
    asyncio.ensure_future(print_in_air(drone))
    asyncio.ensure_future(print_position(drone))

async def acaoReal(arp, buffer, z, yaw):
    if buffer == 1:
      loop.run_until_complete(run_offb_ctrl_velocity_body(arp,5.0,0.0,z,yaw))
    else:
        if buffer == 2:
            loop.run_until_complete(run_offb_ctrl_velocity_body(arp,5.0,0.0,-z,-yaw))
        else:
            loop.run_until_complete(run_offb_ctrl_velocity_body(arp,10.0,0.0,0,0))


async def inferencia(arp, buffer, cap, detection_graph, category_index, z, yaw):
    ret, frame=cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = imutils.resize(gray, width=640)
    image_np = np.array(frame.astype(np.uint8))
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=8)

    acao = "0"
    try:
       if output_dict['detection_classes'] == "1":
           print("Executando Acao 1")
           acao = "1"
       else:
           if output_dict['detection_classes'] == "2":
               print("Executando Acao 2")
               acao = "2"
           else:
               print("Executando Acao 0")
    except:
        print("Executando Acao 0")
    acaoReal(arp, acao, z, yaw)
    #cv2.imshow('Imagem', image_np)
    buffer.append(acao)
    return buffer

    
async def CarregarItens(tipo):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile('/home/viscap/Desktop/Antonio/' + tipo + '/frozen_inference_graph.pb', 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    label_map = label_map_util.load_labelmap('/home/viscap/Desktop/Antonio/' + tipo + '/label_map_pbtxt_fname.pbtxt')
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=3, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return detection_graph, category_index

async def Buffer(arp1, arp2, buffer1, buffer2, vbcap,detection_graph,category_index, z, yaw):
    realizado = False
    while realizado == False:
        buffer1 = inferencia(arp1, buffer1, vbcap,detection_graph,category_index, z, yaw)
        if len(buffer1) > 0 and len(buffer1) >= len(buffer2):
            acaoReal(arp2, buffer1[len(buffer1)-1], yaw, z)
            realizado = True

buffer_cima = []
buffer_lateral = []


async def main():

    start_mavlink("serial:///dev/ttyUSB0:57600")
    arp_cima = mavsdk_connect(host="127.0.0.1")
    start_mavlink("serial:///dev/ttyUSB1:57600")
    arp_lateral = mavsdk_connect(host="127.0.0.1")

    vbcap=cv2.VideoCapture(2, cv2.CAP_V4L)
    vbcap2=cv2.VideoCapture(3, cv2.CAP_V4L)

    print(vbcap.isOpened())
    loop.run_until_complete(inicializar_drone(arp_cima))
    loop.run_until_complete(inicializar_drone(arp_lateral))
    
    detection_graph, category_index = CarregarItens("cima")
    detection_graph2, category_index2 = CarregarItens("lateral")


    while (vbcap.isOpened() and vbcap2.isOpened()):
        await setup_tasks(arp_cima)
        await setup_tasks(arp_lateral)
        
        Buffer(arp_cima, arp_lateral, buffer_cima, buffer_lateral, vbcap, detection_graph, category_index, 0, 30)
        time.sleep(1)
        Buffer(arp_lateral, arp_cima, buffer_lateral, buffer_cima, vbcap2, detection_graph2, category_index2, 30, 0)
        time.sleep(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    await loop.run_until_complete(finalizar_drone(drone))
    vbcap.release()
    vbcap2.release()
    cv2.destroyAllWindows() 

loop = asyncio.get_event_loop()
loop.run_until_complete(main())