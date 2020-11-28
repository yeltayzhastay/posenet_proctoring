import cv2
import numpy as np
import asyncio
import websockets
import json
import time

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

async def res(websocket, path):
    i = 0
    
    while True:
        i+=1
        bufer = await websocket.recv()
        npfst = True
        while npfst:
            try:
                bufer = np.fromstring(bufer)
                image = cv2.imdecode(bufer, cv2.IMREAD_COLOR)
                WarningMessage = ""
                humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=4.0)
                humanCordinates = []
                image, humanCordinates = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
                
                if not humanCordinates:
                    WarningMessage = "No People"
                    await websocket.send(WarningMessage)
                    print(WarningMessage)
                else:
                    if (17 not in humanCordinates[0].keys()) or (16 not in humanCordinates[0].keys()):
                        WarningMessage = "Turn Head"
                        await websocket.send(WarningMessage)
                        print(WarningMessage)
                
                npfst = False
            except:
                bufer = bufer + b'\x00'

w, h = model_wh("432x368")
e = TfPoseEstimator(get_graph_path("mobilenet_thin"), target_size=(w, h), trt_bool=False)

start_server = websockets.serve(res, "192.168.12.16", 8089)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()