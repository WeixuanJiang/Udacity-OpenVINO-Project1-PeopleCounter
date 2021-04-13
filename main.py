"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt
from helper import extract_people
from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

# Frames needed for a person to disappear from screen
N_FRAMES_LIMIT = 8

# Frame rate of the video
FPS = 10

def draw_boxes(image,person,thres):
    h,w,_ = image.shape
    _,_,score,x1,y1,x2,y2 = person

    x1 = int(w * x1)
    x2 = int(w * x2)
    y1 = int(h * y1)
    y2 = int(h * y2)

    if score > thres:
        # draw blue box
        image = cv2.rectangle(image,(x1,y1),(x2,y2),(255,0,0),1)

    return image

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,default='CAM',
                        help="Path to image or video file, if wanto use your web camera input CAM")
    
    # Note - CPU extensions are moved to plugin since OpenVINO release 2020.1. 
    # The extensions are loaded automatically while     
    # loading the CPU plugin, hence 'add_extension' need not be used.
    #     parser.add_argument("-l", "--cpu_extension", required=False, type=str,
    #                         default=None,
    #                         help="MKLDNN (CPU)-targeted custom layers."
    #                              "Absolute path to a shared library with the"
    #                              "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST,MQTT_PORT,MQTT_KEEPALIVE_INTERVAL)

    return client


def most_frequent(List):
    return max(set(List),key = List.count())

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(args.model,args.device)

    ### TODO: Handle the input stream ###
    input_shape = infer_network.get_input_shape()
    instream = args.input
    if instream == 'CAM':
        instream = 0
    cap = cv2.VideoCapture(instream)
    cap.open(instream)
    ### TODO: Loop until stream is over ###
    # initialize variables
    total_people = 0
    number_of_ppl_last_frame = [0] * N_FRAMES_LIMIT
    current_count = 0
    previous_count = 0
    delta = 0
    duration = 0
    publish_duration = 0
    counting = False

    while cap.isOpened():
        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        ### TODO: Pre-process the image as needed ###
        frame = cv2.resize(frame,(input_shape[3],input_shape[2]))
        frame = frame.transpose((2,0,1))
        p_frame = frame.reshape(1,*frame.shape)
        ### TODO: Start asynchronous inference for specified request ###
        infer_network.exec_net(p_frame)
        ### TODO: Wait for the result ###
        if infer_network.wait() == 0:
            ### TODO: Get the results of the inference request ###
            output = infer_network.get_output()
            ### TODO: Extract any desired stats from the results ###
            people = extract_people(output)
            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            # check confidence and get number of people
            scores = people[:,2]
            people_count = people[scores>=prob_threshold].shape[0]

            # get the number of people in the last N frames
            number_of_ppl_last_frame.pop(0)
            number_of_ppl_last_frame.append(people_count)

            # get the most frequent value
            previous_count = current_count
            current_count = most_frequent(number_of_ppl_last_frame)
            delta = current_count - previous_count

            if delta > 0:
                # update number of people seen
                total_people += delta
                n_frames = 0
                counting = True
            elif delta < 0:
                # stop the frame and calcuate duration
                duration = n_frames/FPS
                publish_duration = True
                counting = False
            else:
                if counting:
                    n_frames += 1
            for person in people:
                frame = draw_boxes(frame,person,prob_threshold)

            if not client is None:
                client.publish('person',json.dumps({'count':current_count,
                                                    'total':total_people}))
                if publish_duration:
                    client.publish('people/duration',json.dumps({'duration':duration}))

            if cv2.waitKey(1) & 0xff==ord('q'):
                break

        ### TODO: Send the frame to the FFMPEG server ###
        try:
            sys.stdout.buffer.write(frame)
            sys.stdout.flush()
        except:
            print('FFMPEG failed',sys.stderr)
        
        ### TODO: Write an output image if `single_image_mode` ###
        cv2.imwrite('pred_output.jpg',frame)
        sys.stderr.close()
        client.disconnect()


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
