# Project Write-Up
## Explaining Custom Layers

The process behind converting custom layers involves that if the Inference engine loads the layer from IR files that contains
layers are not in the list of known layers for the device, then the inference engine considers that layer to be unsupported 
and reports an error.

Handling custom layers in OpenVINO first of all requires to identify such layers. 
This can be done either empirically, while trying to convert a model into an IR, 
or programmatically, by looking at the layers supported by a specific device.

## Comparing Model Performance

The model is from http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz

    python /intel/openvino_2021.1.110/deployment_tools/model_optimizer/mo.py 
    --input_model /nd131-openvino-fundamentals-project-starter-master/model/TFmodel/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb
    --tensorflow_object_detection_api-pipeline_config /nd131-openvino-fundamentals-project-starter-master/model/TFmodel/ssd_mobilenet_v2_coco_2018_03_29/pipeline.config 
    --reverse_input_channels 
    --transformations_config /intel/openvino_2021.1.110/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json

My method to compare models before and after conversion to Intermediate Representations
were the orignial tensorflow model running on cpu and the optimised IR runing on cpu. 

The difference between model accuracy pre- and post-conversion was negligible, with the following results:

Tensorflow:


        [0.9925442]


OpenVINO IR on CPU:


        [0.9893007]

The size of the model pre- and post-conversion was 210 Mb and 34 Mb respectively, considering the folder with all the files. 

The running time for original tensorflow model was 159ms, for optimized IR was about 34ms

## Assess Model Use Cases

Some of the potential use cases of the people counter app are monitoring number of customes entering a retail store over 
one day. By compareing number of people entering of store and daily sales revenue, retail store manager could get an idea 
on how to improve sales performance.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows:
1. Different lighting condition could affact model performance becuase it will change the hue and brightness of the frames, likely will 
   lower the model performance.

2. Camera focal length/image size, the image size could influence the model inferencing speed if the image has been set with
   high resolution. The camera position is very important as it may affect the size of the object in the image.
