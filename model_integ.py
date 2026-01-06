# general package imports
import numpy as np
import torch
from easydict import EasyDict as edict

import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# define models and req functions path
from pretrained.darknet.models.darknet2pytorch import Darknet as darknet
from pretrained.darknet.utils.evaluation_utils import post_processing_v2
from pretrained.resnet.models import fpn_resnet
from pretrained.resnet.utils.evaluation_utils import decode, post_processing
from pretrained.resnet.utils.torch_utils import _sigmoid


def integ_model_config(model_name='darknet', configs=None):

    """
    This function initializes and sets the configuration for different models based on the model name provided.
    It supports 'darknet(complexyolov4)' and 'fpn_resnet' models. If no configuration is passed, it creates a new one.

    Parameters:
    model_name (str): Name of the model to configure ('darknet' or 'fpn_resnet').
    configs (edict, optional): Existing configuration dictionary. Defaults to None.

    Returns:
    edict: Updated configuration dictionary.
    """

    if configs==None:
        configs = edict()  


    path_current = os.path.dirname(os.path.realpath(__file__))
    original_path = configs.model_path = os.path.abspath(os.path.join(path_current, os.pardir))    
    

    if model_name == "darknet":
        configs.model_path = os.path.join(original_path, 'pretrained', 'darknet')
        configs.pretrained_filename = os.path.join(configs.model_path, 'pretrainedModel', 'complex_yolov4_mse_loss.pth')
        configs.arch = 'darknet'
        configs.batch_size = 4
        configs.cfgfile = os.path.join(configs.model_path, 'config', 'complex_yolov4.cfg')
        configs.conf_thresh = 0.1
        configs.distributed = False
        configs.img_size = 608
        configs.nms_thresh = 0.4
        configs.num_samples = None
        configs.num_workers = 4
        configs.pin_memory = True
        configs.use_giou_loss = False

    elif model_name == 'fpn_resnet':
        configs.arch = 'fpn_resnet'
        configs.conf_thresh = 0.5
        configs.saved_fn = 'fpn_resnet'
        configs.pretrained_path = 'pretrained/resnet/pretrainedModel/fpn_resnet_18_epoch_300.pth'
        configs.k = 50
        configs.no_cuda = True
        configs.gpu_idx = 0
        configs.batch_size = 1
        configs.num_samples = None
        configs.num_workers = 1
        configs.peak_thresh = 0.2
        configs.save_test_output = False
        configs.output_format = 'image'
        configs.output_video_fn = 'out_fpn_resnet'
        configs.output_width = 608
        configs.pin_memory = True
        configs.distributed = False
        configs.input_size = (608, 608)
        configs.hm_size = (152, 152)
        configs.down_ratio = 4
        configs.max_objects = 50
        configs.imagenet_pretrained = False
        configs.head_conv = 64
        configs.num_classes = 3
        configs.num_center_offset = 2
        configs.num_z = 1
        configs.num_dim = 3
        configs.num_direction = 2
        configs.heads = {'hm_cen': configs.num_classes, 'cen_offset': configs.num_center_offset, 
                         'direction': configs.num_direction, 'z_coor': configs.num_z,'dim': configs.num_dim}
        configs.num_input_features = 4

        configs.model_path = os.path.join(original_path, 'pretrained', 'resnet')
        configs.pretrained_filename = os.path.join(configs.model_path, 'pretrainedModel', 'fpn_resnet_18_epoch_300.pth')
        
        
        
         #////////////////////////////////////////

    else:
        raise ValueError("Error: Invalid model name")

    # GPU vs. CPU
    configs.no_cuda = True 
    configs.gpu_idx = 0  
    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))

    return configs










def initialize_detection_configs(model_name='fpn_resnet', configs=None):
    """
    Initialize and load all object-detection parameters into an edict.

    Parameters:
    model_name (str): Name of the model to configure ('darknet' or 'fpn_resnet').
    configs (edict, optional): Existing parameter dictionary. Defaults to None.

    Returns:
    edict: Updated parameter dictionary with detection and model parameters.
    """

    if configs == None:
        configs = edict()    

    configs.detection_range_x = [0, 50]  # Detection range in meters
    configs.detection_range_y = [-25, 25]
    configs.detection_range_z = [-1, 3]
    configs.reflected_intensity_range = [0, 1.0]  # rflected lidar intensity
    
    configs.bev_image_width = 608  # Pixel resolution of BEV image
    configs.bev_image_height = 608
    
    configs.minimum_iou = 0.5

    configs = integ_model_config(model_name, configs)

    configs.result_image_width = 608  # Width of the result imag
    configs.object_colors = [[0, 255, 255], [0, 0, 255], [255, 0, 0]]  # 'Pedestrian': 0, 'Car': 1, 'Cyclist': 2

    return configs


def build_model(configs):
    """
    Create a model according to the selected model type.

    Parameters:
    configs (edict): Configuration dictionary with model parameters.

    Returns:
    model: Initialized and loaded model ready for evaluation.
    """

    assert os.path.isfile(configs.pretrained_filename), "No file at {}".format(configs.pretrained_filename)

    if (configs.arch == 'darknet') and (configs.cfgfile is not None):
        print('using darknet')
        model = darknet(cfgfile=configs.cfgfile, use_giou_loss=configs.use_giou_loss)    
    
    elif (configs.arch == 'fpn_resnet'):
        print('using ResNet arch with feature pyramid')
        num_layers = 18
        model = fpn_resnet.get_pose_net(num_layers=num_layers, heads=configs.heads, 
                                        head_conv=configs.head_conv, 
                                        imagenet_pretrained=configs.imagenet_pretrained)
    else:
        assert False, ' model backbone is undifned'

    model.load_state_dict(torch.load(configs.pretrained_filename, map_location='cpu'))
    print(' weights loaded from {}\n'.format(configs.pretrained_filename))

    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_index))
    model = model.to(device=configs.device)  # Load model to either CPU or GPU
    model.eval()          

    return model








def detect_trained_objects(bev_maps, detection_model, configs):
    """
    Detect trained objects in birds-eye view (BEV) maps using the specified model and configuration parameters.

    Parameters:
    bev_maps (tensor): Input BEV maps.
    detection_model (model): Trained model for object detection.
    configs (edict): Configuration parameters for object detection.

    Returns:
    list: List of detected objects with their attributes.
    """

    with torch.no_grad():  
        model_outputs = detection_model(bev_maps)

        if 'darknet' in configs.arch:
            processed_outputs = post_processing_v2(model_outputs, conf_thresh=configs.conf_thresh, nms_thresh=configs.nms_thresh) 
            detections = []
            for i in range(len(processed_outputs)):
                if processed_outputs[i] is None:
                    continue
                detection = processed_outputs[i]
                for obj in detection:
                    x, y, w, l, im, re, _, _, _ = obj
                    yaw = np.arctan2(im, re)
                    detections.append([1, x, y, 0.0, 1.50, w, l, yaw])
        elif 'fpn_resnet' in configs.arch:
            model_outputs['hm_cen'] = _sigmoid(model_outputs['hm_cen'])
            model_outputs['cen_offset'] = _sigmoid(model_outputs['cen_offset'])
            detections = decode(model_outputs['hm_cen'], model_outputs['cen_offset'], model_outputs['direction'], model_outputs['z_coor'],
                                model_outputs['dim'], K=40)
            detections = detections.cpu().numpy().astype(np.float32)
            detections = post_processing(detections, configs)
            detections = detections[0][1]
           
    detected_objects = []
    for obj in detections:
        class_id, bev_x, bev_y, z, h, bev_w, bev_l, yaw = obj
        x = bev_y / configs.bev_image_height * (configs.detection_range_x[1] - configs.detection_range_x[0])
        y = bev_x / configs.bev_image_width * (configs.detection_range_y[1] - configs.detection_range_y[0]) - (configs.detection_range_y[1] - configs.detection_range_y[0]) / 2.0 
        w = bev_w / configs.bev_image_width * (configs.detection_range_y[1] - configs.detection_range_y[0]) 
        l = bev_l / configs.bev_image_height * (configs.detection_range_x[1] - configs.detection_range_x[0])
        
        if ((x >= configs.detection_range_x[0]) and (x <= configs.detection_range_x[1])
         and (y >= configs.detection_range_y[0]) and (y <= configs.detection_range_y[1])
         and (z >= configs.detection_range_z[0]) and (z <= configs.detection_range_z[1])):
             detected_objects.append([1, x, y, z, h, w, l, yaw])
    
    print("Number of detections:", len(detected_objects))
    return detected_objects
   