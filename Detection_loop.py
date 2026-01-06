#libraries defining
import os
import sys
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import copy
import zlib
from easydict import EasyDict as edict
from PIL import Image
import io
import sys
import os
#import open3d as o3d
import math
import numpy as np
import zlib
from easydict import EasyDict as edict
from sklearn.preprocessing import RobustScaler, StandardScaler, QuantileTransformer
sys.path.append(os.getcwd())




# simple waymo reader cloned from  https://github.com/gdlg/simple-waymo-open-dataset-reader 
from simple_waymo_open_dataset_reader import WaymoDataFileReader, dataset_pb2, label_pb2
from simple_waymo_open_dataset_reader import utils as waymo_utils


# improt main func folder that containes all files funcs impllemnted

from main_funcs.load_save_funcs import loadObject,saveObject,prepare_execution_list
import main_funcs.pcl as pcl
import main_funcs.model_integ as model
import main_funcs.birdeyeview as  bev
import main_funcs.obj_labels as label
import main_funcs.visualization as vis
import main_funcs.evaluation as eval


# waymo tf data used and assigned to df_file varibale where each tf is around 1 gb and has 200 frames
#df_file = 'training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord' # Sequence 1
#df_file = 'training_segment-10072231702153043603_5725_000_5745_000_with_camera_labels.tfrecord' # Sequence 2
df_file = 'training_segment-10963653239323173269_1924_000_1944_000_with_camera_labels.tfrecord'  # Sequence 3

rangeFrames = [50,51]  # used to plot the range number of the frames 


stop_plot_timer = 0  # variable assigned to 0 in which every displayed frame will be plotted after pressing any key

#  loading the data prep
data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Waymo_Dataset', df_file)  
output_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'output' )
datafile = WaymoDataFileReader(data_path)
iterator_datafile = iter(datafile)  # iterate over frames in data_path instance


####################

# # detection
det_config = model.initialize_detection_configs(model_name='darknet') # or 'darknet'fpn_resnet
model_config = model.build_model(det_config)
det_config.use_labels_as_objects =True # False 

## list for choosing the funcs to be excuated
detection_tasks  = ['generate_bev_map','detect_trained_objects','validate_labels','evaluate_detection_performance'] #['bev_from_pcl','obj_detected', 'validate_object_labels', 'evaluate_detection_performance' ] ##['rangeimgTopcl','convert_pcl_to_bev'] # options are 'bev_from_pcl', 'obj_detected', 'validate_object_labels', 'evaluate_detection_performance'; options not in the list will be loaded from file
data_tasks = ['extract_point_cloud_from_frame','img_load']   #['rangeimgTopcl'] #, 'img_load']
visualization_tasks = ['get_detection_performance_frames']   # ['show_detection_performance']#['view_pointcloud'] #['view_lidar_rangeimg'] ['view_pointcloud'] #'view_lidar_rangeimg'# options are 'show_range_image', 'show_bev', 'show_pcl', 'show_labels_in_image', 'show_objects_and_labels_in_bev', 'show_objects_in_bev_labels_in_camera', 'show_detection_performance'
task_list  = prepare_execution_list(detection_tasks,visualization_tasks,data_tasks)

##################

frame_counter = 0
total_labels = []
eval_detection = []



while True:
    try:
        #next frame
        frame = next(iterator_datafile)
        if frame_counter < rangeFrames[0]:
            frame_counter = frame_counter + 1
            continue
        elif frame_counter > rangeFrames[1]:
            print('the number of frames reached the limit ')
            break
        
        print('#########################################')
        print('running frame number is :' + str(frame_counter))


        ## get calib data from lidar and camera
        lidar_name = dataset_pb2.LaserName.TOP
        camera_name = dataset_pb2.CameraName.FRONT
        lidar_pcl = pcl.extract_point_cloud_from_frame(frame, lidar_name)
        lidar_calibration = waymo_utils.get(frame.context.laser_calibrations, lidar_name)        
        camera_calibration = waymo_utils.get(frame.context.camera_calibrations, camera_name)
       
        if 'img_load' in task_list:
            image = vis.get_front_camera_image(frame) 


        ## convert the range image to point cloud from lidar
        if 'extract_point_cloud_from_frame' in task_list:
            print('converting the range image to point cloud from lidar')
            lidar_pcl = pcl.extract_point_cloud_from_frame(frame, lidar_name)
        
        print(lidar_pcl.shape)

   
#/////////////////////////////////////////////////////////////////
#////////BEV/////////////////////



        ## converting point cloud to birds eye view from lidar
        if 'generate_bev_map' in task_list:
            print('Converting point cloud to birds eye view from lidar')
            bev_lidar = bev.generate_bev_map(lidar_pcl, det_config)
            

     #////////////////////////////////////////////////////////////////////////////////////
     #/////////DETECTION////////////////////
      
      # DETEction in 3d using the pretrained models
        
        if (det_config.use_labels_as_objects==True):
            print('Applying groundtruth labels as objects.')
            vehicale_detection = label.convert_labels_to_obj(frame.laser_labels, det_config)
        else:
            if 'obj_detected' in task_list:
                print('detecting objects in lidar pointcloud')   
                vehicale_detection = model.obj_detected(bev_lidar, model_config, det_config)
   
        ## Check object labels
        if 'validate_labels' in task_list:
            print("Performing validation of object labels.")
            correct_labels = label.validate_labels(frame.laser_labels, lidar_pcl, det_config, 0 if det_config.use_labels_as_objects==True else 10)
        else:
            #  dfault correct_labels with all set to True
            correct_labels = [True] * len(frame.laser_labels)
           
 
        ## object detection performance
        if 'evaluate_detection_performance' in task_list:
            print('Evaluating detection performance.')
            det_performance = eval.evaluate_detection_performance(vehicale_detection, frame.laser_labels, correct_labels, det_config.minimum_iou)   
            eval_detection.append(det_performance)






   #//////////////////////////////////////////////////////////
   #///////////////////VISUAL///////////////////////


        # obj detect visualizations
        if 'range_image' in task_list:
            range_img = pcl.view_lidar_rangeimg(frame, lidar_name)
            range_img = range_img.astype(np.uint8)
            cv2.imshow('range_image', range_img)
            cv2.waitKey(stop_plot_timer)

        if 'view_pointcloud' in task_list:
            pcl.view_pointcloud(lidar_pcl)

                 
        if 'display_bev' in task_list:
            vis.display_bev(bev_lidar, det_config)  
            cv2.waitKey(stop_plot_timer)          

        if 'labels_to_camera_plot' in task_list:
            img_labels = vis.labels_to_camera_plot(camera_calibration, image, frame.laser_labels, correct_labels, 0.5)
            cv2.imshow('img_labels', img_labels)
            cv2.waitKey(stop_plot_timer)

        if 'visualize_objects_and_labels_in_bev' in task_list:
            vis.visualize_objects_and_labels_in_bev(vehicale_detection, frame.laser_labels, bev_lidar, det_config)
            cv2.waitKey(stop_plot_timer)         

        if 'plot_obj_bevLabels_camera' in task_list:
            vis.plot_obj_bevLabels_camera(vehicale_detection, bev_lidar, image, frame.laser_labels, correct_labels, camera_calibration, det_config)
            cv2.waitKey(stop_plot_timer)
            # cv2.imwrite(f'output_{frame_counter}.png', bev_lidar)  # Save the BEV image to a file
            # cv2.imwrite(f'output_labels_{frame_counter}.png', image)               


        frame_counter = frame_counter + 1    

    except StopIteration:
        # if raies stop loop
        print("StopIteration is raised\n")
        break


if 'get_detection_performance_frames' in task_list:
    eval.get_detection_performance_frames(eval_detection)









