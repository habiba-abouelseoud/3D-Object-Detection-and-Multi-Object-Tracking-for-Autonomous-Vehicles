#@ this one will be related to converting to bev 
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import torch
import zlib
import open3d as o3d

from simple_waymo_open_dataset_reader import utils as waymo_utils
from simple_waymo_open_dataset_reader import WaymoDataFileReader, dataset_pb2, label_pb2
# know i want to convert pcl to bev to construct bev map wich from there i can start detec
from easydict import EasyDict as edict


import matplotlib.cm
cmap = matplotlib.cm.get_cmap("viridis")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np



##################
# BIRDS-EYE VIEW

def render_detections_on_bev(bev_image, detected_objects, configs, object_colors=[]):
    """
    Projects detected bounding boxes onto a birds-eye view (BEV) map. Converts detected object 
    coordinates from metric units to pixel coordinates and draws bounding boxes and front indicators 
    on the BEV map.
    """
    for detection in detected_objects:
        obj_id, obj_x, obj_y, obj_z, obj_height, obj_width, obj_length, obj_yaw = detection

        x_pixel = (obj_y - configs.detection_range_y[0]) / (configs.detection_range_y[1] - configs.detection_range_y[0]) * configs.bev_image_width
        y_pixel = (obj_x - configs.detection_range_x[0]) / (configs.detection_range_x[1] - configs.detection_range_x[0]) * configs.bev_image_height
        z_adjusted = obj_z - configs.detection_range_z[0]
        width_pixel = obj_width / (configs.detection_range_y[1] - configs.detection_range_y[0]) * configs.bev_image_width
        length_pixel = obj_length / (configs.detection_range_x[1] - configs.detection_range_x[0]) * configs.bev_image_height
        yaw_angle = -obj_yaw

        if not object_colors:
            color = configs.object_colors[int(obj_id)]
        
        corners = np.zeros((4, 2), dtype=np.float32)
        cos_yaw = np.cos(yaw_angle)
        sin_yaw = np.sin(yaw_angle)
        corners[0, 0] = x_pixel - width_pixel / 2 * cos_yaw - length_pixel / 2 * sin_yaw
        corners[0, 1] = y_pixel - width_pixel / 2 * sin_yaw + length_pixel / 2 * cos_yaw
        corners[1, 0] = x_pixel - width_pixel / 2 * cos_yaw + length_pixel / 2 * sin_yaw
        corners[1, 1] = y_pixel - width_pixel / 2 * sin_yaw - length_pixel / 2 * cos_yaw
        corners[2, 0] = x_pixel + width_pixel / 2 * cos_yaw + length_pixel / 2 * sin_yaw
        corners[2, 1] = y_pixel + width_pixel / 2 * sin_yaw - length_pixel / 2 * cos_yaw
        corners[3, 0] = x_pixel + width_pixel / 2 * cos_yaw - length_pixel / 2 * sin_yaw
        corners[3, 1] = y_pixel + width_pixel / 2 * sin_yaw + length_pixel / 2 * cos_yaw

        corners_int = corners.reshape(-1, 1, 2).astype(int)
        cv2.polylines(bev_image, [corners_int], True, color, 2)

        front_line_start = (int(corners[0, 0]), int(corners[0, 1]))
        front_line_end = (int(corners[3, 0]), int(corners[3, 1]))
        cv2.line(bev_image, front_line_start, front_line_end, (255, 255, 0), 2)











import matplotlib.pyplot as plt


"""
This function generates a bird's-eye view (BEV) representation of LiDAR data. It filters out points based on specified
limits, transforms the coordinates to BEV map coordinates, and creates intensity, height, and density layers for the 
BEV map. tthe final output is a tensor of the BEV map.
"""



def generate_bev_map(lidar_data, configs):
    valid_points = np.where((lidar_data[:, 0] >= configs.detection_range_x[0]) & (lidar_data[:, 0] <= configs.detection_range_x[1]) &
                            (lidar_data[:, 1] >= configs.detection_range_y[0]) & (lidar_data[:, 1] <= configs.detection_range_y[1]) &
                            (lidar_data[:, 2] >= configs.detection_range_z[0]) & (lidar_data[:, 2] <= configs.detection_range_z[1]))
    lidar_data = lidar_data[valid_points]
    lidar_data[:, 2] -= configs.detection_range_z[0]
    
    bev_resolution = (configs.detection_range_x[1] - configs.detection_range_x[0]) / configs.bev_image_height
    lidar_data_copy = np.copy(lidar_data)
    lidar_data_copy[:, 0] = np.int_(np.floor(lidar_data_copy[:, 0] / bev_resolution))
    lidar_data_copy[:, 1] = np.int_(np.floor(lidar_data_copy[:, 1] / bev_resolution)) + ((configs.bev_image_width + 1) / 2)
    lidar_data_copy[:, 1] = np.abs(lidar_data_copy[:, 1])
    
    intensity_layer = np.zeros((configs.bev_image_height, configs.bev_image_width))
    lidar_data_copy[lidar_data_copy[:, 3] > 1.0, 3] = 1.0
    sorted_indices = np.lexsort((-lidar_data_copy[:, 2], lidar_data_copy[:, 1], lidar_data_copy[:, 0]))
    lidar_data_top = lidar_data_copy[sorted_indices]
    unique_points, indices, counts = np.unique(lidar_data_copy[:, 0:2], axis=0, return_index=True, return_counts=True)
    lidar_data_top = lidar_data_copy[indices]
    intensity_layer[np.int_(lidar_data_top[:, 0]), np.int_(lidar_data_top[:, 1])] = lidar_data_top[:, 3] / (np.amax(lidar_data_top[:, 3]) - np.amin(lidar_data_top[:, 3]))
    
    img_intensity = intensity_layer * 256
    img_intensity = img_intensity.astype(np.uint8)
    #cv2.imshow('intensity_channel', img_intensity)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # plt.figure(figsize=(10, 10))
    # plt.imshow(img_intensity, cmap='hot', interpolation='nearest')
    # plt.title('density channel Heatmap')
    # plt.colorbar()
    # plt.show()
    
    height_layer = np.zeros((configs.bev_image_height, configs.bev_image_width))
    height_layer[np.int_(lidar_data_top[:, 0]), np.int_(lidar_data_top[:, 1])] = lidar_data_top[:, 2] / float(np.abs(configs.detection_range_z[1] - configs.detection_range_z[0]))
    
    img_height = height_layer * 256
    img_height = img_height.astype(np.uint8)
    cv2.imshow('height_channel', height_layer)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    plt.figure(figsize=(10, 10))
    plt.imshow(height_layer, cmap='hot', interpolation='nearest')
    plt.title('Height Map Heatmap')
    plt.colorbar()
    plt.show()
    
    density_layer = np.zeros((configs.bev_image_height + 1, configs.bev_image_width + 1))
    _, _, counts = np.unique(lidar_data_copy[:, 0:2], axis=0, return_index=True, return_counts=True)
    normalized_counts = np.minimum(1.0, np.log(counts + 1) / np.log(64))
    density_layer[np.int_(lidar_data_top[:, 0]), np.int_(lidar_data_top[:, 1])] = normalized_counts
    
    img_density = density_layer * 256
    img_density = img_density.astype(np.uint8)
    #cv2.imshow('density_channel', img_density)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # plt.figure(figsize=(10, 10))
    # plt.imshow(img_density, cmap='hot', interpolation='nearest')
    # plt.title('density channel Heatmap')
    # plt.colorbar()
    # plt.show()
    
    bev_map = np.zeros((3, configs.bev_image_height, configs.bev_image_width))
    bev_map[2, :, :] = density_layer[:configs.bev_image_height, :configs.bev_image_width]
    bev_map[1, :, :] = height_layer[:configs.bev_image_height, :configs.bev_image_width]
    bev_map[0, :, :] = intensity_layer[:configs.bev_image_height, :configs.bev_image_width]
    
    s1, s2, s3 = bev_map.shape
    bev_tensor = np.zeros((1, s1, s2, s3))
    bev_tensor[0] = bev_map
    bev_tensor = torch.from_numpy(bev_tensor)
    output_bev_map = bev_tensor.to(configs.device, non_blocking=True).float()
    
    return output_bev_map
