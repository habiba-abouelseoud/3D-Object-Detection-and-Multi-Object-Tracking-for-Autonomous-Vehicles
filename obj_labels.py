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


import matplotlib.cm
cmap = matplotlib.cm.get_cmap("viridis")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np



##

"""
This function validates object labels by checking if they contain a sufficient number of LiDAR points
and whether they are within the detection range. Labels marked as difficult to detect or not of type 
"vehicle" are flagged as invalid.
"""
def validate_labels(labels, point_cloud, config_params, min_points):
    flags = np.ones(len(labels)).astype(bool)
    transforms = [np.linalg.inv(waymo_utils.get_box_transformation_matrix(label.box)) for label in labels]
    transforms = np.stack(transforms)
    pcl_no_intensity = point_cloud[:, :3]
    homogeneous_pcl = np.concatenate((pcl_no_intensity, np.ones_like(pcl_no_intensity[:, 0:1])), axis=1)
    transformed_pcl = np.einsum('lij,bj->lbi', transforms, homogeneous_pcl)
    point_mask = np.logical_and.reduce(np.logical_and(transformed_pcl >= -1, transformed_pcl <= 1), axis=2)
    point_counts = point_mask.sum(1)
    flags = point_counts >= min_points

    for idx, label in enumerate(labels):
        label_info = [label.type, label.box.center_x, label.box.center_y, label.box.center_z,
                      label.box.height, label.box.width, label.box.length, label.box.heading]
        flags[idx] = flags[idx] and label_within_detection_area(label_info, config_params)
        if label.detection_difficulty_level > 0 or label.type != label_pb2.Label.Type.TYPE_VEHICLE:
            flags[idx] = False

    return flags



"""
This function converts ground truth labels into 3D objects by filtering out non-vehicle types and
those outside the detection range.
"""
def convert_labels_to_obj(labels, config_params):
    objects = []
    for label in labels:
        if label.type == label_pb2.Label.Type.TYPE_VEHICLE:
            obj = [label.type, label.box.center_x, label.box.center_y, label.box.center_z,
                   label.box.height, label.box.width, label.box.length, label.box.heading]
            if label_within_detection_area(obj, config_params):
                objects.append(obj)
    return objects


"""
This function computes the coordinates of each corner of a bounding box given its center coordinates,
width, length, and yaw angle. It returns the corners in the order: [front_left, rear_left, rear_right, front_right].
"""
def get_box_corners(center_x, center_y, width, length, yaw_angle):
    cos_yaw = np.cos(yaw_angle)
    sin_yaw = np.sin(yaw_angle)

    front_left = (center_x - width / 2 * cos_yaw - length / 2 * sin_yaw,
                  center_y - width / 2 * sin_yaw + length / 2 * cos_yaw)

    rear_left = (center_x - width / 2 * cos_yaw + length / 2 * sin_yaw,
                 center_y - width / 2 * sin_yaw - length / 2 * cos_yaw)

    rear_right = (center_x + width / 2 * cos_yaw + length / 2 * sin_yaw,
                  center_y + width / 2 * sin_yaw - length / 2 * cos_yaw)

    front_right = (center_x + width / 2 * cos_yaw - length / 2 * sin_yaw,
                   center_y + width / 2 * sin_yaw + length / 2 * cos_yaw)

    return [front_left, rear_left, rear_right, front_right]







"""
This function checks whether a label is within the detection area based on a specified minimum overlap.
It converts the label and detection area into polygons and computes their intersection to determine if 
the label is inside the detection area.
"""
def label_within_detection_area(label_info, config_params, min_overlap=0.5):
    _, x, y, _, _, width, length, yaw_angle = label_info
    label_corners = get_box_corners(x, y, width, length, yaw_angle)
    label_polygon = Polygon(label_corners)

    detection_area_width = (config_params.detection_range_x[1] - config_params.detection_range_x[0])
    detection_area_length = (config_params.detection_range_y[1] - config_params.detection_range_y[0])
    detection_area_center_x = config_params.detection_range_x[0] + detection_area_width / 2
    detection_area_center_y = config_params.detection_range_y[0] + detection_area_length / 2
    detection_area_corners = get_box_corners(detection_area_center_x, detection_area_center_y, detection_area_width, detection_area_length, 0)
    detection_area_polygon = Polygon(detection_area_corners)

    intersection_area = detection_area_polygon.intersection(label_polygon)
    overlap_ratio = intersection_area.area / label_polygon.area

    return False if(overlap_ratio <= min_overlap) else True
