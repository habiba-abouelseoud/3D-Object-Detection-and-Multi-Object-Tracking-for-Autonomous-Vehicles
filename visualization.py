
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


from main_funcs.obj_labels import convert_labels_to_obj 

from main_funcs.birdeyeview import render_detections_on_bev




# # extract RGB front camera image and camera calibration
# def extract_front_camera_image(frame):
#     # extract camera and calibration from frame
#     camera_name = dataset_pb2.CameraName.FRONT
#     camera = waymo_utils.get(frame.images, camera_name)

#     # get image and convert tom RGB
#     image = waymo_utils.decode_image(camera)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     return image


# def show_bev(bev_maps, configs):

#     bev_map = (bev_maps.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
#     bev_map = cv2.resize(bev_map, (configs.bev_width, configs.bev_height))
#     bev_map = cv2.rotate(bev_map, cv2.ROTATE_180)
#     cv2.imshow('BEV map', bev_map)


# # visualize ground-truth labels as overlay in birds-eye view
# def show_objects_labels_in_bev(detections, object_labels, bev_maps, configs):

#     # project detections and labels into birds-eye view
#     bev_map = (bev_maps.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
#     bev_map = cv2.resize(bev_map, (configs.bev_width, configs.bev_height))
    
#     label_detections = convert_labels_to_obj(object_labels, configs)
#     render_detections_on_bev(bev_map, label_detections, configs, [0,255,0])
#     render_detections_on_bev(bev_map, detections, configs, [0,0,255])
    

#     bev_map = cv2.rotate(bev_map, cv2.ROTATE_180)
#     cv2.imshow('labels (green) vs. detected objects (red)', bev_map)


#//////////////////////////////////////////////////



def get_front_camera_image(frame_data):
    """
    Extract the front camera image and camera calibration from the given frame data.

    Parameters:
    frame_data (Frame): Frame data containing camera images and calibration.

    Returns:
    image (ndarray): RGB image from the front camera.
    """
    camera_name = dataset_pb2.CameraName.FRONT
    camera = waymo_utils.get(frame_data.images, camera_name)

    image = waymo_utils.decode_image(camera)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

def display_bev(bev_data, config):

    """
    Display the Bird's Eye View (BEV) map.

    Parameters:
    bev_data (Tensor): BEV map data.
    config (Edict): Configuration parameters.

    Returns:
    None
    """
    bev_map = (bev_data.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    bev_map = cv2.resize(bev_map, (config.bev_image_width, config.bev_image_height))
    bev_map = cv2.rotate(bev_map, cv2.ROTATE_180)
    cv2.imshow('BEV Map', bev_map)


def visualize_objects_and_labels_in_bev(detected_objects, ground_truth_labels, bev_data, config):
    """
    Visualize ground-truth labels and detected objects as overlay in Bird's Eye View (BEV).

    Parameters:
    detected_objects (list): List of detected objects.
    ground_truth_labels (list): List of ground-truth labels.
    bev_data (Tensor): BEV map data.
    config (Edict): Configuration parameters.

    Returns:
    None
    """
    bev_map = (bev_data.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    bev_map = cv2.resize(bev_map, (config.bev_image_width, config.bev_image_height))
    
    label_detections = convert_labels_to_obj(ground_truth_labels, config)
    render_detections_on_bev(bev_map, label_detections, config, [0, 255, 0])
    render_detections_on_bev(bev_map, detected_objects, config, [0, 0, 255])
    
    bev_map = cv2.rotate(bev_map, cv2.ROTATE_180)
    cv2.imshow('Labels (green) vs. Detected Objects (red)', bev_map)





















#///////////////////////////////////////////////////////////////































# # visualize detection results as overlay in birds-eye view and ground-truth labels in camera image
# def show_objects_in_bev_labels_in_camera(detections, bev_maps, image, object_labels, object_labels_valid, camera_calibration, configs):

#     # project detections into birds-eye view
#     bev_map = (bev_maps.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
#     bev_map = cv2.resize(bev_map, (configs.bev_image_width, configs.bev_image_height))
#     render_detections_on_bev(bev_map, detections, configs)
#     bev_map = cv2.rotate(bev_map, cv2.ROTATE_180)

#     # project ground-truth labels into camera image
#     img_rgb = project_labels_into_camera(camera_calibration, image, object_labels, object_labels_valid)

#     # merge camera image and bev image into a combined view
#     img_rgb_h, img_rgb_w = img_rgb.shape[:2]
#     ratio_rgb = configs.result_image_width / img_rgb_w
#     output_rgb_h = int(ratio_rgb * img_rgb_h)
#     ret_img_rgb = cv2.resize(img_rgb, (configs.result_image_width, output_rgb_h))

#     img_bev_h, img_bev_w = bev_map.shape[:2]
#     ratio_bev = configs.result_image_width / img_bev_w
#     output_bev_h = int(ratio_bev * img_bev_h)
#     ret_img_bev = cv2.resize(bev_map, (configs.result_image_width, output_bev_h))

#     out_img = np.zeros((output_rgb_h + output_bev_h, configs.result_image_width, 3), dtype=np.uint8)
#     out_img[:output_rgb_h, ...] = ret_img_rgb
#     out_img[output_rgb_h:, ...] = ret_img_bev

#      # Add title to the combined image
#     title_font = cv2.FONT_HERSHEY_SIMPLEX
#     title_text = "Labels vs. Detected Objects"
#     title_size = 1
#     title_color = (255, 255, 255)  # white
#     title_thickness = 2
#     title_position = (10, 30)  # x, y coordinates

#     cv2.putText(out_img, title_text, title_position, title_font, title_size, title_color, title_thickness)

#     # Add grid lines to the BEV image for better visualization
#     num_grid_lines = 10  # adjust number of grid lines
#     bev_img_with_grid = ret_img_bev.copy()
#     bev_h, bev_w = bev_img_with_grid.shape[:2]
#     step_size = bev_w // num_grid_lines

#     for i in range(1, num_grid_lines):
#         # Vertical lines
#         cv2.line(bev_img_with_grid, (i * step_size, 0), (i * step_size, bev_h), (0, 255, 0), 1)
#         # Horizontal lines
#         cv2.line(bev_img_with_grid, (0, i * step_size), (bev_w, i * step_size), (0, 255, 0), 1)

#     out_img[output_rgb_h:, ...] = bev_img_with_grid

#     # Show combined view
#     cv2.imshow('Labels vs. Detected Objects', out_img)
#     cv2.waitKey(0)  # Wait for a key press to close the window
#     cv2.destroyAllWindows()

#     #     # Plot the camera image
#     # fig, ax = plt.subplots(2, 1, figsize=(12, 8))
#     # ax[0].imshow(cv2.cvtColor(ret_img_rgb, cv2.COLOR_BGR2RGB))
#     # ax[0].set_title('Camera Image with Ground-Truth Labels')
#     # ax[0].axis('off')
    
#     # # Plot the BEV map
#     # ax[1].imshow(ret_img_bev)
#     # ax[1].set_title('Bird\'s Eye View with Detections')
#     # ax[1].axis('off')

#     # plt.tight_layout()
#     # plt.show()

#     # show combined view
#    # cv2.imshow('labels vs. detected objects', out_img)








# # visualize object labels in camera image
# def project_labels_into_camera(camera_calibration, image, labels, labels_valid, img_resize_factor=1.0):

#     # get transformation matrix from vehicle frame to image
#     vehicle_to_image = waymo_utils.get_image_transform(camera_calibration)

#     # draw all valid labels
#     for label, vis in zip(labels, labels_valid):
#         if vis:
#             colour = (0, 255, 0)
#         else:
#             colour = (255, 0, 0)

#         # only show labels of type "vehicle"
#         if(label.type == label_pb2.Label.Type.TYPE_VEHICLE):
#             waymo_utils.draw_3d_box(image, vehicle_to_image, label, colour=colour)

#     # resize image
#     if (img_resize_factor < 1.0):
#         width = int(image.shape[1] * img_resize_factor)
#         height = int(image.shape[0] * img_resize_factor)
#         dim = (width, height)
#         img_resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
#         return img_resized
#     else:
#         return image





#///////////////////////////



def plot_obj_bevLabels_camera(detected_objects, bev_maps, camera_image, ground_truth_labels, ground_truth_labels_valid, camera_calibration, configs):
    """
    Visualize detection results as overlay in birds-eye view and ground-truth labels in camera image.

    Parameters:
    detected_objects (list): List of detected objects.
    bev_maps (Tensor): BEV map data.
    camera_image (ndarray): RGB image from the front camera.
    ground_truth_labels (list): List of ground-truth labels.
    ground_truth_labels_valid (list): List of valid ground-truth labels.
    camera_calibration (Calibration): Camera calibration data.
    configs (Edict): Configuration parameters.

    Returns:
    None
    """
    # Project detections into birds-eye view
    bev_map_array = (bev_maps.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    bev_map_resized = cv2.resize(bev_map_array, (configs.bev_image_width, configs.bev_image_height))
    render_detections_on_bev(bev_map_resized, detected_objects, configs)
    bev_map_resized_rotated = cv2.rotate(bev_map_resized, cv2.ROTATE_180)

    # Project ground-truth labels into camera image
    camera_image_with_labels = labels_to_camera_plot(camera_calibration, camera_image, ground_truth_labels, ground_truth_labels_valid)

    # Merge camera image and BEV image into a combined view
    camera_image_height, camera_image_width = camera_image_with_labels.shape[:2]
    camera_image_ratio = configs.result_image_width / camera_image_width
    camera_image_output_height = int(camera_image_ratio * camera_image_height)
    resized_camera_image = cv2.resize(camera_image_with_labels, (configs.result_image_width, camera_image_output_height))

    bev_map_height, bev_map_width = bev_map_resized_rotated.shape[:2]
    bev_map_ratio = configs.result_image_width / bev_map_width
    bev_map_output_height = int(bev_map_ratio * bev_map_height)
    resized_bev_map = cv2.resize(bev_map_resized_rotated, (configs.result_image_width, bev_map_output_height))

    combined_image = np.zeros((camera_image_output_height + bev_map_output_height, configs.result_image_width, 3), dtype=np.uint8)
    combined_image[:camera_image_output_height, ...] = resized_camera_image
    combined_image[camera_image_output_height:, ...] = resized_bev_map

    # Add title to the combined image
    title_font = cv2.FONT_HERSHEY_SIMPLEX
    title_text = "Labels (green) vs. Detected Objects (red)"
    title_size = 1
    title_color = (255, 255, 255)  # white
    title_thickness = 2
    title_position = (10, 30)  # x, y coordinates

    cv2.putText(combined_image, title_text, title_position, title_font, title_size, title_color, title_thickness)

    # Add grid lines to the BEV image for better visualization
    num_grid_lines = 10  # adjust number of grid lines
    bev_map_with_grid = resized_bev_map.copy()
    bev_height, bev_width = bev_map_with_grid.shape[:2]
    grid_step_size = bev_width // num_grid_lines

    for i in range(1, num_grid_lines):
        # Vertical lines
        cv2.line(bev_map_with_grid, (i * grid_step_size, 0), (i * grid_step_size, bev_height), (0, 255, 0), 1)
        # Horizontal lines
        cv2.line(bev_map_with_grid, (0, i * grid_step_size), (bev_width, i * grid_step_size), (0, 255, 0), 1)

    combined_image[camera_image_output_height:, ...] = bev_map_with_grid

    # Show combined view
    cv2.imshow('Labels (green) vs. Detected Objects (red)', combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
    #     # Plot the camera image
    # fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    # ax[0].imshow(cv2.cvtColor(ret_img_rgb, cv2.COLOR_BGR2RGB))
    # ax[0].set_title('Camera Image with Ground-Truth Labels')
    # ax[0].axis('off')
    
    # # Plot the BEV map
    # ax[1].imshow(ret_img_bev)
    # ax[1].set_title('Bird\'s Eye View with Detections')
    # ax[1].axis('off')

    # plt.tight_layout()
    # plt.show()

    # show combined view
   # cv2.imshow('labels vs. detected objects', out_img)







def labels_to_camera_plot(camera_calibration, camera_image, ground_truth_labels, ground_truth_labels_valid, img_resize_factor=1.0):
    """
    Visualize object labels in the camera image.

    Parameters:
    camera_calibration (Calibration): Camera calibration data.
    camera_image (ndarray): RGB image from the front camera.
    ground_truth_labels (list): List of ground-truth labels.
    ground_truth_labels_valid (list): List of valid ground-truth labels.
    img_resize_factor (float): Factor to resize the image.

    Returns:
    img_resized (ndarray): Resized image with projected labels.
    """
    # Get transformation matrix from vehicle frame to image
    vehicle_to_image_transform = waymo_utils.get_image_transform(camera_calibration)

    # Draw all valid labels
    for label, is_valid in zip(ground_truth_labels, ground_truth_labels_valid):
        label_color = (0, 255, 0) if is_valid else (255, 0, 0)

        # Only show labels of type "vehicle"
        if label.type == label_pb2.Label.Type.TYPE_VEHICLE:
            waymo_utils.draw_3d_box(camera_image, vehicle_to_image_transform, label, color=label_color)

    # Resize image if needed
    if img_resize_factor < 1.0:
        resized_width = int(camera_image.shape[1] * img_resize_factor)
        resized_height = int(camera_image.shape[0] * img_resize_factor)
        dimensions = (resized_width, resized_height)
        resized_image = cv2.resize(camera_image, dimensions, interpolation=cv2.INTER_AREA)
        return resized_image
    else:
        return camera_image


