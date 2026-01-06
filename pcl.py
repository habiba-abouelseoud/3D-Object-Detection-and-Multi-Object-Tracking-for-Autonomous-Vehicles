
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
from easydict import EasyDict as edict

from simple_waymo_open_dataset_reader.utils import compute_beam_inclinations,compute_range_image_polar,compute_range_image_cartesian, get_rotation_matrix 
import matplotlib.cm
cmap = matplotlib.cm.get_cmap("viridis")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np


#////////////////////////////////////////////////////////

# THE FIRST 5 FUNCTIONS ARE COPIED FROM  simple_waymo_open_dataset_reader.utils  LIBRARY

def compute_beam_inclinations(calibration, height):
    """ Compute the inclination angle for each beam in a range image. """

    if len(calibration.beam_inclinations) > 0:
        return np.array(calibration.beam_inclinations)
    else:
        inclination_min = calibration.beam_inclination_min
        inclination_max = calibration.beam_inclination_max

        return np.linspace(inclination_min, inclination_max, height)


def compute_range_image_polar(range_image, extrinsic, inclination):
    """ Convert a range image to polar coordinates. """

    height = range_image.shape[0]
    width = range_image.shape[1]

    az_correction = math.atan2(extrinsic[1,0], extrinsic[0,0])
    azimuth = np.linspace(np.pi,-np.pi,width) - az_correction

    azimuth_tiled = np.broadcast_to(azimuth[np.newaxis,:], (height,width))
    inclination_tiled = np.broadcast_to(inclination[:,np.newaxis],(height,width))

    return np.stack((azimuth_tiled,inclination_tiled,range_image))


def compute_range_image_cartesian(range_image_polar, extrinsic, pixel_pose, frame_pose):
    """ Convert polar coordinates to cartesian coordinates. """

    azimuth = range_image_polar[0]
    inclination = range_image_polar[1]
    range_image_range = range_image_polar[2]

    cos_azimuth = np.cos(azimuth)
    sin_azimuth = np.sin(azimuth)
    cos_incl = np.cos(inclination)
    sin_incl = np.sin(inclination)

    x = cos_azimuth * cos_incl * range_image_range
    y = sin_azimuth * cos_incl * range_image_range
    z = sin_incl * range_image_range

    range_image_points = np.stack([x,y,z,np.ones_like(z)])
    range_image_points = np.einsum('ij,jkl->ikl', extrinsic,range_image_points)

    return range_image_points


def get_rotation_matrix(roll, pitch, yaw):
    """ Convert Euler angles to a rotation matrix"""

    cos_roll = np.cos(roll)
    sin_roll = np.sin(roll)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    cos_pitch = np.cos(pitch)
    sin_pitch = np.sin(pitch)

    ones = np.ones_like(yaw)
    zeros = np.zeros_like(yaw)

    r_roll = np.stack([
        [ones,  zeros,     zeros],
        [zeros, cos_roll, -sin_roll],
        [zeros, sin_roll,  cos_roll]])

    r_pitch = np.stack([
        [ cos_pitch, zeros, sin_pitch],
        [ zeros,     ones,  zeros],
        [-sin_pitch, zeros, cos_pitch]])

    r_yaw = np.stack([
        [cos_yaw, -sin_yaw, zeros],
        [sin_yaw,  cos_yaw, zeros],
        [zeros,    zeros,   ones]])

    pose = np.einsum('ijhw,jkhw,klhw->ilhw',r_yaw,r_pitch,r_roll)
    pose = pose.transpose(2,3,0,1)
    return pose


def project_to_pointcloud(frame, ri, camera_projection, range_image_pose, calibration):
    """ Create a pointcloud in vehicle space from LIDAR range image. """
    beam_inclinations = compute_beam_inclinations(calibration, ri.shape[0])
    beam_inclinations = np.flip(beam_inclinations)

    extrinsic = np.array(calibration.extrinsic.transform).reshape(4,4)
    frame_pose = np.array(frame.pose.transform).reshape(4,4)

    ri_polar = compute_range_image_polar(ri[:,:,0], extrinsic, beam_inclinations)


    ri_cartesian = compute_range_image_cartesian(ri_polar, extrinsic, None, frame_pose)
    ri_cartesian = ri_cartesian.transpose(1,2,0)

    mask = ri[:,:,0] > 0

    return ri_cartesian[mask,:3], ri[mask]



##/////////////////////////////////////////////////////////////////////////////////////////////////////





def overlay_pointcloud_on_image(image, point_cloud, transform_matrix):
    """
    This function projects a point cloud onto an image using a given transformation matrix, 
    filters points that are behind the camera or outside the image boundaries, and draws 
    colored circles on the image based on the points' distances.
    """
    point_cloud_hom = np.concatenate((point_cloud, np.ones_like(point_cloud[:, 0:1])), axis=1)
    projected_point_cloud = np.einsum('ij,bj->bi', transform_matrix, point_cloud_hom) 

    valid_points_mask = projected_point_cloud[:, 2] > 0
    projected_point_cloud = projected_point_cloud[valid_points_mask]
    valid_point_attrs = point_cloud[valid_points_mask]

    projected_point_cloud = projected_point_cloud[:, :2] / projected_point_cloud[:, 2:3]

    in_image_mask = np.logical_and(
        np.logical_and(projected_point_cloud[:, 0] > 0, projected_point_cloud[:, 0] < image.shape[1]),
        np.logical_and(projected_point_cloud[:, 1] > 0, projected_point_cloud[:, 1] < image.shape[0]))

    projected_point_cloud = projected_point_cloud[in_image_mask]
    valid_point_attrs = valid_point_attrs[in_image_mask]

    color_intensity = 255 * cmap(valid_point_attrs[:, 0] / 30)

    for idx in range(projected_point_cloud.shape[0]):
        cv2.circle(image, (int(projected_point_cloud[idx, 0]), int(projected_point_cloud[idx, 1])), 1, color_intensity[idx])








def extract_point_cloud_from_frame(frame_data, lidar_name):
    """
    This function extracts the point cloud from a given frame of LIDAR data. It retrieves the LIDAR data,
    parses the range image, converts the range image to a point cloud, and stacks the point cloud
    with its intensity values.
    """
    lidar_data = waymo_utils.get(frame_data.lasers, lidar_name)
    range_image_data, cam_proj_data, range_image_pose_data = waymo_utils.parse_range_image_and_camera_projection(lidar_data)

    lidar_calibration = waymo_utils.get(frame_data.context.laser_calibrations, lidar_name)
    point_cloud, point_cloud_attrs = project_to_pointcloud(frame_data, range_image_data, cam_proj_data, range_image_pose_data, lidar_calibration)

    combined_points = np.column_stack((point_cloud, point_cloud_attrs[:, 1]))

    return combined_points



#/////////////////////////////////////////////////////////





def view_lidar_rangeimg(frame, lidar):
    laserrange = [obj for obj in frame.lasers if obj.name == lidar][0]

    ri1 = []  # to store ri data
    if len(laserrange.ri_return1.range_image_compressed) > 0:
        ri1 = dataset_pb2.MatrixFloat()
        # ParseFromString is a method -- it does not return anything, but rather fills in self with the parsed content.
        ri1.ParseFromString(zlib.decompress(laserrange.ri_return1.range_image_compressed))
        # ParseFromString takes the binary data (in this case, the decompressed range image data)
        # and fills the fields of the protocol buffer message object (ri) with the values extracted
        # from that data.
        ri1 = np.array(ri1.data).reshape(ri1.shape.dims)  # The data from the parsed ri object is converted into a NumPy array and reshaped according to the dimensions

    ri1[ri1 < 0] = 0.0

    # [height, width, channels]
    ri_scaled = ri1[:, :, 0]
    ri_scaled = ((ri_scaled * 255) / (np.amax(ri_scaled) - np.amin(ri_scaled)))
    img_scaled = ri_scaled.astype(np.uint8)

    ri_inten = ri1[:, :, 1]
    ri_inten = np.amax(ri_inten) / 2 * ri_inten * 255 / (np.amax(ri_inten) - np.amin(ri_inten))
    img_inten = ri_inten.astype(np.uint8)

    stack_range_inten = np.vstack((img_scaled, img_inten))
    stackedTobit = stack_range_inten.astype(np.uint8)
    # we want driver view so we going to crop img +-90shape[1] horizontel
    degree = int(img_inten.shape[1] / 4)
    img_center = int(img_inten.shape[1] / 2)
    # shape [0:, 1 ] we are using 1 and : to starte the end and beging
    croped_imgIntensity = img_inten[:, img_center - degree: img_center + degree]

    return croped_imgIntensity
#///////////////////////////////////////






def view_pointcloud(pcl):
    # https://www.open3d.org/docs/release/index.html#python-api-index



    vis_window = o3d.visualization.VisualizerWithKeyCallback()
    vis_window.create_window(window_name='The Lidar Point Cloud Image')

    global plot_key
    plot_key = True

    def esc_click(vis_window):
        global plot_key
        print('The user clicked on the Escape keyboard key')
        plot_key = False
        return
    vis_window.register_key_callback(256, esc_click)
    
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pcl[:,:3])

    vis_window.add_geometry(point_cloud)
    while plot_key:
        vis_window.poll_events()
        vis_window.update_renderer()




























