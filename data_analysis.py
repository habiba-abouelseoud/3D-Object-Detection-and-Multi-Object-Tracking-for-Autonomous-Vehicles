#Data_processing.py
# general package imports
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
from sklearn.preprocessing import RobustScaler, StandardScaler, QuantileTransformer
# working direc path
sys.path.append(os.getcwd())

# using simpler Waymo open dataset reader repo cloned from github
from simple_waymo_open_dataset_reader import WaymoDataFileReader, dataset_pb2, label_pb2
from simple_waymo_open_dataset_reader import utils as waymo_utils
#/////////////////////////////////////////////////////////////////////////////////////////////



# reading the tf files from the data and assigning them to variable 
df_file = 'training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord' # Sequence 1
#df_file = 'training_segment-10072231702153043603_5725_000_5745_000_with_camera_labels.tfrecord' # Sequence 2
#df_file = 'training_segment-10963653239323173269_1924_000_1944_000_with_camera_labels.tfrecord'  # Sequence 3

rangeFrames = [0, 10]  # used to plot the range number of the frames 


stop_plot_timer = 0  # variable assigned to 0 in which every displayed frame will be plotted after pressing any key

#  loading the data prep
data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Waymo_Dataset', df_file)  
output_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'output')
datafile = WaymoDataFileReader(data_path)
iterator_datafile = iter(datafile)  # iterate over frames in data_path instance

##################

frame_counter = 0

while True:
    try:
        lidar = dataset_pb2.LaserName.TOP # Accessing top lidar data

        #////////////////////////////////////////////////
        # function to print no of vehicals
        # from simple-waymo-open-dataset-reader\examples\groundtruth_extraction.py
        #////////////////////////////////////////////////

        
        def display_vehc_counts(frame):
            veh_conter = 0
            for label in  frame.laser_labels:
                # if cond to assign label type to vehicle label and then count
                if label.type == label.TYPE_VEHICLE: 
                    veh_conter = veh_conter+1
            print("The Number of Vehicals in this Frame are : "+str(veh_conter))
            
# function to plot images from the camera

        def display_cameraImages(frame):

            camera_name = dataset_pb2.CameraName.FRONT #reading camera front data
            image =[imgType for imgType in frame.images if imgType.name == camera_name][0]
            # convet binary image to numpay array and covert bgr to rgb 
            img =np.array(Image.open(io.BytesIO(image.image)))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #resize  the first dimension (obj.shape[0]) corresponds to rows, while the second (obj.shape[1]) corresponds to columns
            dsize=(int(img.shape[1]*0.5),int(img.shape[0]*0.5))
            sized_img = cv2.resize(img,dsize)
            cv2.imshow('Image Displayed from the Fornt Camera of Waymo Car',sized_img)
            cv2.waitKey(0)


 

    # function to plot vertical angle view from lidar , HFOV=360 degrees
        def vFOV_calculation(frame,lidar):
            
            laserCalib = [obj for obj in frame.context.laser_calibrations if obj.name == lidar][0]

            vertical_angle = (((laserCalib.beam_inclination_max - laserCalib.beam_inclination_min)*180)/np.pi)
            hfov_deg = 360

            # Get the number of beam inclinations
            num_beam_inclinations = len(laserCalib.beam_inclinations)

            extrin_param = np.array(laserCalib.extrinsic.transform)

            print(f"Vertical Field of View (VFOV): {vertical_angle} degrees") # 20.360222720319797
            # Print HFOV
            print(f"Horizontal Field of View (HFOV): {hfov_deg} degrees")
            print(f"Number of Beam Inclinations: {num_beam_inclinations}")
             # Print extrinsic parameters
            print("Extrinsic Parameters (Position and Orientation):")
            print(extrin_param)


            # so now i need to plot range images where it will help me to onvert to 3d point clouds
            # https://github.com/Jossome/Waymo-open-dataset-document
            #         |   |-- lasers ⇒ list of Laser
            # |   |   |-- name (LaserName)
            # |   |   |-- ri_return1 (RangeImage class)
            # |   |   |   |-- range_image_compressed
            # |   |   |   |-- camera_projection_compressed
            # |   |   |   |-- range_image_pose_compressed
            # |   |   |   `-- range_image
            # |   |   `-- ri_return2 (same as ri_return1)


            #//////////RANGE IMAGE ///////////
            # as a photograph, but instead of color or intensity
            # values, each pixel stores the distance to the 
            # nearest object at that particular angle.

        def plot_range_image(frame,lidar):
            laserrange = [obj for obj in frame.lasers if obj.name == lidar][0]

            ri1 = [] # to store ri data
            if len(laserrange.ri_return1.range_image_compressed)>0:
                ri1= dataset_pb2.MatrixFloat()
                #ParseFromString is a method -- it does not return anything, but rather fills in self with the parsed content. 
                ri1.ParseFromString(zlib.decompress(laserrange.ri_return1.range_image_compressed))
                # ParseFromString takes the binary data (in this case, the decompressed range image data) 
                # and fills the fields of the protocol buffer message object (ri) with the values extracted 
                # from that data.
                ri1=np.array(ri1.data).reshape(ri1.shape.dims) #The data from the parsed ri object is converted into a NumPy array and reshaped according to the dimensions 
                print(ri1.shape)
                return ri1


        def process_range_image_maxmin(frame,lidar):
            # in this function is responsible for cleaning the data of range images from lidar as lidar has
            # 4 channels and the first channel is for the range , so the lidar range data has different values 
            # like +ve and -ve and we dont need the negate values as it dosnt make since so i will set it to 0 
            # and that means the farthest dis is max rang , min range is nearst dis

            ri_cleaned = plot_range_image(frame,lidar)
            ri_cleaned [ri_cleaned < 0] =0.0
            max_ri = round( np.amax(ri_cleaned[:, :, 0]),2)
            min_ri =round(np.amin(ri_cleaned[:, :, 0]),2)
            print("Maximum Range dedacted = " + str( max_ri) + "meters")
            print("Minimum Range dedacted = " + str( min_ri) + "meters")


        def display_image_range(frame,lidar):
            ''' fhh
            dfjh
            fjffu
            '''
            ri_cleaned = plot_range_image(frame,lidar)
            ri_cleaned [ri_cleaned < 0] =0.0

            ri_scaled = ri_cleaned[ :, :, 0]
            ri_scaled = ((ri_scaled *255)/ (np.amax(ri_scaled)- np.amin(ri_scaled)))
            img_scaled = ri_scaled.astype(np.uint8)
            # we want driver view so we going to crop img +-45 shape[1] horizontel
            degree= int (img_scaled.shape[1]/8)
            img_center = int (img_scaled.shape[1]/2)
            # shape [0:, 1 ] we are using 1 and : to starte the end and beging
            croped_imgRange = img_scaled[ :, img_center - degree: img_center + degree]
            cv2.imshow('range_image', croped_imgRange)
            cv2.waitKey(0)



          ##NEED A MINOR FIX ---- FIXED ALREADY KHALAS 
        def display_intensity_imgRange(frame,lidar):
            img_range_intensity = [obj for obj in frame.lasers if obj.name == lidar][0]
            if len(img_range_intensity.ri_return1.range_image_compressed)>0:
                ri1= dataset_pb2.MatrixFloat()
                #ParseFromString is a method -- it does not return anything, but rather fills in self with the parsed content. 
                ri1.ParseFromString(zlib.decompress(img_range_intensity.ri_return1.range_image_compressed))
                # ParseFromString takes the binary data (in this case, the decompressed range image data) 
                # and fills the fields of the protocol buffer message object (ri) with the values extracted 
                # from that data.
                ri1=np.array(ri1.data).reshape(ri1.shape.dims) 
            ri1[ri1<0]=0.0
            ri_inten = ri1[ :, :, 1]
            ri_inten = np.amax(ri_inten)/2 * ri_inten * 255 / (np.amax(ri_inten) - np.amin(ri_inten))
            img_inten = ri_inten.astype(np.uint8)

            # we want driver view so we going to crop img +-45 shape[1] horizontel
            degree= int (img_inten.shape[1]/8)
            img_center = int (img_inten.shape[1]/2)
            # shape [0:, 1 ] we are using 1 and : to starte the end and beging
            croped_imgIntensity = img_inten[ :, img_center - degree: img_center + degree]
            cv2.imshow('Range_Image_Intensity', croped_imgIntensity)
            cv2.waitKey(0)


  




        

     

                  

            



 
            
            

                #///////////////////////////////////////
                # more functions to implement  for ri1
                #//////////////////////////////////////


                # 1- Comparative Analysis: Compare different aspects of the range images
                # from ri_return1 and ri_return2. This could involve plotting both range
                # images side by side or overlaying them for comparison.

                # 2- Statistical Analysis: You could compute statistics on the range images,
                # such as mean, median, standard deviation, etc., and visualize these statistics 
                # to gain insights into the distribution and variability of the range data.

                # 3- Pose Visualization: Visualize the range_image_pose_compressed data to 
                # understand the spatial orientation or position of the range image in relation 
                # to some reference frame.



        

        def stat_analysis(frame):
            laserCalib =  frame.context.stats 
            obj_coun = len(laserCalib.laser_object_counts)
            weatheratframe = laserCalib.weather
                # Printing statistics and weather information
            print("Statistics Analysis:")
            print(f"Number of objects detected: {obj_coun}")
            print(f"Weather at frame: {weatheratframe}")




        #def pose_lid(frame,lidar):
        #     # posli = frame.pose
        #     # translation = posli.translation
        #     # rotation = posli.rotation

        #     #     # Format translation
        #     # translation_str = f"Translation: X={translation.x}, Y={translation.y}, Z={translation.z}"
        #     # # Format rotation (in quaternion)
        #     # rotation_str = f"Rotation (quaternion): X={rotation.x}, Y={rotation.y}, Z={rotation.z}, W={rotation.w}"
        #     # print(translation_str)
        #     # print(rotation_str)
            
        #     laserpos = [obj for obj in frame.lasers if obj.name == lidar][0]

        #     ri1 = [] # to store ri data
        #     if len(laserpos.ri_return1.range_image_pose_compressed)>0:
        #         ri1= dataset_pb2.MatrixFloat()
        #         #ParseFromString is a method -- it does not return anything, but rather fills in self with the parsed content. 
        #         ri1.ParseFromString(zlib.decompress(laserpos.ri_return1.range_image_pose_compressed))
        #         # ParseFromString takes the binary data (in this case, the decompressed range image data) 
        #         # and fills the fields of the protocol buffer message object (ri) with the values extracted 
        #         # from that data.
        #         ri1=np.array(ri1.data).reshape(ri1.shape.dims) #The data from the parsed ri object is converted into a NumPy array and reshaped according to the dimensions 
        #         print(ri1.shape)

        #         translation_threshold = 1000
        #         rotation_yaw_range = (-0.1, 0.1) 

        #         # Print the 6 data elements per point
        #         for i in range(ri1.shape[0]):
        #             for j in range(ri1.shape[1]):
        #                 point_data = ri1[i, j]
        #                 x, y, z, roll, pitch, yaw = point_data
        #                 if x > translation_threshold or rotation_yaw_range[0] <= yaw <= rotation_yaw_range[1]:
        #                     print(f"Point ({i}, {j}): X={x}, Y={y}, Z={z}, "
        #                           f"Roll={roll}, Pitch={pitch}, Yaw={yaw}")
            
        
        # calc pitch angel Pitch Angle Resolution= Vertical Field of View/ Number of Channels−1


            
        def pitch_angle_resolution(frame,lidar):
            laserrange = [obj for obj in frame.lasers if obj.name == lidar][0]

            ri1 = [] # to store ri data
            if len(laserrange.ri_return1.range_image_compressed)>0:
                ri1= dataset_pb2.MatrixFloat()
                #ParseFromString is a method -- it does not return anything, but rather fills in self with the parsed content. 
                ri1.ParseFromString(zlib.decompress(laserrange.ri_return1.range_image_compressed))
                # ParseFromString takes the binary data (in this case, the decompressed range image data) 
                # and fills the fields of the protocol buffer message object (ri) with the values extracted 
                # from that data.
                ri1=np.array(ri1.data).reshape(ri1.shape.dims) #The data from the parsed ri object is converted into a NumPy array and reshaped according to the dimensions 
                
                # VFOV
                
            laserCalib = [obj for obj in frame.context.laser_calibrations if obj.name == lidar][0]
            vertical_angle = (laserCalib.beam_inclination_max - laserCalib.beam_inclination_min)

            #PITCH resol  ,    ri1.shape[0]=64
            pitch_resol = (((vertical_angle/ri1.shape[0])*180)/np.pi)
            
            print(f"The Pitch Angle Resolution : {pitch_resol:.2f} degrees")



        #def convert_PointCloud(frame,lidar):
            





            
        

           

        




            








        frame = next(iterator_datafile)
        if frame_counter < rangeFrames[0]:
            frame_counter = frame_counter+1
            continue
        elif frame_counter > rangeFrames[1]:

            print(" Number of Frames Reached")

            break
        print('Frames Processed #' + str(frame_counter))
        #display_vehc_counts(frame)
        #display_cameraImages(frame)
        #vFOV_calculation(frame,lidar)
        plot_range_image(frame,lidar)
        #stat_analysis(frame)
        #pose_lid(frame,lidar) #NOT WORKING
        #pitch_angle_resolution(frame,lidar)
        process_range_image_maxmin(frame,lidar)
        #display_image_range(frame,lidar)

        display_intensity_imgRange(frame,lidar)

  
        


        # increment frame counter
        frame_counter = frame_counter + 1











    except StopIteration:

        break






























