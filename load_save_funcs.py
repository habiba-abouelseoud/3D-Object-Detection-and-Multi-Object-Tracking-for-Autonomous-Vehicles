import pickle
import os

# load function
def loadObject(path,filename,objectname,frameID=1):
    
    objectname = os.path.join(path, os.path.splitext(filename)[0]
                              + "__frame-" + str(frameID) + "__" + objectname + ".pkl")
    with open(objectname, 'rb') as f:
        object = pickle.load(f)
        return object


def saveObject(object,path,filename,objectname,frameID=1):

    objectname = os.path.join(path, os.path.splitext(filename)[0]+ "__frame-" + str(frameID) + "__" + objectname + ".pkl")
    with open(objectname, 'wb') as f:
        pickle.dump(object,f)


## func to prepare the functions in a list to be running so the will organize if 
# u want to plot pcl only or bev only or detec or eval only




def prepare_execution_list(detection_tasks, visualization_tasks,data_tasks): 
    
   
    task_list  = detection_tasks  + visualization_tasks + data_tasks
    
    
    if any(i in task_list  for i in ('validate_and_filter_labels', 'convert_pcl_to_bev', 'view_pointcloud','plot_obj_bevLabels_camera')): 
        task_list .append('extract_point_cloud_from_frame')
   
    if any(i in task_list  for i in ( 'display_detections_in_bev_and_labels_in_camera')):
        task_list .append('img_load')
    
    return task_list 