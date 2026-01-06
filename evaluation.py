# general package imports
import numpy as np
import matplotlib
matplotlib.use('wxagg') # change backend so that figure maximizing works on Mac as well     
import matplotlib.pyplot as plt

import torch
from shapely.geometry import Polygon
from operator import itemgetter
import csv

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# object detection tools and helper functions
import main_funcs.pcl as pcl

import main_funcs.birdeyeview as  bev
import main_funcs.obj_labels as labelobj
import main_funcs.visualization as vis


def evaluate_detection_performance(detected_objects, ground_truth_labels, valid_labels, min_iou=0.5):
    """
    Compute various performance measures to assess object detection.

    Parameters:
    detected_objects (list): List of detected objects with their attributes.
    ground_truth_labels (list): List of ground truth labels.
    valid_labels (list): List indicating if the ground truth labels are valid.
    min_iou (float): Minimum IoU threshold to consider a detection as a true positive.

    Returns:
    list: Detection performance metrics including IoUs, center deviations, and counts of positives/negatives.
    """

    true_positives = 0
    center_devs = []
    ious = []

    for label, valid in zip(ground_truth_labels, valid_labels):
        matches_lab_det = []
        if valid:
            box = label.box
            box_corners_label = labelobj.get_box_corners(box.center_x, box.center_y, box.width, box.length, box.heading)
            
            for detection in detected_objects:
                _id, x, y, z, _h, w, l, yaw = detection
                box_corners_det = labelobj.get_box_corners(x, y, w, l, yaw)

                dist_x = np.array(box.center_x - x).item()
                dist_y = np.array(box.center_y - y).item()
                dist_z = np.array(box.center_z - z).item()

                try:
                    poly_label = Polygon(box_corners_label)
                    poly_det = Polygon(box_corners_det)
                    intersection = poly_label.intersection(poly_det).area 
                    union = poly_label.union(poly_det).area
                    iou = intersection / union
                except Exception as err:
                    print("Error in computation:", err)
                    continue

                if iou > min_iou:
                    matches_lab_det.append([iou, dist_x, dist_y, dist_z])
            
        if matches_lab_det:
            best_match = max(matches_lab_det, key=itemgetter(0))  # retrieve entry with max iou
            ious.append(best_match[0])
            center_devs.append(best_match[1:])
    
    print("Evaluating detection performance metrics")

    total_positives = valid_labels.sum()
    true_positives = len(ious)
    false_negatives = total_positives - true_positives
    false_positives = len(detected_objects) - true_positives

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    pos_negs = [total_positives, true_positives, false_negatives, false_positives]
    detection_performance = [ious, center_devs, pos_negs]

    return detection_performance


















import numpy as np
import matplotlib.pyplot as plt

def get_detection_performance_frames(all_detection_performance):
    """
    Evaluate object detection performance based on all frames.

    Parameters:
    all_detection_performance (list): List of detection performance metrics for all frames.

    Returns:
    None
    """

    ious = []
    center_devs = []
    pos_negs = []

    for performance in all_detection_performance:
        ious.append(performance[0])
        center_devs.append(performance[1])
        pos_negs.append(performance[2])
    
    pos_negs_arr = np.asarray(pos_negs)

    print('Evaluating performance metrics')

    ## Extract the total number of positives, true positives, false negatives, and false positives 
    total_positives = sum(pos_negs_arr[:, 0])
    true_positives = sum(pos_negs_arr[:, 1])
    false_negatives = sum(pos_negs_arr[:, 2])
    false_positives = sum(pos_negs_arr[:, 3])

    ## Compute precision
    precision = true_positives / float(true_positives + false_positives)

    ## Compute recall 
    recall = true_positives / float(true_positives + false_negatives)

    print('Precision = ' + str(precision) + ", Recall = " + str(recall))

    # Serialize intersection-over-union and deviations in x, y, z
    all_ious = [iou for sublist in ious for iou in sublist]
    devs_x_all = []
    devs_y_all = []
    devs_z_all = []

    for dev in center_devs:
        for d in dev:
            dev_x, dev_y, dev_z = d
            devs_x_all.append(dev_x)
            devs_y_all.append(dev_y)
            devs_z_all.append(dev_z)

    # Compute statistics
    iou_mean = np.mean(all_ious)
    iou_std = np.std(all_ious)

    devx_mean = np.mean(devs_x_all)
    devx_std = np.std(devs_x_all)

    devy_mean = np.mean(devs_y_all)
    devy_std = np.std(devs_y_all)

    devz_mean = np.mean(devs_z_all)
    devz_std = np.std(devs_z_all)

    # Plot results
    data = [precision, recall, all_ious, devs_x_all, devs_y_all, devs_z_all]
    titles = ['Detection Precision', 'Detection Recall', 'Intersection Over Union', 'Position Errors in X', 'Position Errors in Y', 'Position Errors in Z']
    textboxes = [
        '',
        '',
        '',
        '\n'.join((r'$\mathrm{mean}=%.4f$' % (devx_mean,), r'$\mathrm{sigma}=%.4f$' % (devx_std,), r'$\mathrm{n}=%.0f$' % (len(devs_x_all),))),
        '\n'.join((r'$\mathrm{mean}=%.4f$' % (devy_mean,), r'$\mathrm{sigma}=%.4f$' % (devy_std,), r'$\mathrm{n}=%.0f$' % (len(devs_y_all),))),
        '\n'.join((r'$\mathrm{mean}=%.4f$' % (devz_mean,), r'$\mathrm{sigma}=%.4f$' % (devz_std,), r'$\mathrm{n}=%.0f$' % (len(devs_z_all),)))
    ]

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs = axs.ravel()
    num_bins = 20
    props = dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.5)
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow']

    for idx, ax in enumerate(axs):
        ax.hist(data[idx], num_bins, color=colors[idx % len(colors)])
        ax.set_xlabel(titles[idx], fontsize=12, labelpad=10)
        if textboxes[idx]:
            ax.text(0.95, 0.05, textboxes[idx], transform=ax.transAxes, fontsize=10,
                    verticalalignment='bottom', horizontalalignment='right', bbox=props)
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()














# # compute various performance measures to assess object detection
# def measure_detection_performance(detections, labels, labels_valid, min_iou=0.5):
   
#      # find best detection for each valid label 
#     true_positives = 0 # no. of correctly detected objects
#     center_devs = []
#     ious = []
#     for label, valid in zip(labels, labels_valid):
#         matches_lab_det = []
#         if valid: # exclude all labels from statistics which are not considered valid
            
#             # compute intersection over union (iou) and distance between centers

#             ####### ID_S4_EX1 START #######     
#             #######
#             print("student task ID_S4_EX1 ")

#             ## step 1 : extract the four corners of the current label bounding-box
#             box = label.box
#             box_1 = labelobj.get_box_corners(box.center_x, box.center_y, box.width, box.length, box.heading)
            
#             ## step 2 : loop over all detected objects
            
#             for detection in detections:
                
#                 ## step 3 : extract the four corners of the current detection
#                 _id, x, y,z, _h, w, l, yaw = detection
#                 box_2 = labelobj.get_box_corners(x, y, w, l, yaw)

#                 ## step 4 : computer the center distance between label and detection bounding-box in x, y, and z
#                 dist_x = np.array(box.center_x - x).item()
#                 dist_y = np.array(box.center_y - y).item()
#                 dist_z = np.array(box.center_z - z).item()
#                 # dist_x = box.center_x - x
#                 # dist_y = box.center_y - y
#                 # dist_Z = box.center_z - z

#                 ## step 5 : compute the intersection over union (IOU) between label and detection bounding-box
#                 try:
#                     poly_1 = Polygon(box_1)
#                     poly_2 = Polygon(box_2)
#                     intersection = poly_1.intersection(poly_2).area 
#                     union = poly_1.union(poly_2).area
#                     iou = intersection / union
#                 except Exception as err:
#                     print("Error in computation",err)
#                 ## step 6 : if IOU exceeds min_iou threshold, store [iou,dist_x, dist_y, dist_z] in matches_lab_det and increase the TP count
#                 if iou > min_iou:
#                     matches_lab_det.append([iou,dist_x, dist_y, dist_z ])
#                     # true_positives = true_positives + 1
#             #######
#             ####### ID_S4_EX1 END #######     
            
#         # find best match and compute metrics
#         if matches_lab_det:
#             best_match = max(matches_lab_det,key=itemgetter(1)) # retrieve entry with max iou in case of multiple candidates   
#             ious.append(best_match[0])
#             center_devs.append(best_match[1:])
            

 

#     ####### ID_S4_EX2 START #######     
#     #######
#     print("student task ID_S4_EX2")
    
#     # compute positives and negatives for precision/recall   
#     ## step 1 : compute the total number of positives present in the scene

#     all_positives = labels_valid.sum()

#     ## step 2 : compute the number of false negatives
#     true_positives=len(ious)
#     false_negatives = all_positives - true_positives

#     ## step 3 : compute the number of false positives

#     false_positives = len(detections) - true_positives
    
#     #######
#     ####### ID_S4_EX2 END #######     
#     precision=(true_positives)/(true_positives+false_positives)
#     recall=(true_positives)/(true_positives+false_negatives)
#     pos_negs = [all_positives, true_positives, false_negatives, false_positives]
#     det_performance = [ious, center_devs, pos_negs]
 
#     return det_performance












# # evaluate object detection performance based on all frames
# def compute_performance_stats(det_performance_all):

#     # extract elements
#     ious = []
#     center_devs = []
#     pos_negs = []
#     for item in det_performance_all:
#         ious.append(item[0])
#         center_devs.append(item[1])
#         pos_negs.append(item[2])
#     pos_negs_arr = np.asarray(pos_negs)
    
#     ####### ID_S4_EX3 START #######     
#     #######    
#     print('student task ID_S4_EX3')

#     ## step 1 : extract the total number of positives, true positives, false negatives and false positives 
#     positives = sum(pos_negs_arr[:,0])
#     true_positives = sum(pos_negs_arr[:,1])
#     false_negatives = sum(pos_negs_arr[:,2])
#     false_positives = sum(pos_negs_arr[:,3])

#     # step 2 : compute precision

#     precision = true_positives /float(true_positives + false_positives) # When an object is detected, what are the chances of it being real?  

#     ## step 3 : compute recall 

#     recall = true_positives / float(true_positives + false_negatives) # What are the chances of a real object being detected?
    
#     print('precision = ' + str(precision) + ", recall = " + str(recall)) 

#     #######    
#     ####### ID_S4_EX3 END #######     

     

#     # serialize intersection-over-union and deviations in x,y,z
#     ious_all = [element for tupl in ious for element in tupl]
#     devs_x_all = []
#     devs_y_all = []
#     devs_z_all = []
#     for tuple in center_devs:
#         for elem in tuple:
#             dev_x, dev_y, dev_z = elem
#             devs_x_all.append(dev_x)
#             devs_y_all.append(dev_y)
#             devs_z_all.append(dev_z)
    

#     # compute statistics
#     stdev__ious = np.std(ious_all)
#     mean__ious = np.mean(ious_all)

#     stdev__devx = np.std(devs_x_all)
#     mean__devx = np.mean(devs_x_all)

#     stdev__devy = np.std(devs_y_all)
#     mean__devy = np.mean(devs_y_all)

#     stdev__devz = np.std(devs_z_all)
#     mean__devz = np.mean(devs_z_all)
#     #std_dev_x = np.std(devs_x)

#     # plot results
#     data = [precision, recall, ious_all, devs_x_all, devs_y_all, devs_z_all]
#     titles = ['detection precision', 'detection recall', 'intersection over union', 'position errors in X', 'position errors in Y', 'position error in Z']
#     textboxes = ['', '', '',
#                  '\n'.join((r'$\mathrm{mean}=%.4f$' % (np.mean(devs_x_all), ), r'$\mathrm{sigma}=%.4f$' % (np.std(devs_x_all), ), r'$\mathrm{n}=%.0f$' % (len(devs_x_all), ))),
#                  '\n'.join((r'$\mathrm{mean}=%.4f$' % (np.mean(devs_y_all), ), r'$\mathrm{sigma}=%.4f$' % (np.std(devs_y_all), ), r'$\mathrm{n}=%.0f$' % (len(devs_x_all), ))),
#                  '\n'.join((r'$\mathrm{mean}=%.4f$' % (np.mean(devs_z_all), ), r'$\mathrm{sigma}=%.4f$' % (np.std(devs_z_all), ), r'$\mathrm{n}=%.0f$' % (len(devs_x_all), )))]


#     f, a = plt.subplots(2, 3, figsize=(15, 10))
#     a = a.ravel()
#     num_bins = 20
#     props = dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.5)
#     colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow']

#     for idx, ax in enumerate(a):
#         ax.hist(data[idx], num_bins, color=colors[idx % len(colors)])
#         ax.set_xlabel(titles[idx], fontsize=12, labelpad=10)  # Set x-axis label as the title
#         if textboxes[idx]:
#             ax.text(0.95, 0.05, textboxes[idx], transform=ax.transAxes, fontsize=10,
#                     verticalalignment='bottom', horizontalalignment='right', bbox=props)
#         ax.grid(True)  # Add grid lines for better visualization
#     plt.tight_layout()
#     plt.show()































