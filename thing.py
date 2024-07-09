import os
import numpy as np
import pandas as pd
import cv2
from tracker.boost_track import BoostTrack

import os
import shutil
import time
from typing import Tuple

import torch
import cv2
import torchvision
import torchreid
import numpy as np
import dataset
import utils
from args import make_parser
from external.adaptors import detector
from tracker.GBI import GBInterpolation
from tracker.boost_track import BoostTrack

def read_labels(label_path):
    filename = os.path.splitext(os.path.basename(label_path))[0]
    frame_number = int(filename.replace('freqsweep', ''))
    
    with open(label_path, 'r') as file:
        labels = file.readlines()
    
    parsed_labels = []
    for label in labels:
        class_id, x_center, y_center, width, height, confidence = map(float, label.split())
        parsed_labels.append((frame_number, x_center, y_center, width, height, class_id, confidence))
    
    return parsed_labels

def read_all_labels(directory_path):
    all_labels = []
    
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt') and filename.startswith('freqsweep'):
            label_path = os.path.join(directory_path, filename)
            labels = read_labels(label_path)
            all_labels.extend(labels)
    
    return all_labels

def non_max_suppression_fast(boxes, scores, overlapThresh):
    if len(boxes) == 0:
        return [], []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], x2[idxs[:last]])


        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int"), pick

# Initialize SORT
tracker = BoostTrack(det_thresh=0.2,lambda_iou=0.4,lambda_mhd=0.4,lambda_shape=0.2, max_age=50, iou_threshold=0.1) 
if tracker is not None:
    tracker.dump_cache()                               
# Read labels from the directory
label_dir = "./labels"
labels = read_all_labels(label_dir)
label_dataframe = pd.DataFrame(data=labels, columns=["frame_number", "x_center", "y_center", "width", "height", "class_id", "confidence"])

# Read image files
#image_dir = 'testfiles/'

image_dir = 'imageset_2_10_100/'
image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')])

colorID = {}
tracked_targets = []
# Loop through each frame and process
for frame_number, image_path in enumerate(image_files):
    start_time = time.time()
    tag = f"video:{frame_number}"



    frame = cv2.imread(image_path)

    # Step 2: Convert the image from BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Step 3: Convert the image to a NumPy array (if not already)
    # (OpenCV's imread already returns a NumPy array)

    # Step 4: Transpose the image dimensions to match PyTorch's expected format (Channels x Height x Width)
    frame_rgb = frame_rgb.transpose(2, 0, 1)

    # Step 5: Convert the NumPy array to a PyTorch tensor
    frame_tensor = torch.tensor(frame_rgb, dtype=torch.float32).unsqueeze(0).cuda()  # Add batch dimension and move to CUDA
    
    # Get image dimensions
    h, w, _ = frame.shape
    
    # Filter labels for the current frame number
    filtered_df = label_dataframe[label_dataframe['frame_number'] == frame_number]
    
    # Convert normalized coordinates to absolute coordinates
    bboxes_with_scores = []
    for _, row in filtered_df.iterrows():
        x_center, y_center, width, height = row[['x_center', 'y_center', 'width', 'height']]
        x1 = int((x_center - width / 2) * w)
        y1 = int((y_center - height / 2) * h)
        x2 = int((x_center + width / 2) * w)
        y2 = int((y_center + height / 2) * h)
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))
        
        if x2 > x1 and y2 > y1:
            score = row['confidence']
            bboxes_with_scores.append([x1, y1, x2, y2, score])
            #print(f'Bounding box (frame {frame_number}): [{x1}, {y1}, {x2}, {y2}, {score}]')

    bboxes_with_scores = np.array(bboxes_with_scores)
    if len(bboxes_with_scores) == 0:
        bboxes_with_scores= np.empty((0, 5))  # Skip if no valid bounding boxes are found for this frame
    

    targets = tracker.update(bboxes_with_scores , frame_tensor, frame, tag)
    #[x1,y1,x2,y2,score]
    
    
    
    # Visualize results
    for d in targets:
        x1, y1, x2, y2, obj_id, score = int(d[0]), int(d[1]), int(d[2]), int(d[3]), int(d[4]), d[5]
        tracked_targets.append([frame_number, x1, y1, x2, y2, obj_id, score])
        if obj_id not in colorID:
            colorID[obj_id] = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        color = colorID[obj_id]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if(np.abs(x1-x2)>20):
            cv2.putText(frame, f'ID: {obj_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        #print(f'Tracker bbox (ID {obj_id}): [{x1}, {y1}, {x2}, {y2}]')
    
    # Save the frame with tracked particles
    output_path = os.path.join("./outputfiles", f'frame_{frame_number:04d}.jpg')
    cv2.imwrite(output_path, frame)

    cv2.destroyAllWindows()

# Save tracked targets to CSV
tracked_targets_df = pd.DataFrame(tracked_targets, columns=["frame_number", "x1", "y1", "x2", "y2", "obj_id", "score"])
tracked_targets_df.to_csv("tracked_targets_3.csv", index=False)