import sys
import os
import cv2
import numpy as np

from PIL import Image
sys.path.append('./')

from .implementation import Tracker
from .implementation.utils import load_video_info, get_sequence_info

def ECO(path):
    cur_path = os.path.dirname(os.path.abspath(__file__))

    video_path = path # "{}/sequences/Crossing".format(cur_path)
    seq, ground_truth = load_video_info(video_path)
    seq = get_sequence_info(seq)
    frames = [np.array(Image.open(f)) for f in seq["image_files"]]
    is_color = True if (len(frames[0].shape) == 3) else False
    tracker = Tracker(seq, frames[0], is_color)
    f = open("ECO.txt", "w+")
    for i, frame in enumerate(frames):
        bbox, time = tracker.Track(frame, i)
        print(bbox)
        print(time)
        if is_color:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        frame = cv2.rectangle(frame,
                              (int(bbox[0]), int(bbox[2])),
                              (int(bbox[1]), int(bbox[3])),
                              (0, 255, 255), 1)
        gt_bbox = (ground_truth[i, 0], 
                   ground_truth[i, 0] + ground_truth[i, 2],
                   ground_truth[i, 1],
                   ground_truth[i, 1] + ground_truth[i, 3])
        frame = cv2.rectangle(frame,
                              (int(gt_bbox[0]), int(gt_bbox[2])),
                              (int(gt_bbox[1]), int(gt_bbox[3])),
                              (0, 255, 0), 1)
        #cv2.imshow('', frame)
        #cv2.waitKey(1)
        rect = (bbox[0], bbox[1], np.abs(bbox[0] - bbox[2]), np.abs(bbox[1] - bbox[3]))
        f.write("%d %d %d %d\r\n" % rect)
    f.close()
