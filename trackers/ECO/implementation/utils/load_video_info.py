import numpy as np
import os.path

def load_video_info(path):
    gt_path = "{}/groundtruth_rect.txt".format(path)
    ground_truth = np.loadtxt(gt_path)

    seq = dict()
    seq["format"] = "otb"
    seq["len"] = len(ground_truth)
    seq["init_rect"] = ground_truth[0]

    img_path = path + '/img/'
    img_files = list()
    if (os.path.isfile(img_path + '%04d' % 1 + '.png')):
        for i in range(1,seq["len"]):
            img_files.append(img_path + '%04d' % i + '.png')
    elif (os.path.isfile(img_path + '%04d' % 1 + '.jpg')):
        for i in range(1,seq["len"]):
            img_files.append(img_path + '%04d' % i + '.jpg')
    elif (os.path.isfile(img_path + '%04d' % 1 + '.bmp')):
        for i in range(1,seq["len"]):
            img_files.append(img_path + '%04d' % i + '.bmp')
    else:
         raise ValueError("No images loaded")

    seq["image_files"] = img_files
    return(seq, ground_truth)
