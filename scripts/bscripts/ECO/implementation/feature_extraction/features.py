import mxnet as mx
import numpy as np
import cv2 as cv
import scipy.io
import os

from mxnet.gluon.model_zoo import vision
from mxnet.gluon.nn import AvgPool2D
from ._gradient import *

from ..runfiles import settings

def _round(x):
    res = x.copy()
    res[0] = np.ceil(x[0]) if x[0] - np.floor(x[0]) >= 0.5 else np.floor(x[0])
    res[1] = np.ceil(x[1]) if x[1] - np.floor(x[1]) >= 0.5 else np.floor(x[1])
    return res

def init_features(is_color_image = False, img_sample_sz = [], size_mode = ''):
    if (size_mode == ''):
        size_mode = 'same'

    features = settings.params['t_features']
    gparams = settings.params['t_global']

    gp_keys = gparams.keys()
    if not 'normalize_power' in gp_keys:
        gparams['normalize_power'] = []
    if not 'normalize_size' in gp_keys:
        gparams['normalize_size'] = True
    if not 'normalize_dim' in gp_keys:
        gparams['normalize_dim'] = False
    if not 'square_root_normalization' in gp_keys:
        gparams['square_root_normalization'] = False
    if not 'use_gpu' in gp_keys:
        gparams['use_gpu'] = False

    keep_features = []
    for i in range(0,len(features)):
        f_keys = features[i]['fparams']
        if not 'useForColor' in f_keys:
            features[i]['fparams']['useForColor'] = True
        if not 'useForGray' in f_keys:
            features[i]['fparams']['useForGray'] = True

        if ((features[i]['fparams']['useForColor'] and is_color_image) or
            (features[i]['fparams']['useForGray'] and  not is_color_image)):
            keep_features.append(features[i])

    features = keep_features

    cell_szs = []
    for i in range(0,len(features)):
        if features[i]['name'] == 'get_fhog':
            if not 'nOrients' in features[i]["fparams"].keys():
                features[i]["fparams"]["nOrients"] = 9
            features[i]["fparams"]["nDim"] = np.array([3*features[i]["fparams"]["nOrients"] + 5 - 1])
            features[i]["is_cell"] = False
            features[i]["is_cnn"] = False
            features[i]['feature'] = (lambda im, pos, sample_sz, scale_factor, feat_ind: get_fhog(im, pos, sample_sz, scale_factor, feat_ind))

        elif features[i]['name'] == 'get_table_feature':
            cur_path = os.path.dirname(os.path.abspath(__file__))
            load_path = cur_path + '/lookup_tables/' + features[i]["fparams"]["tablename"]
            table = scipy.io.loadmat(load_path)
            features[i]["table"] = table
            features[i]["fparams"]["nDim"] = [table[features[i]["fparams"]["tablename"]].shape[1]]
            features[i]["is_cell"] = False
            features[i]["is_cnn"] = False
            features[i]['feature'] = (lambda im, pos, sample_sz, scale_factor, feat_ind: get_table_feature(im, pos, sample_sz, scale_factor, feat_ind))

        elif features[i]['name'] == 'get_colorspace':
            features[i]["fparams"]["nDim"] = 1
            features[i]["is_cell"] = False
            features[i]["is_cnn"] = False

        elif features[i]['name'] == 'get_cnn_layers' or features[i]['name'] == 'get_OFcnn_layers':
            features[i]["fparams"]["output_layer"].sort()
            if not 'input_size_mode' in features[i]["fparams"].keys():
                features[i]["fparams"]["input_size_mode"] = 'adaptive'
            if not 'input_size_scale' in features[i]["fparams"].keys():
                features[i]["fparams"]["input_size_scale"] = 1
            if not 'downsample_factor' in features[i]["fparams"].keys():
                features[i]["fparams"]["downsample_factor"] = np.ones((1, len(features[i]["fparams"]["output_layer"])))

            features[i]["fparams"]["nDim"] = np.array([64, 512]) #[96 512] net["info"]["dataSize"][layer_dim_ind, 2]

            features[i]["fparams"]["cell_size"] = np.array([4, 16]) #stride_tmp*downsample_factor

            features[i]["is_cell"] = True
            features[i]["is_cnn"] = True
            features[i]['feature'] = (lambda im, pos, sample_sz, scale_factor, feat_ind: get_cnn_layers(im, pos, sample_sz, scale_factor, feat_ind))
        else:
            raise ValueError("Unknown feature type")

        if not 'cell_size' in features[i]["fparams"].keys():
            features[i]["fparams"]["cell_size"] = 1
        if not 'penalty' in features[i]["fparams"].keys():
            if len(features[i]["fparams"]["nDim"]) == 1:
                features[i]["fparams"]["penalty"] = 0
            else:
                features[i]["fparams"]["penalty"] = np.zeros((2, 1))
        features[i]["fparams"]["min_cell_size"] = np.min(features[i]["fparams"]["cell_size"])
        cell_szs.append(features[i]["fparams"]["cell_size"])

    cnn_feature_ind = -1
    for i in range(0,len(features)):
        if features[i]["is_cnn"]:
            cnn_feature_ind = i  #last cnn feature

    if cnn_feature_ind >= 0 :
        # scale = features[cnn_feature_ind]["fparams"]["input_size_scale"]
        new_img_sample_sz = np.array(img_sample_sz, dtype=np.int32)

        if size_mode != "same" and features[cnn_feature_ind]["fparams"]["input_size_mode"] == "adaptive":
            orig_sz = np.ceil(new_img_sample_sz/16)

            if size_mode == "exact":
                desired_sz = orig_sz + 1
            elif size_mode == "odd_cells":
                desired_sz = orig_sz + 1 + orig_sz%2
            new_img_sample_sz = desired_sz*16

        if (features[cnn_feature_ind]["fparams"]["input_size_mode"] == "adaptive"):
            features[cnn_feature_ind]["img_sample_sz"] = np.round(new_img_sample_sz) # == feature_info.img_support_sz
        else:
            features[cnn_feature_ind]["img_sample_sz"] = np.array(img_sample_sz) #net["meta"].normalization.imageSize[0:2]
    else:
        max_cell_size = max(cell_szs)

        if size_mode == "same":
            features[cnn_feature_ind]["img_sample_sz"] = np.ceil(img_sample_sz)
        elif size_mode == "exact":
            features[cnn_feature_ind]["img_sample_sz"] = round(img_sample_sz / max_cell_size) * max_cell_size
        elif size_mode == "odd_cells":
            new_img_sample_sz = (1 + 2*_round(img_sample_sz / (2*max_cell_size))) * max_cell_size
            feature_sz_choices = np.array([(new_img_sample_sz.reshape(-1, 1) + np.arange(0, max_cell_size).reshape(1, -1)) // x for x in cell_szs])
            num_odd_dimensions = np.sum((feature_sz_choices % 2) == 1, axis=(0,1))
            best_choice = np.argmax(num_odd_dimensions.flatten())
            features[cnn_feature_ind]["img_sample_sz"] = _round(new_img_sample_sz + best_choice)

    for i in range(0, len(features)):
        if (not features[i]["is_cell"]):
            features[i]["img_sample_sz"] = features[cnn_feature_ind]["img_sample_sz"]
            features[i]["data_sz"] = np.round(features[i]["img_sample_sz"]/features[i]["fparams"]["cell_size"])
        else:
            features[i]["data_sz"] = np.round(features[i]["img_sample_sz"]/features[i]["fparams"]["cell_size"][:, None])


def feature_normalization(x):
    gparams = settings.params['t_global']
    if ('normalize_power' in gparams.keys()) and gparams["normalize_power"] > 0:
        if gparams["normalize_power"] == 2:
            x = x * np.sqrt((x.shape[0]*x.shape[1]) ** gparams["normalize_size"] * (x.shape[2]**gparams["normalize_dim"]) / (x**2).sum(axis=(0, 1, 2)))
        else:
            x = x * ((x.shape[0]*x.shape[1]) ** gparams["normalize_size"]) * (x.shape[2]**gparams["normalize_dim"]) / ((np.abs(x) ** (1. / gparams["normalize_power"])).sum(axis=(0, 1, 2)))

    if gparams["square_root_normalization"]:
        x = np.sign(x) * np.sqrt(np.abs(x))
    return x.astype(np.float32)

def get_sample(im, pos, img_sample_sz, output_sz):
    pos = np.floor(pos)
    sample_sz = np.maximum(_round(img_sample_sz), 1)
    x = np.floor(pos[1]) + np.arange(0, sample_sz[1]+1) - np.floor((sample_sz[1]+1)/2)
    y = np.floor(pos[0]) + np.arange(0, sample_sz[0]+1) - np.floor((sample_sz[0]+1)/2)
    x_min = max(0, int(x.min()))
    x_max = min(im.shape[1], int(x.max()))
    y_min = max(0, int(y.min()))
    y_max = min(im.shape[0], int(y.max()))
    # extract image
    im_patch = im[y_min:y_max, x_min:x_max, :]
    left = right = top = down = 0
    if x.min() < 0:
        left = int(abs(x.min()))
    if x.max() > im.shape[1]:
        right = int(x.max() - im.shape[1])
    if y.min() < 0:
        top = int(abs(y.min()))
    if y.max() > im.shape[0]:
        down = int(y.max() - im.shape[0])
    if left != 0 or right != 0 or top != 0 or down != 0:
        im_patch = cv.copyMakeBorder(im_patch, top, down, left, right, cv.BORDER_REPLICATE)
    im_patch = cv.resize(im_patch, (int(output_sz[0]), int(output_sz[1])), cv.INTER_CUBIC)
    if len(im_patch.shape) == 2:
        im_patch = im_patch[:, :, np.newaxis]
    return im_patch

def forward_pass(x):
    vgg16 = vision.vgg16(pretrained=True)
    avg_pool2d = AvgPool2D()

    conv1_1 = vgg16.features[0].forward(x)
    relu1_1 = vgg16.features[1].forward(conv1_1)
    conv1_2 = vgg16.features[2].forward(relu1_1)
    relu1_2 = vgg16.features[3].forward(conv1_2)
    pool1 = vgg16.features[4].forward(relu1_2) # x2
    pool_avg = avg_pool2d(pool1)

    conv2_1 = vgg16.features[5].forward(pool1)
    relu2_1 = vgg16.features[6].forward(conv2_1)
    conv2_2 = vgg16.features[7].forward(relu2_1)
    relu2_2 = vgg16.features[8].forward(conv2_2)
    pool2 = vgg16.features[9].forward(relu2_2) # x4

    conv3_1 = vgg16.features[10].forward(pool2)
    relu3_1 = vgg16.features[11].forward(conv3_1)
    conv3_2 = vgg16.features[12].forward(relu3_1)
    relu3_2 = vgg16.features[13].forward(conv3_2)
    conv3_3 = vgg16.features[14].forward(relu3_2)
    relu3_3 = vgg16.features[15].forward(conv3_3)
    pool3 = vgg16.features[16].forward(relu3_3) # x8

    conv4_1 = vgg16.features[17].forward(pool3)
    relu4_1 = vgg16.features[18].forward(conv4_1)
    conv4_2 = vgg16.features[19].forward(relu4_1)
    relu4_2 = vgg16.features[20].forward(conv4_2)
    conv4_3 = vgg16.features[21].forward(relu4_2)
    relu4_3 = vgg16.features[22].forward(conv4_3)
    pool4 = vgg16.features[23].forward(relu4_3) # x16
    return [pool_avg.asnumpy().transpose(2, 3, 1, 0),
            pool4.asnumpy().transpose(2, 3, 1, 0)]

def get_cnn_layers(im, pos, sample_sz, scale_factor, feat_ind=0):
    gparams = settings.params['t_global']
    fparams = settings.params['t_features'][0]['fparams']

    compressed_dim = fparams["compressed_dim"] # TODO: check
    cell_size = fparams["cell_size"]
    penalty = fparams["penalty"]
    min_cell_size = np.min(cell_size)

    if im.shape[2] == 1:
        im = cv.cvtColor(im.squeeze(), cv.COLOR_GRAY2RGB)
    if not isinstance(scale_factor, list) and not isinstance(scale_factor, np.ndarray):
        scale_factor = [scale_factor]
    patches = []
    for scale in scale_factor:
        patch = get_sample(im, pos, sample_sz*scale, sample_sz)
        patch = mx.nd.array(patch / 255.)
        normalized = mx.image.color_normalize(patch, mean=mx.nd.array([0.485, 0.456, 0.406]),
                                                    std=mx.nd.array([0.229, 0.224, 0.225]))
        normalized = normalized.transpose((2, 0, 1)).expand_dims(axis=0)
        patches.append(normalized)
    patches = mx.nd.concat(*patches, dim=0)
    f1, f2 = forward_pass(patches)
    f1 = feature_normalization(f1)
    f2 = feature_normalization(f2)
    return f1, f2

def get_fhog(img, pos, sample_sz, scale_factor, feat_ind=0):
    fparams = settings.params['t_features'][1]['fparams']

    feat = []
    if not isinstance(scale_factor, list) and not isinstance(scale_factor, np.ndarray):
        scale_factor = [scale_factor]
    for scale in scale_factor:
        patch = get_sample(img, pos, sample_sz*scale, sample_sz)
        # h, w, c = patch.shape
        M, O = gradMag(patch.astype(np.float32), 0, True)
        H = fhog(M, O, fparams["cell_size"], fparams["nOrients"], -1, .2)
        # drop the last dimension
        H = H[:, :, :-1]
        feat.append(H)
    feat = feature_normalization(np.stack(feat, axis=3))
    return [feat]

def integralImage(img):
    w, h, c = img.shape
    intImage = np.zeros((w+1, h+1, c), dtype=img.dtype)
    intImage[1:, 1:, :] = np.cumsum(np.cumsum(img, 0), 1)
    return intImage

def avg_feature_region(features, region_size):
    region_area = region_size ** 2
    if features.dtype == np.float32:
        maxval = 1.
    else:
        maxval = 255
    intImage = integralImage(features)
    i1 = np.arange(region_size, features.shape[0]+1, region_size).reshape(-1, 1)
    i2 = np.arange(region_size, features.shape[1]+1, region_size).reshape(1, -1)
    region_image = (intImage[i1, i2, :] - intImage[i1, i2-region_size,:] - intImage[i1-region_size, i2, :] + intImage[i1-region_size, i2-region_size, :])  / (region_area * maxval)
    return region_image

def get_table_feature(img, pos, sample_sz, scale_factor, feat_ind):
    feature = settings.params['t_features'][feat_ind]
    name = feature["fparams"]["tablename"]

    feat = []
    factor = 32
    den = 8

    if not isinstance(scale_factor, list) and not isinstance(scale_factor, np.ndarray):
        scale_factor = [scale_factor]
    for scale in scale_factor:
        patch = get_sample(img, pos, sample_sz*scale, sample_sz)
        h, w, c = patch.shape
        if c == 3:
            RR = patch[:, :, 0].astype(np.int32)
            GG = patch[:, :, 1].astype(np.int32)
            BB = patch[:, :, 2].astype(np.int32)
            index = RR // den + (GG // den) * factor + (BB // den) * factor * factor
            f = feature["table"][name][index.flatten()].reshape((h, w, feature["table"][name].shape[1]))
        else:
            f = feature["table"][name][patch.flatten()].reshape((h, w, feature["table"][name].shape[1]))
        if feature["fparams"]["cell_size"] > 1:
            f = avg_feature_region(f, feature["fparams"]["cell_size"])
        feat.append(f)
    feat = feature_normalization(np.stack(feat, axis=3))
    return [feat]

def _fhog(I, bin_size=8, num_orients=9, clip=0.2, crop=False):
    soft_bin = -1
    M, O = gradMag(I.astype(np.float32), 0, True)
    H = fhog(M, O, bin_size, num_orients, soft_bin, clip)
    return H