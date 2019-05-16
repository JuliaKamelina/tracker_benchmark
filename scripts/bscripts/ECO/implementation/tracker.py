import numpy as np
import cv2 as cv
import math
import time
import sys

# import cProfile

from scipy import signal

from .feature_extraction import init_features, get_cnn_layers, get_fhog, get_table_feature
from .initialization import get_interp_fourier, get_reg_filter
from .fourier_tools import cfft2, interpolate_dft, shift_sample, full_fourier_coeff, ifft2, fft2, compact_fourier_coeff, sample_fs
from .dim_reduction import *
from .sample_space_model import update_sample_space_model
from .train import train_joint, train_filter
from .optimize_scores import *
from .scale_filter import ScaleFilter
from .runfiles import settings

def _round(x):
    res = x.copy()
    res[0] = np.ceil(x[0]) if x[0] - np.floor(x[0]) >= 0.5 else np.floor(x[0])
    res[1] = np.ceil(x[1]) if x[1] - np.floor(x[1]) >= 0.5 else np.floor(x[1])
    return res

class Tracker:
    def InitCG(self):
        params = settings.params
        self.init_CG_opts = {
            "CG_use_FR": True,
            "tol": 1e-6,
            "CG_standard_alpha": True
        }
        self.CG_opts = {
            "CG_use_FR": params["CG_use_FR"],
            "tol": 1e-6,
            "CG_standard_alpha": params["CG_standard_alpha"]
        }
        if params["CG_forgetting_rate"] == np.inf or params["learning_rate"] >= 1:
            self.CG_opts["init_forget_factor"] = 0
        else:
            self.CG_opts["init_forget_factor"] = (1-params["learning_rate"])**params["CG_forgetting_rate"]

    def SetCosWindow(self, sz):
        cos_window = []
        for i in range(0, len(sz)):
            cos_y = np.hanning(int(sz[i][0]+2))
            cos_x = np.hanning(int(sz[i][1]+2))
            cos_x = cos_x.reshape(len(cos_x), 1)
            cos_window.append(cos_y*cos_x)
        cos_window = np.array(cos_window)
        for i in range(0, len(cos_window)):
            cos_window[i] = cos_window[i][1:-1,1:-1]
            cos_window[i] = cos_window[i].reshape(cos_window[i].shape[0], cos_window[i].shape[1], 1, 1)
        self.cos_window = cos_window

    def SetScales(self, im):
        if settings.params["use_scale_filter"]:
            self.scale_filter = ScaleFilter(self.target_sz)
            self.nScales = self.scale_filter.num_scales
            self.scaleFactors = self.scale_filter.scale_factors
            scale_step = self.scale_filter.scale_step
        else:
            self.nScales = settings.params["number_of_scales"]
            scale_step = settings.params["scale_step"]
            scale_exp = np.arange(-np.floor((self.nScales-1)/2), np.ceil((self.nScales-1)/2)+1)
            self.scaleFactors = scale_step**scale_exp

        if self.nScales > 0:
            # force reasonable scale changes
            self.min_scale_factor = scale_step ** np.ceil(np.log(np.max(5 / self.img_support_sz)) / np.log(scale_step))
            self.max_scale_factor = scale_step ** np.floor(np.log(np.min(im.shape[:2] / self.base_target_sz)) / np.log(scale_step))

    def __init__(self, seq, im, is_color=True):
        params = settings.params
        features = params["t_features"]

        self.is_color_image = is_color
        self.pos = seq["init_pos"]
        self.target_sz = seq["init_sz"]
        self.nSamples = min(params["nSamples"], seq["num_frames"])
        self.num_frames = seq["num_frames"]

        search_area = np.prod(self.target_sz * params["search_area_scale"])
        if search_area > params["max_image_sample_size"]:
            self.currentScaleFactor = math.sqrt(search_area / params["max_image_sample_size"])
        elif search_area < params["min_image_sample_size"]:
            self.currentScaleFactor = math.sqrt(search_area / params["min_image_sample_size"])
        else:
            self.currentScaleFactor = 1.0

        self.base_target_sz = np.array(self.target_sz, np.float32) / self.currentScaleFactor

        if params["search_area_shape"] == 'proportional':
            img_sample_sz = math.floor(self.base_target_sz * params["search_area_scale"])
        if params["search_area_shape"] == 'square':
            img_sample_sz = np.tile(math.sqrt(np.prod(self.base_target_sz*params["search_area_scale"])), (1, 2))[0]
        if params["search_area_shape"] == 'fix_padding':
            img_sample_sz = self.base_target_sz + math.sqrt(np.prod(self.base_target_sz*params["search_area_scale"]) + (self.base_target_sz[0] - self.base_target_sz[1])/4) - sum(self.base_target_sz)/2
        if params["search_area_shape"] == 'custom':
            img_sample_sz = np.array((self.base_target_sz[0]*2, self.base_target_sz[1]*2), float)

        img_sample_sz[0] = np.ceil(img_sample_sz[0]) if img_sample_sz[0] - np.floor(img_sample_sz[0]) > 0.5 else np.floor(img_sample_sz[0])
        img_sample_sz[1] = np.ceil(img_sample_sz[1]) if img_sample_sz[1] - np.floor(img_sample_sz[1]) > 0.5 else np.floor(img_sample_sz[1])

        init_features(self.is_color_image, img_sample_sz, 'odd_cells')

        # Set feature info
        self.img_support_sz = features[0]["img_sample_sz"]

        feature_sz = []
        for i in range(0, len(features)):
            if len(features[i]["data_sz"].shape) > 1:
                for item in features[i]["data_sz"]:
                    feature_sz.append(item)
            else:
                feature_sz.append(features[i]["data_sz"])
        self.feature_dim = [item for i in range(0, len(features)) for item in features[i]["fparams"]["nDim"]]
        self.num_feature_blocks = len(self.feature_dim)

        if params["use_projection_matrix"]:
            self.sample_dim = [x for feature in features for x in feature["fparams"]["compressed_dim"]]
            # self.sample_dim = [features[i]["fparams"]["compressed_dim"] for i in range(0, len(features))]
            # self.sample_dim = np.concatenate((self.sample_dim[0], np.array(self.sample_dim[1]).reshape(1,)))
        else:
            self.sample_dim = self.feature_dim

        feature_sz = np.array(feature_sz)
        self.filter_sz = feature_sz + (feature_sz + 1) % 2

        self.k_max = np.argmax(self.filter_sz)
        self.output_sz = self.filter_sz[self.k_max]  # The size of the label function DFT == maximum filter size

        self.block_inds = list(range(0, self.num_feature_blocks))
        self.block_inds.remove(self.k_max)

        self.pad_sz = np.array([(self.output_sz - sz)/2 for sz in self.filter_sz], np.int32)
        self.kx = [np.arange(-1*int(np.ceil(sz[0] - 1)/2.0), 1, dtype=np.float32) for sz in self.filter_sz]
        self.ky = [np.arange(-1*int(np.ceil(sz[0] - 1)/2.0), int(np.ceil(sz[0] - 1)/2.0) + 1, dtype=np.float32) for sz in self.filter_sz]

        #Gaussian label function
        sig_y = np.sqrt(np.prod(np.floor(self.base_target_sz))) * params["output_sigma_factor"] * (self.output_sz / self.img_support_sz)
        yf_y = [np.sqrt(2*math.pi)*sig_y[0]/self.output_sz[0]*np.exp(-2*(math.pi*sig_y[0]*y/self.output_sz[0])**2) for y in self.ky]
        yf_x = [np.sqrt(2*math.pi)*sig_y[1]/self.output_sz[1]*np.exp(-2*(math.pi*sig_y[1]*x/self.output_sz[1])**2) for x in self.kx]
        self.yf = [y.reshape(-1, 1)*x for y, x in zip(yf_y, yf_x)]

        self.SetCosWindow(feature_sz)

        #Fourier for interpolation func
        interp1_fs = []
        interp2_fs = []
        for i in range(0, len(self.filter_sz)):
            interp1, interp2 = get_interp_fourier(self.filter_sz[i])
            interp1_fs.append(interp1.reshape(interp1.shape[0], 1, 1, 1))
            interp2_fs.append(interp2.reshape(1, interp2.shape[0], 1, 1))
        self.interp1_fs = interp1_fs
        self.interp2_fs = interp2_fs

        reg_window_edge = np.array([])
        shape = 0
        for i in range(0, len(features)):
            shape += len(features[i]["fparams"]["nDim"])
        reg_window_edge = reg_window_edge.reshape((shape, 0))

        self.reg_filter = [get_reg_filter(self.img_support_sz, self.base_target_sz, reg_win_edge)
                                     for reg_win_edge in reg_window_edge]
        self.reg_energy = [np.real(np.vdot(reg_filter.flatten(), reg_filter.flatten()))
                            for reg_filter in self.reg_filter]
        
        self.SetScales(im)
        self.InitCG()

        #init and alloc
        self.prior_weights = np.zeros((self.nSamples,1), dtype=np.float32)
        self.samplesf = [[]] * self.num_feature_blocks
        for i in range(self.num_feature_blocks):
            self.samplesf[i] = np.zeros((int(self.filter_sz[i, 0]), int((self.filter_sz[i, 1]+1)/2),
                                        self.sample_dim[i], self.nSamples), dtype=np.complex64)
    
        self.scores_fs_feat = [[]] * self.num_feature_blocks

        self.distance_matrix = np.ones((self.nSamples, self.nSamples), dtype=np.float32) * np.inf  # stores the square of the euclidean distance between each pair of samples
        self.gram_matrix = np.ones((self.nSamples, self.nSamples), dtype=np.float32) * np.inf  # Kernel matrix, used to update distance matrix

        self.frames_since_last_train = 0
        self.num_training_samples = 0

    # @profile
    def Track(self, frame, iter):
        params = settings.params
        features = params["t_features"]
        tic = time.clock()

        if iter == 0:  # INIT AND UPDATE TRACKER
            self.sample_pos = [0,0]
            self.sample_pos[0] = np.ceil(self.pos[0]) if self.pos[0] - np.floor(self.pos[0]) > 0.5 else np.floor(self.pos[0])
            self.sample_pos[1] = np.ceil(self.pos[1]) if self.pos[1] - np.floor(self.pos[1]) > 0.5 else np.floor(self.pos[1])
            sample_scale = self.currentScaleFactor
            xl = [x for i in range(0, len(features))
                    for x in features[i]["feature"](frame, self.sample_pos, features[i]['img_sample_sz'], self.currentScaleFactor, i)]
            # print(xl)

            xlw = [x * y for x, y in zip(xl, self.cos_window)]      # do windowing of feature
            xlf = [cfft2(x) for x in xlw]                      # compute the fourier series
            xlf = interpolate_dft(xlf, self.interp1_fs, self.interp2_fs) # interpolate features
            xlf = compact_fourier_coeff(xlf)                   # new sample to add
            # shift sample
            shift_samp = 2 * np.pi * (self.pos - self.sample_pos) / (sample_scale * self.img_support_sz) # img_sample_sz
            xlf = shift_sample(xlf, shift_samp, self.kx, self.ky)
            self.proj_matrix = init_projection_matrix(xl, self.sample_dim, params['proj_init_method'])  # init projection matrix
            xlf_proj = project_sample(xlf, self.proj_matrix)  # project sample

            merged_sample, new_sample, merged_sample_id, new_sample_id = update_sample_space_model(self.samplesf, xlf_proj, self.num_training_samples, 
                                                                                                    self.distance_matrix, self.gram_matrix, self.prior_weights)
            self.num_training_samples += 1

            if params["update_projection_matrix"]:
                # insert new sample
                for i in range(0, self.num_feature_blocks):
                    self.samplesf[i][:, :, :, new_sample_id:new_sample_id+1] = new_sample[i]

            self.sample_energy = [np.real(x * np.conj(x)) for x in xlf_proj]

            # init CG params
            self.CG_state = None
            if params["update_projection_matrix"]:
                self.init_CG_opts['maxit'] = np.ceil(params["init_CG_iter"] / params["init_GN_iter"])
                self.hf = [[[]] * self.num_feature_blocks for _ in range(2)]
                feature_dim_sum = float(np.sum(self.feature_dim))
                proj_energy = [2 * np.sum(np.abs(yf_.flatten())**2) / feature_dim_sum * np.ones_like(P)
                                for P, yf_ in zip(self.proj_matrix, self.yf)]
            else:
                self.CG_opts['maxit'] = params["init_CG_iter"]
                self.hf = [[[]] * self.num_feature_blocks]

            # init filter
            for i in range(0, self.num_feature_blocks):
                self.hf[0][i] = np.zeros((int(self.filter_sz[i][0]), int((self.filter_sz[i][1]+1)/2), int(self.sample_dim[i]), 1), dtype=np.complex64)
            if params['update_projection_matrix']:
                # init gauss-newton optimiztion of filter and proj matrix
                self.hf, self.proj_matrix = train_joint(self.hf, self.proj_matrix, xlf, self.yf, self.reg_filter, self.sample_energy, self.reg_energy, proj_energy, self.init_CG_opts)
                xlf_proj = project_sample(xlf, self.proj_matrix) # reproject
                for i in range(0, self.num_feature_blocks):
                    self.samplesf[i][:, :, :, 0:1] = xlf_proj[i]  # insert new sample

                if params['distance_matrix_update_type'] == 'exact':
                    # find the norm of reproj sample
                    new_train_sample_norm = 0
                    for i in range(0, self.num_feature_blocks):
                        new_train_sample_norm += 2 * np.real(np.vdot(xlf_proj[i].flatten(), xlf_proj[i].flatten()))
                    self.gram_matrix[0, 0] = new_train_sample_norm
            self.hf_full = full_fourier_coeff(self.hf)

            if params['use_scale_filter'] and self.nScales > 0:
                self.scale_filter.update(frame, self.pos, self.base_target_sz, self.currentScaleFactor)
        else:   # TARGET LOCALIZATION
            old_pos = np.zeros((2))
            for _ in range(0, params['refinement_iterations']):
                if not np.allclose(old_pos, self.pos):
                    old_pos = self.pos.copy()
                    self.sample_pos = _round(self.pos)
                    sample_scale = self.currentScaleFactor*self.scaleFactors
                    xt = [x for i in range(0, len(features))
                          for x in features[i]["feature"](frame, self.sample_pos, features[i]['img_sample_sz'], sample_scale, i)] # extract features

                    xt_proj = project_sample(xt, self.proj_matrix)  # project sample
                    xt_proj = [fmap * cos for fmap, cos in zip(xt_proj, self.cos_window)]  # do windowing
                    xtf_proj = [cfft2(x) for x in xt_proj]  # fouries series
                    xtf_proj = interpolate_dft(xtf_proj, self.interp1_fs, self.interp2_fs)  # interpolate features

                    # compute convolution for each feature block in the fourier domain, then sum over blocks
                    self.scores_fs_feat = [[]]*self.num_feature_blocks
                    self.scores_fs_feat[self.k_max] = np.sum(self.hf_full[self.k_max]*xtf_proj[self.k_max], 2)
                    scores_fs = self.scores_fs_feat[self.k_max]

                    for ind in self.block_inds:
                        self.scores_fs_feat[ind] = np.sum(self.hf_full[ind]*xtf_proj[ind], 2)
                        scores_fs[int(self.pad_sz[ind][0]):int(self.output_sz[0]-self.pad_sz[ind][0]),
                                  int(self.pad_sz[ind][1]):int(self.output_sz[0]-self.pad_sz[ind][1])] += self.scores_fs_feat[ind]

                    # OPTIMIZE SCORE FUNCTION with Newnot's method.
                    trans_row, trans_col, scale_idx = optimize_scores(scores_fs, params["newton_iterations"])

                    # compute the translation vector in pixel-coordinates and round to the cloest integer pixel
                    translation_vec = np.array([trans_row, trans_col])*(self.img_support_sz/self.output_sz)*self.currentScaleFactor*self.scaleFactors[scale_idx]
                    scale_change_factor = self.scaleFactors[scale_idx]

                    # update_position
                    self.pos = self.sample_pos + translation_vec

                    if params['clamp_position']:
                        self.pos = np.maximum(np.array(0, 0), np.minimum(np.array(frame.shape[:2]), self.pos))

                    # do scale tracking with scale filter
                    if self.nScales > 0 and params['use_scale_filter']:
                        scale_change_factor = self.scale_filter.track(frame, self.pos, self.base_target_sz, self.currentScaleFactor)

                    # update scale
                    self.currentScaleFactor *= scale_change_factor

                    # adjust to make sure we are not to large or to small
                    if self.currentScaleFactor < self.min_scale_factor:
                        self.currentScaleFactor = self.min_scale_factor
                    elif self.currentScaleFactor > self.max_scale_factor:
                        self.currentScaleFactor = self.max_scale_factor

            # MODEL UPDATE STEP
            if params['learning_rate'] > 0:
                # use sample that was used for detection
                sample_scale = sample_scale[scale_idx]
                xlf_proj = [xf[:, :(xf.shape[1]+1)//2, :, scale_idx:scale_idx+1] for xf in xtf_proj]

                # shift sample target is centred
                shift_samp = 2*np.pi*(self.pos - self.sample_pos)/(sample_scale*self.img_support_sz)
                xlf_proj = shift_sample(xlf_proj, shift_samp, self.kx, self.ky)

            # update the samplesf to include the new sample. The distance matrix, kernel matrix and prior weight are also updated
            merged_sample, new_sample, merged_sample_id, new_sample_id = update_sample_space_model(self.samplesf, xlf_proj, self.num_training_samples, self.distance_matrix, self.gram_matrix, self.prior_weights)
            if self.num_training_samples < self.nSamples:
                self.num_training_samples += 1

            if params['learning_rate'] > 0:
                for i in range(0, self.num_feature_blocks):
                    if merged_sample_id >= 0:
                        self.samplesf[i][:,:,:,merged_sample_id:merged_sample_id+1] = merged_sample[i]
                    if new_sample_id >= 0:
                        self.samplesf[i][:,:,:,new_sample_id:new_sample_id+1] = new_sample[i]

            # train filter
            if iter < params['skip_after_frame'] or self.frames_since_last_train >= params['train_gap']:
                new_sample_energy = [np.real(xlf * np.conj(xlf)) for xlf in xlf_proj]
                self.CG_opts['maxit'] = params['CG_iter']
                self.sample_energy = [(1 - params['learning_rate'])*se + params['learning_rate']*nse
                                 for se, nse in zip(self.sample_energy, new_sample_energy)]

                # do CG opt for filter
                self.hf, self.CG_state = train_filter(self.hf, self.samplesf, self.yf, self.reg_filter, self.prior_weights, self.sample_energy, self.reg_energy, self.CG_opts, self.CG_state)
                self.hf_full = full_fourier_coeff(self.hf)
                self.frames_since_last_train = 0
            else:
                self.frames_since_last_train += 1
            if params['use_scale_filter']:
                self.scale_filter.update(frame, self.pos, self.base_target_sz, self.currentScaleFactor)

            # update target size
            self.target_sz = self.base_target_sz*self.currentScaleFactor
        tracker_time = time.clock() - tic
        bbox = (int(self.pos[1] - self.target_sz[1]/2),  # x_min
                int(self.pos[1] + self.target_sz[1]/2),  # x_max
                int(self.pos[0] - self.target_sz[0]/2),  # y_min
                int(self.pos[0] + self.target_sz[0]/2))  # y_max
        print(self.pos)
        print(self.target_sz)
        return (bbox, tracker_time)
