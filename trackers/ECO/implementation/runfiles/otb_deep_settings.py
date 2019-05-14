import numpy as np

class OTBDeep:
    hog_params = {
        "cell_size": 4,
        "compressed_dim": [10]
    }

    cnn_params = {
        "nn_name": 'vgg16',
        "output_layer": np.array([3, 14]),
        "downsample_factor": np.array([2, 1]),  # How much to downsample each output layer
        "compressed_dim": [16, 64],   # Compressed dimensionality of each output layer
        "input_size_mode": 'adaptive',
        "input_size_scale": 1
    }

    params = {
        "t_features": [
            {
                'name': 'get_cnn_layers',
                # 'feature': (lambda im, pos, sample_sz, scale_factor: get_cnn_layers(im, pos, sample_sz, scale_factor)),
                'fparams': cnn_params
            },
            {
                'name': 'get_fhog',
                # 'feature': (lambda im, pos, sample_sz, scale_factor: get_fhog(im, pos, sample_sz, scale_factor)),
                'fparams': hog_params
            }
        ],

        "t_global": {
            "normalize_power": 2,    # Lp normalization with this p
            "normalize_size": True,  # Also normalize with respect to the spatial size of the feature
            "normalize_dim": True   # Also normalize with respect to the dimensionality of the feature
        },

        # Image sample parameters
        "search_area_shape": 'square',
        "search_area_scale": 4.5,
        "min_image_sample_size": 200**2,
        "max_image_sample_size": 250**2,

        # Detection parameters
        "refinement_iterations": 1,
        "newton_iterations": 5,
        "clamp_position": False,

        # Learning parameters
        "output_sigma_factor": 1/8.0, # 1/12.0
        "learning_rate": 0.009,  #0.01
        "nSamples": 50,
        "sample_replace_strategy": 'lowest_prior',
        "lt_size": 0,
        "train_gap": 5,
        "skip_after_frame": 1,
        "use_detection_sample": True,

        # Factorized convolution parameters
        "use_projection_matrix": True,
        "update_projection_matrix": True,
        "proj_init_method": 'pca',
        "projection_reg": 5e-8,

        # Generative sample space model parameters
        "use_sample_merge": True,
        "sample_merge_type": 'merge',
        "distance_matrix_update_type": 'exact',
        "minimum_sample_weight": 0.009*(1-0.009)**(2*50),

        # Conjugate Gradient parameters
        "CG_iter": 5,
        "init_CG_iter": 10*15, #15*15
        "init_GN_iter": 10,    #15
        "CG_use_FR": False,
        "CG_standard_alpha": True,
        "CG_forgetting_rate": 75,
        "precond_data_param": 0.3,
        "precond_reg_param": 0.015,
        "precond_proj_param": 35,

        # Regularization window parameters
        "use_reg_window": True,
        "reg_window_min": 1e-4,
        "reg_window_edge": 10e-3,
        "reg_window_power": 2,
        "reg_sparsity_threshold": 0.05,

        # Interpolation parameters
        "interpolation_method": 'bicubic',
        "interpolation_bicubic_a": -0.75,
        "interpolation_centering": True,
        "interpolation_windowing": False,

        # Scale parameters for the translation model
        "use_scale_filter": False,
        "number_of_scales": 5,
        "scale_step": 1.02,

        "visualization": 0,
        "use_gpu": False
    }

