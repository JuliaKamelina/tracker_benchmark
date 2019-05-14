import numpy as np

class OTBHC:
    hog_params = {
        "cell_size": 6,
        "compressed_dim": [10]
    }

    cn_params = {
        "tablename": "CNnorm",
        "useForColor": True,
        "cell_size": 4,
        "compressed_dim": [3]
    }

    ic_params = {
        "tablename": "intensityChannelNorm6",
        "useForColor": False,
        "cell_size": 4,
        "compressed_dim": [3]
    }

    params = {
        "t_features": [
            {
                'name': 'get_table_feature',
                'fparams': cn_params
            },
            {
                'name': 'get_fhog',
                'fparams': hog_params
            }
            # {
            #     'name': 'get_table_feature',
            #     'fparams': ic_params
            # }
        ],

        "t_global": {
            "normalize_power": 2,    # Lp normalization with this p
            "normalize_size": True,  # Also normalize with respect to the spatial size of the feature
            "normalize_dim": True   # Also normalize with respect to the dimensionality of the feature
        },

        # Image sample parameters
        "search_area_shape": 'square',
        "search_area_scale": 4.0,
        "min_image_sample_size": 150**2,
        "max_image_sample_size": 200**2,

        # Detection parameters
        "refinement_iterations": 1,
        "newton_iterations": 5,
        "clamp_position": False,

        # Learning parameters
        "output_sigma_factor": 1/14.0, # 1/16.0
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
        "projection_reg": 1e-7,

        # Generative sample space model parameters
        "use_sample_merge": True,
        "sample_merge_type": 'merge',
        "distance_matrix_update_type": 'exact',
        "minimum_sample_weight": 0.009*(1-0.009)**(2*50),

        # Conjugate Gradient parameters
        "CG_iter": 5,
        "init_CG_iter": 5*15, #10*15
        "init_GN_iter": 5,    #10
        "CG_use_FR": False,
        "CG_standard_alpha": True,
        "CG_forgetting_rate": 50,
        "precond_data_param": 0.75,
        "precond_reg_param": 0.25,
        "precond_proj_param": 40,

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
        "use_scale_filter": True,
        "scale_sigma_factor": 1 / 16.0,
        "scale_learning_rate": 0.025,
        "number_of_scales_filter": 17,        # number of scales
        "number_of_interp_scales": 33,        # number of interpolated scales
        "scale_model_factor": 1.0,            # scaling of the scale model
        "scale_step_filter": 1.02,            # the scale factor of the scale sample patch
        "scale_model_max_area": 32 * 16,      # maximume area for the scale sample patch
        "scale_feature": 'HOG4',              # features for the scale filter (only HOG4 supported)
        "s_num_compressed_dim": 'MAX',        # number of compressed feature dimensions in the scale filter
        "lambda": 1e-2,                       # scale filter regularization
        "do_poly_interp": True,

        "use_gpu": False
    }