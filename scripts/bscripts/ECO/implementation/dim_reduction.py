import numpy as np

def init_projection_matrix(init_sample, compressed_dim, proj_method):
        x = [np.reshape(x, (-1, x.shape[2])) for x in init_sample]
        x = [z - z.mean(0) for z in x]
        proj_matrix = []
        if proj_method == 'pca':
            for x_, compressed_dim_  in zip(x, compressed_dim):
                proj_mat, _, _ = np.linalg.svd(x_.T.dot(x_))
                proj_mat = proj_mat[:, :compressed_dim_]
                proj_matrix.append(proj_mat)
        elif proj_method == 'rand_uni':
            for x_, compressed_dim_ in zip(x, compressed_dim):
                proj_mat = np.random.uniform(size=(x_.shape[1], compressed_dim_))
                proj_mat /= np.sqrt(np.sum(proj_mat**2, axis=0, keepdims=True))
                proj_matrix.append(proj_mat)
        return proj_matrix

def project_sample(x, P):
    return [np.matmul(P_.T, x_) for x_, P_ in zip(x, P)]