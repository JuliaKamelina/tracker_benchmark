import numpy as np
import warnings

from scipy.signal import convolve

from .fourier_tools import symmetrize_filter
from .runfiles import settings

def pcg(A, b, opts, M1, M2, ip, x0, state=None, use_gpu=False):
    maxit  = int(opts['maxit'])
    if 'init_forget_factor' not in opts:
        opts['init_forget_factor'] = 1

    x = x0
    p = None
    rho = 1
    r_prev = None

    if state == None:
        state = {}
    else:
        if opts['init_forget_factor'] > 0:
            if 'p' in state:
                p = state['p']
            if 'rho' in state:
                rho = state['rho'] / opts['init_forget_factor']
            if 'r_prev' in state and not opts['CG_use_FR']:
                r_prev = state['r_prev']
    state['flag'] = 1

    r = []
    for z, y in zip(b, A(x)):
        r.append([z1 - y1 for z1, y1 in zip(z, y)])
    
    resvec = []
    for i in range(0, maxit):
        if not M1 == None:
            y = M1(r)
        else:
            y = r

        if not M2 == None:
            z = M2(y)
        else:
            z = y

        rho1 = rho
        rho = ip(r, z)
        if rho == 0 or np.isinf(rho):
            state['flag'] = 4
            break

        if i == 0 and p == None:
            p = z
        else:
            if opts['CG_use_FR']:
                betta = rho/rho1
            else:
                rho2 = ip(r_prev, z)
                betta = (rho - rho2)/rho1
            if betta == 0 or np.isinf(betta):
                state['flag'] = 4
                break
            betta = max(0, betta)
            tmp = []
            for zz, pp in zip(z, p):
                tmp.append([zz1 + betta * pp1 for zz1, pp1 in zip(zz, pp)])
            p = tmp

        q = A(p)
        pq = ip(p, q)
        if pq <= 0 or np.isinf(pq):
            state['flag'] = 4
            break
        else:
            if opts['CG_standard_alpha']:
                alpha = rho / pq
            else:
                alpha = ip(p, r) / pq
        if np.isinf(alpha):
            state['flag'] = 4
        # save old r if not using FR formula for betta
        if not opts['CG_use_FR']:
            r_prev = r

        # form new iterate
        tmp = []
        for xx, pp in zip(x, p):
            tmp.append([xx1 + alpha * pp1 for xx1, pp1 in zip(xx, pp)])
        x = tmp
        if i < maxit:
            tmp = []
            for rr, qq in zip(r, q):
                tmp.append([rr1 - alpha * qq1 for rr1, qq1 in zip(rr, qq)])
            r = tmp

    # save the state
    state['p'] = p
    state['rho'] = rho
    if not opts['CG_use_FR']:
        state['r_prev'] = r_prev
    return x, resvec, state

def diag_precond(hf, M_diag):
    ret = []
    for x, y in zip(hf, M_diag):
        ret.append([x1 / y1 for x1, y1 in zip(x, y)])
    return ret

def inner_product_joint(xf, yf):
    """
        Computes the joint inner product between two filters and projection matrices
    """

    ip = 0
    for i in range(0, len(xf[0])):
        ip += 2*np.vdot(xf[0][i].flatten(), yf[0][i].flatten()) - np.vdot(xf[0][i][:, -1, :].flatten(), yf[0][i][:, -1, :].flatten()) # filter part
        ip += np.vdot(xf[1][i].flatten(), yf[1][i].flatten()) # projection_matrix part
    return np.real(ip)

def lhs_operation_joint(hf, samplesf, reg_filter, init_samplef, XH, init_hf, proj_reg, use_gpu=False):
    """
        left-hand-side operation in Conjugate Gradient
    """

    if use_gpu:
        print("GPU")
        raise(NotImplementedError)

    hf_out = [[[]] * len(hf[0]) for _ in range(len(hf))]

    P = [np.real(h) for h in hf[1]]  # extract projection matrix and filter
    hf = hf[0]

    num_features = len(hf)
    filter_sz = np.zeros((num_features, 2), np.int32)
    for i in range(0, num_features):
        filter_sz[i, :] = np.array(hf[i].shape[:2])

    k1 = np.argmax(filter_sz[:, 0]) # index for the feature block with the largest spatial size
    block_inds = list(range(0, num_features))
    block_inds.remove(k1)
    output_sz = np.array([hf[k1].shape[0], hf[k1].shape[1]*2-1])

    # blockwise matrix multiplications: A^H A f; H = diag(sample_weights)
    # sum over all features and feature blocks
    sh = np.matmul(samplesf[k1].transpose(0, 1, 3, 2), hf[k1])
    pad_sz = [[]] * num_features
    for i in block_inds:
        pad_sz[i] = ((output_sz - np.array([hf[i].shape[0], hf[i].shape[1]*2-1])) / 2).astype(np.int32)
        sh[pad_sz[i][0]:output_sz[0]-pad_sz[i][0], pad_sz[i][1]:, :, :] += np.matmul(samplesf[i].transpose(0, 1, 3, 2), hf[i])

    # multiply with the transpose
    hf_out1 = [[]] * num_features
    hf_out1[k1] = np.matmul(np.conj(samplesf[k1]), sh)
    for i in block_inds:
        hf_out1[i] = np.matmul(np.conj(samplesf[i]), sh[pad_sz[i][0]:output_sz[0]-pad_sz[i][0], pad_sz[i][1]:, :, :])

    # operations corresponding to the regularization term (convolve each feature dimension with the DFT of w, and the transposed
    # operation) add the regularization part
    for i in range(0, num_features):
        reg_pad = min(reg_filter[i].shape[1] - 1, hf[i].shape[1]-1)

        # add part needed for convolution
        hf_conv = np.concatenate([hf[i], np.conj(np.rot90(hf[i][:, -reg_pad-1:-1, :], 2))], axis=1)

        # do first convolution
        hf_conv = convolve(hf_conv, reg_filter[i][:, :, np.newaxis, np.newaxis])
        # do final convolution and put together result
        hf_out1[i] += convolve(hf_conv[:, :-reg_pad, :], reg_filter[i][:, :, np.newaxis, np.newaxis], 'valid')
    
    # B * P
    BP_list = [np.matmul(init_hf_.transpose(0, 1, 3, 2), np.matmul(p.T, init_samp))
               for init_samp, p, init_hf_ in zip(init_samplef, P, init_hf)]
    BP = BP_list[k1]
    for i in block_inds:
        BP[pad_sz[i][0]:output_sz[0]-pad_sz[i][0], pad_sz[i][1]:, :, :] += BP_list[i]

    # multiply with the transpose: A^H * BP
    hf_out[0][k1] = hf_out1[k1] + (np.conj(samplesf[k1]) * BP)

    # B^H * BP
    fBP = [[]] * num_features
    fBP[k1] = (np.conj(init_hf[k1]) * BP).reshape((-1, init_hf[k1].shape[2]))

    # compute proj matrix part: B^H * A_m * f
    shBP = [[]] * num_features
    shBP[k1] = (np.conj(init_hf[k1]) * sh).reshape((-1, init_hf[k1].shape[2]))

    for i in block_inds:
        # multiply with the transpose: A^H * BP
        hf_out[0][i] = hf_out1[i] + (BP[pad_sz[i][0]:output_sz[0]-pad_sz[i][0], pad_sz[i][1]:, :, :] * np.conj(samplesf[i]))

        # B^H * BP
        fBP[i] = (np.conj(init_hf[i]) * BP[pad_sz[i][0]:output_sz[0]-pad_sz[i][0], pad_sz[i][1]:, :, :]).reshape((-1, init_hf[i].shape[2]))

        # compute proj matrix part: B^H * A_m * f
        shBP[i] = (np.conj(init_hf[i]) * sh[pad_sz[i][0]:output_sz[0]-pad_sz[i][0], pad_sz[i][1]:, :, :]).reshape((-1, init_hf[i].shape[2]))

    for i in range(0, num_features):
        fi = hf[i].shape[0] * (hf[i].shape[1] - 1) # index where the last frequency column starts

        # B^H * BP + \lambda \delta P
        hf_out2 = 2 * np.real(XH[i].dot(fBP[i]) - XH[i][:, fi:].dot(fBP[i][fi:, :])) + proj_reg * P[i]

        # compute proj matrix part: B^H * A_m * f
        hf_out[1][i] = hf_out2 + 2 * np.real(XH[i].dot(shBP[i]) - XH[i][:, fi:].dot(shBP[i][fi:, :]))
    return hf_out

def lhs_operation(hf, samplesf, reg_filter, sample_weights, use_gpu=False):
    """
        This is the left-hand-side operation in Conjugate Gradient
    """
    if use_gpu:
        raise(NotImplementedError)

    num_features = len(hf[0])
    filter_sz = np.zeros((num_features, 2), np.int32)
    for i in range(num_features):
        filter_sz[i, :] = np.array(hf[0][i].shape[:2])

    # index for the feature block with the largest spatial size
    k1 = np.argmax(filter_sz[:, 0])

    block_inds = list(range(0, num_features))
    block_inds.remove(k1)
    output_sz = np.array([hf[0][k1].shape[0], hf[0][k1].shape[1]*2-1])

    # compute the operation corresponding to the data term in the optimization 
    # implements: A.H diag(sample_weights) A f

    # sum over all features and feature blocks
    sh = np.matmul(hf[0][k1].transpose(0, 1, 3, 2), samplesf[k1])
    pad_sz = [[]] * num_features
    for i in block_inds:
        pad_sz[i] = ((output_sz - np.array([hf[0][i].shape[0], hf[0][i].shape[1]*2-1])) / 2).astype(np.int32)
        sh[pad_sz[i][0]:output_sz[0]-pad_sz[i][0], pad_sz[i][1]:, :, :] += np.matmul(hf[0][i].transpose(0, 1, 3, 2), samplesf[i])

    # weight all the samples
    sh = sample_weights.reshape(1, 1, 1, -1) * sh

    # multiply with the transpose
    hf_out = [[]] * num_features
    hf_out[k1] = np.matmul(np.conj(samplesf[k1]), sh.transpose(0, 1, 3, 2))
    for i in block_inds:
        hf_out[i] = np.matmul(np.conj(samplesf[i]), sh[pad_sz[i][0]:output_sz[0]-pad_sz[i][0], pad_sz[i][1]:, :, :].transpose(0, 1, 3, 2))

    # compute the operation corresponding to the regularization term (convolve each feature dimension
    # with the DFT of w, and the transposed operation) add the regularization part
    # W^H W f
    for i in range(num_features):
        reg_pad = min(reg_filter[i].shape[1] - 1, hf[0][i].shape[1]-1)

        # add part needed for convolution
        hf_conv = np.concatenate([hf[0][i], np.conj(np.rot90(hf[0][i][:, -reg_pad-1:-1, :], 2))], axis=1)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            # do first convolution
            hf_conv = convolve(hf_conv, reg_filter[i][:,:,np.newaxis,np.newaxis])
            # do final convolution and put together result
            hf_out[i] += convolve(hf_conv[:, :-reg_pad, :], reg_filter[i][:,:,np.newaxis,np.newaxis], 'valid')
    return [hf_out]

def train_joint(hf, projection_matrix, xlf, yf, reg_filter, sample_energy, reg_energy, proj_energy, init_CG_opts):
    params = settings.params

    if params["use_gpu"]:
        print("GPU")
        raise(NotImplementedError)

    lf_ind = [x.shape[0] * (x.shape[1]-1) for x in hf[0]]  # Index for the start of the last column of frequencies
    init_samplef = xlf
    init_samplef_H = [np.conj(x.reshape((-1, x.shape[2]))).T for x in init_samplef]

    # preconditioner
    diag_M = [[], []]
    diag_M[0] = [(1 - params['precond_reg_param']) * (params['precond_data_param']*m + (1 - params['precond_data_param']) * np.mean(m, 2, keepdims=True)) +
                 params['precond_reg_param'] * reg_energy_ for m, reg_energy_ in zip(sample_energy, reg_energy)]
    diag_M[1] = [params['precond_proj_param'] * (m + params['projection_reg']) for m in proj_energy]

    rhs_samplef = [[]] * len(hf[0])
    for iter in range(0, params['init_GN_iter']):
        init_samplef_proj = [np.matmul(P.T, x) for x, P in zip(init_samplef, projection_matrix)]  # project sample with new projection_matrix
        init_hf = hf[0]
        rhs_samplef[0] = [np.conj(x) * y[:, :, np.newaxis, np.newaxis] for x, y in zip(init_samplef_proj, yf)] # right-hand-side vector for filter
        # right hand side vector for the projection matrix part
        fyf = [np.reshape(np.conj(f) * y[:, :, np.newaxis, np.newaxis], (-1, f.shape[2])) for f, y in zip(hf[0], yf)]
        rhs_samplef[1] = [2 * np.real(XH.dot(fyf_) - XH[:, fi:].dot(fyf_[fi:, :])) - params['projection_reg'] * P
                         for P, XH, fyf_, fi in zip(projection_matrix, init_samplef_H, fyf, lf_ind)]

        hf[1] = [np.zeros_like(P) for P in projection_matrix]

        # CG
        hf, _, _ = pcg(lambda x: lhs_operation_joint(x, init_samplef_proj, reg_filter, init_samplef, init_samplef_H, init_hf, params["projection_reg"], params['use_gpu']), # A
                       rhs_samplef, # b
                       init_CG_opts,
                       lambda x: diag_precond(x, diag_M),  # M1
                       None, # M2
                       inner_product_joint,
                       hf)
        hf[0] = symmetrize_filter(hf[0])
        projection_matrix = [x + y for x, y in zip(projection_matrix, hf[1])]

    # extract filter
    hf = hf[0]
    return hf, projection_matrix

def inner_product_filter(xf, yf):
    """
        computes the inner product between two filters
    """

    ip = 0
    for i in range(len(xf[0])):
        ip += 2 * np.vdot(xf[0][i].flatten(), yf[0][i].flatten()) - np.vdot(xf[0][i][:, -1, :].flatten(), yf[0][i][:, -1, :].flatten())
    return np.real(ip)

def train_filter(hf, samplesf, yf, reg_filter, sample_weights, sample_energy, reg_energy, CG_opts, CG_state):
    """
        do conjugate graident optimization of the filter
    """
    params = settings.params

    if params['use_gpu']:
        raise(NotImplementedError)

    # construct the right hand side vector (A^H weight yf)
    rhs_samplef = [np.matmul(x, sample_weights) for x in samplesf]
    rhs_samplef = [(np.conj(x) * y[:,:,np.newaxis,np.newaxis])
                   for x, y in zip(rhs_samplef, yf)]

    # construct preconditioner
    diag_M = [(1 - params['precond_reg_param']) * (params['precond_data_param'] * m + (1-params['precond_data_param'])*np.mean(m, 2, keepdims=True)) +
              params['precond_reg_param'] * reg_energy_ for m, reg_energy_ in zip(sample_energy, reg_energy)]
    hf, _, CG_state = pcg(
        lambda x: lhs_operation(x, samplesf, reg_filter, sample_weights, params['use_gpu']), # A
        [rhs_samplef],  # b
        CG_opts,
        lambda x: diag_precond(x, [diag_M]),
        None,
        inner_product_filter,
        [hf],
        CG_state)
    return hf[0], CG_state
