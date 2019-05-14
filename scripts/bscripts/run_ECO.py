from config import *
import scripts.butil
import sys
import time
import numpy as np

from .ECO import ECO

def run_ECO(seq, rp, bSaveImage):
    seq_path = seq.path.split("/")[:-2]
    seq_path = "/".join(seq_path)

    tic = time.clock()
    ECO(seq_path)
    duration = time.clock() - tic

    result = dict()
    res = np.loadtxt('/home/jkamelin/Documents/tracker_benchmark/trackers/ECO/ECO.txt', dtype=int)
    result['res'] = res.tolist()
    result['type'] = 'rect'
    result['fps'] = round(seq.len / duration, 3)
    return result