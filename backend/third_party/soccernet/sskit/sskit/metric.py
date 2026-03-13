import numpy as np
from scipy.optimize import linear_sum_assignment

def match(gt, est, th=0.25):
    dists = np.array([[((a - b)**2).sum() for a in gt] for b in est])
    rows, cols = linear_sum_assignment(dists)

    msk = np.sqrt(dists[rows, cols]) < th
    rows = rows[msk]
    cols = cols[msk]

    detected = len(rows)
    missed = len(gt) - detected
    extra = len(est) - detected
    distances = np.sqrt(dists[rows, cols])
    matches = zip(rows, cols)
    return detected, missed, extra, distances, matches
