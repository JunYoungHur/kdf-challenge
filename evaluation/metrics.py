import numpy as np


def eer(labels_spoof, scores):
    """EER where higher score = more likely spoof. labels_spoof: 1 = spoof, 0 = bona fide."""
    y = np.asarray(labels_spoof, bool)
    s = np.asarray(scores, float)
    ths = np.unique(s)
    far = np.array([(s[~y] >= t).mean() for t in ths])   # bona fide flagged as spoof
    frr = np.array([(s[y] < t).mean() for t in ths])     # spoof accepted as bona fide
    i = int(np.argmin(np.abs(far - frr)))
    return float((far[i] + frr[i]) / 2), float(ths[i])
