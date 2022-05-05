"""
Copulas adapted from from https://jsdajournal.springeropen.com/articles/10.1186/s40488-021-00118-z
"""
import numpy as np

def copula_clip(u1, u2, vec):

    upper = np.min((u1, u2), axis=0)
    lower = np.clip(u1 + u2 - 1, 0, None)
    return np.clip(vec, lower, upper)

def cqq(q1, q2, r):
    """
    The basic copula. 
    :param q1:  probability or np-array of probabilities of event 1 not occurring
    :param q2: probability or np-array of probabilities of event 2 not occurring
    :param r: correlation of event 1 and 2
    :return: joint probability of neither event 1 or 2 occuring
    """
    np.clip(r, -1, 1)
    p1, p2 = 1 - q1, 1 - q2
    return_vec = 1 - p1 - p2 + p1 * p2 + r * np.sqrt(p1 * p2 * q1 * q2)
    return_vec = copula_clip(q1, q2, return_vec)
    return return_vec


def Copula2d(p1,p2,r):
    """
    :param p1: probability or np-array of probabilities of event 1
    :param p2: probability or np-array of probabilities of event 2
    :param r: correlation of event 1 and 2
    :return: joint probability at least one event occurring
        """
    q1,q2 = 1-p1, 1-p2
    return_vec =  1 - cqq(q1, q2, r)
    return return_vec

def Copula3d(p1, p2, p3, r1, r2, r3):
    """
    Note that this can throw up an error if p2 = 0. This error does not effect the results.
    :param p1: probability or np-array of probabilities of event 1
    :param p2: probability or np-array of probabilities of event 2
    :param p3: probability or np-array of probabilities of event 3
    :param r1: correlation of event 1 and 2
    :param r2: correlation of event 2 and 3
    :param r3: correlation of event 1 and 3 given that event 2 did not occur
    :return: joint probability at least one event occurring
    """
    q1, q2, q3 = 1 - p1, 1 - p2, 1 - p3

    C12 = cqq(q1, q2, r1)
    C23 = cqq(q2, q3, r2)

    q10 = C12 / q2
    q30 = C23 / q2
    C130 = cqq(q10, q30, r3)

    return_vec = 1 - q2 * C130
    return_vec = np.where(np.isclose(q2, 0.0), 1.0, return_vec)
    return return_vec
    return_vec = np.where(np.isclose(q2, 0.0), 1.0, return_vec)
    return return_vec
