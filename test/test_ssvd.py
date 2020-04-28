# -*- coding: utf-8 -*-
import sys
sys.path.append("..")

from ssvd_pkg.ssvd import pos_new, sign_new
import numpy as np

def test_pos():
    a = np.array([1, -1, 1, 0])
    assert np.allclose(pos_new(a), np.array([1, 0, 1, 0]))
    print("Success")
    
def test_sign():
    a = np.array([2, 3, 0, -2, -3])
    assert np.allclose(sign_new(a), np.array([1, 1, 0, -1, -1]))
    print("Success")
    
test_pos()
test_sign()