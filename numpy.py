# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 09:26:28 2020

@author: Bharanitharan
"""

import numpy as np
import os

a = np.arange(15).reshape(3,5)

a

a.type()
a.shape
a.ndim
a[1]

np.save('out.txt')

os.getcwd()

np.save('txt.txt',a)

pwd

