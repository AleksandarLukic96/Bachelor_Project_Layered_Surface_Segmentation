import itertools
import numpy as np
import os
import pytest
from slgbuilder import GraphObject

''' 
These tests are made to ensure the functionality of the slgbuilder tools.
In this test file, we test the GraphObject class.
As the class is used as a holder for graph data,
we ensure that any graph of dimension n>2 is valid.
'''

# Create tuple for shape of array
t = tuple()
dim = np.random.randint(2, 10)
for d in range(0, dim):
    t += (d+1,)

# Creating a n-dimensional array, based on shape
arr = np.ones(shape = [*t], dtype = int)
for idx in itertools.product(*[range(s) for s in arr.shape]):
    arr[idx] = np.random.randint(0, 255+1)

# Create GraphObject by parsing the n-D data, leaving the remaining fields empty.
g_obj = GraphObject(
            data = arr, 
            sample_points = None, 
            block_ids = None
        )

# Testing dimensions with numpy.shape()
def test_graphobject_data_shape():
    assert arr.shape == g_obj.data.shape == t
    
def test_graphobject_samplepoints_shape():
    assert arr.shape + (len(arr.shape),) == g_obj.sample_points.shape == (t + (len(arr.shape),))
