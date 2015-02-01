#!/usr/bin/python
#
# test_gate.py - Unit tests for gate module
#
# Author: John T. Sexton (john.t.sexton@rice.edu)
# Date: 2/1/2015
#
# Requires:
#   * gate
#   * numpy

import gate
import numpy as np
import unittest

class TestSimpleGates(unittest.TestCase):
    
    def setUp(self):
        self.d1 = np.array([
            [1, 7, 2],
            [2, 8, 3],
            [3, 9, 4],
            [4, 10, 5],
            [5, 1, 6],
            [6, 2, 7],
            [7, 3, 8],
            [8, 4, 9],
            [9, 5, 10],
            [10, 6, 1],
            ])
        self.d2 = np.array([
            [1],
            [2],
            [3],
            [4],
            [5],
            [6],
            [7],
            [8],
            [9],
            [10],
            ])


    #
    # Test high_low gate
    #

    # Test multi-dimensional data with combinations of high and low values

    def test_high_low_2d_1(self):
        np.testing.assert_array_equal(
            gate.high_low(self.d1,10,1),
            np.array([0,1,1,0,0,1,1,1,0,0],dtype=bool)
            )

    def test_high_low_2d_2(self):
        np.testing.assert_array_equal(
            gate.high_low(self.d1,11,1),
            np.array([0,1,1,1,0,1,1,1,1,0],dtype=bool)
            )

    def test_high_low_2d_3(self):
        np.testing.assert_array_equal(
            gate.high_low(self.d1,10,0),
            np.array([1,1,1,0,1,1,1,1,0,0],dtype=bool)
            )

    def test_high_low_2d_4(self):
        np.testing.assert_array_equal(
            gate.high_low(self.d1,11,0),
            np.array([1,1,1,1,1,1,1,1,1,1],dtype=bool)
            )

    # Test 1D data with combinations of high and low values

    def test_high_low_1d_1(self):
        np.testing.assert_array_equal(
            gate.high_low(self.d2,10,1),
            np.array([0,1,1,1,1,1,1,1,1,0],dtype=bool)
            )

    def test_high_low_1d_2(self):
        np.testing.assert_array_equal(
            gate.high_low(self.d2,11,0),
            np.array([1,1,1,1,1,1,1,1,1,1],dtype=bool)
            )

    #
    # Test extrema gate
    #

    # Test multi-dimensional data with combinations of extrema

    def test_extrema_2d_1(self):
        np.testing.assert_array_equal(
            gate.extrema(self.d1,[10,1]),
            np.array([0,1,1,0,0,1,1,1,0,0],dtype=bool)
            )

    def test_extrema_2d_2(self):
        np.testing.assert_array_equal(
            gate.extrema(self.d1,[11,1]),
            np.array([0,1,1,1,0,1,1,1,1,0],dtype=bool)
            )

    def test_extrema_2d_3(self):
        np.testing.assert_array_equal(
            gate.extrema(self.d1,[10,0]),
            np.array([1,1,1,0,1,1,1,1,0,0],dtype=bool)
            )

    def test_extrema_2d_4(self):
        np.testing.assert_array_equal(
            gate.extrema(self.d1,[11,0]),
            np.array([1,1,1,1,1,1,1,1,1,1],dtype=bool)
            )

    def test_extrema_2d_5(self):
        np.testing.assert_array_equal(
            gate.extrema(self.d1,[]),
            np.array([1,1,1,1,1,1,1,1,1,1],dtype=bool)
            )

    def test_extrema_2d_6(self):
        np.testing.assert_array_equal(
            gate.extrema(self.d1,[1,5,10]),
            np.array([0,1,1,0,0,1,1,1,0,0],dtype=bool)
            )

    # Test 1D data with combinations of extrema

    def test_extrema_1d_1(self):
        np.testing.assert_array_equal(
            gate.extrema(self.d2,[10,1]),
            np.array([0,1,1,1,1,1,1,1,1,0],dtype=bool)
            )

    def test_extrema_1d_2(self):
        np.testing.assert_array_equal(
            gate.extrema(self.d2,[11,0]),
            np.array([1,1,1,1,1,1,1,1,1,1],dtype=bool)
            )

    def test_extrema_1d_3(self):
        np.testing.assert_array_equal(
            gate.extrema(self.d2,[]),
            np.array([1,1,1,1,1,1,1,1,1,1],dtype=bool)
            )

    def test_extrema_1d_4(self):
        np.testing.assert_array_equal(
            gate.extrema(self.d2,[1,5,10]),
            np.array([0,1,1,1,0,1,1,1,1,0],dtype=bool)
            )

class TestCircularMedianGate(unittest.TestCase):
    def setUp(self):
        pass
    def test_circular_median(self):
        pass

if __name__ == '__main__':
    unittest.main()
