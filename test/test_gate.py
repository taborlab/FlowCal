#!/usr/bin/python
#
# test_gate.py - Unit tests for gate module
#
# Author: John T. Sexton (john.t.sexton@rice.edu)
# Date: 5/30/2015
#
# Requires:
#   * gate
#   * numpy
#
# Note: fc.io clashes with native python io module and will cause this set of
# unit tests to fail inside of fc/ folder.

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
        self.d = np.array([
            [4,4],
            [4,5],
            [4,6],
            [5,4],
            [5,5],
            [5,6],
            [6,4],
            [6,5],
            [6,6],
            [5,5],
            ])

    def test_circular_median_1(self):
        np.testing.assert_array_equal(
            gate.circular_median(self.d,
                gate_fraction=10.0/10)[0],
            np.array([1,1,1,1,1,1,1,1,1,1],dtype=bool)
            )

    def test_circular_median_2(self):
        np.testing.assert_array_equal(
            gate.circular_median(self.d,
                gate_fraction=2.0/10)[0],
            np.array([0,0,0,0,1,0,0,0,0,1],dtype=bool)
            )

    def test_circular_median_3(self):
        np.testing.assert_array_equal(
            gate.circular_median(self.d,
                gate_fraction=6.0/10)[0],
            np.array([0,1,0,1,1,1,0,1,0,1],dtype=bool)
            )

    def test_circular_median_4(self):
        np.testing.assert_array_equal(
            gate.circular_median(self.d,
                gate_fraction=0.0/10)[0],
            np.array([0,0,0,0,0,0,0,0,0,0],dtype=bool)
            )

class TestStartStopGate(unittest.TestCase):
    def test_start_stop_1(self):
        d = np.zeros(shape=349,dtype=bool)
        self.assertRaises(ValueError, gate.start_stop, d)
        
    def test_start_stop_2(self):
        d = np.zeros(shape=350,dtype=bool)
        np.testing.assert_array_equal(gate.start_stop(d), d)
        
    def test_start_stop_3(self):
        d = np.zeros(shape=351,dtype=bool)
        d[250] = True
        np.testing.assert_array_equal(gate.start_stop(d), d)
        
if __name__ == '__main__':
    unittest.main()
