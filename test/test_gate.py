#!/usr/bin/python
#
# test_gate.py - Unit tests for gate module
#
# Authors: John T. Sexton (john.t.sexton@rice.edu)
#          Sebastian M. Castillo-Hair (smc9@rice.edu)
# Date: 5/30/2015
#
# Requires:
#   * fc.gate
#   * numpy
#
# Note: fc.io clashes with native python io module and will cause this set of
# unit tests to fail inside of fc/ folder.

import fc.gate
import numpy as np
import unittest

class TestStartEndGate(unittest.TestCase):
    
    def setUp(self):
        self.d = np.array([
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

    def test_start_end(self):
        np.testing.assert_array_equal(
            fc.gate.start_end(self.d, num_start = 2, num_end = 3),
            self.d[np.array([0,0,1,1,1,1,1,0,0,0], dtype = bool)]
            )

    def test_start_end_error(self):
        with self.assertRaises(ValueError):
            fc.gate.start_end(self.d, num_start = 5, num_end = 7)

class TestHighLowGate(unittest.TestCase):
    
    def setUp(self):
        self.d1 = np.array([
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
        self.d2 = np.array([
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

    # Test 1D data with combinations of high and low values

    def test_high_low_1d_1(self):
        np.testing.assert_array_equal(
            fc.gate.high_low(self.d1, high=10, low=1),
            np.array([[2,3,4,5,6,7,8,9]]).T
            )

    def test_high_low_1d_2(self):
        np.testing.assert_array_equal(
            fc.gate.high_low(self.d1, high=11, low=0),
            np.array([[1,2,3,4,5,6,7,8,9,10]]).T
            )

    # Test multi-dimensional data with combinations of high and low values

    def test_high_low_2d_1(self):
        np.testing.assert_array_equal(
            fc.gate.high_low(self.d2, high=10, low=1),
            self.d2[np.array([0,1,1,0,0,1,1,1,0,0], dtype=bool)]
            )

    def test_high_low_2d_2(self):
        np.testing.assert_array_equal(
            fc.gate.high_low(self.d2, high=11, low=1),
            self.d2[np.array([0,1,1,1,0,1,1,1,1,0], dtype=bool)]
            )

    def test_high_low_2d_3(self):
        np.testing.assert_array_equal(
            fc.gate.high_low(self.d2, high=10, low=0),
            self.d2[np.array([1,1,1,0,1,1,1,1,0,0], dtype=bool)]
            )

    def test_high_low_2d_4(self):
        np.testing.assert_array_equal(
            fc.gate.high_low(self.d2, high=11, low=0),
            self.d2[np.array([1,1,1,1,1,1,1,1,1,1], dtype=bool)]
            )

    def test_high_low_2d_5(self):
        np.testing.assert_array_equal(
            fc.gate.high_low(self.d2, channels = 0, high=10, low=1),
            self.d2[np.array([0,1,1,1,1,1,1,1,1,0], dtype=bool)]
            )

    def test_high_low_2d_6(self):
        np.testing.assert_array_equal(
            fc.gate.high_low(self.d2, channels = 0),
            self.d2[np.array([1,1,1,1,1,1,1,1,1,1], dtype=bool)]
            )
        
class TestDensity2dGating(unittest.TestCase):
    
    def setUp(self):
        '''
        Testing proper result of density gating.

        This function applied the density2d gate to Data003.fcs with a gating
        fraction of 0.3. The result is compared to the (previously calculated)
        output of gate.density2d at d23ec66f9039bbe104ff05ede0e3600b9a550078
        using the following command:
        fc.gate.density2d(fc.io.FCSData('Data003.fcs'),
                          channels = ['FSC', 'SSC'],
                          gate_fraction = 0.3)[0]
        '''
        self.ungated_data = fc.io.FCSData('test/Data003.fcs')
        self.gated_data = np.load('test/Data003_gate_density2d.npy')

    def test_density2d(self):
        gated_data = fc.gate.density2d(self.ungated_data,
                                       channels = ['FSC', 'SSC'],
                                       gate_fraction = 0.3)[0]
        np.testing.assert_array_equal(gated_data, self.gated_data)

if __name__ == '__main__':
    unittest.main()
    