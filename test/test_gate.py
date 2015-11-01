#!/usr/bin/python
#
# test_gate.py - Unit tests for gate module
#
# Authors: John T. Sexton (john.t.sexton@rice.edu)
#          Sebastian M. Castillo-Hair (smc9@rice.edu)
# Date:    10/31/2015
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
        
class TestDensity2dGate1(unittest.TestCase):
    
    def setUp(self):
        """
        Testing proper result of density gating.

        This function applied the density2d gate to Data003.fcs with a gating
        fraction of 0.3. The result is compared to the (previously calculated)
        output of gate.density2d at d23ec66f9039bbe104ff05ede0e3600b9a550078
        using the following command:
        fc.gate.density2d(fc.io.FCSData('Data003.fcs'),
                          channels = ['FSC', 'SSC'],
                          gate_fraction = 0.3)[0]
        """
        self.ungated_data = fc.io.FCSData('test/Data003.fcs')
        self.gated_data = np.load('test/Data003_gate_density2d.npy')

    def test_density2d(self):
        gated_data = fc.gate.density2d(self.ungated_data,
                                       channels = ['FSC', 'SSC'],
                                       gate_fraction = 0.3)
        np.testing.assert_array_equal(gated_data, self.gated_data)

class TestDensity2dGate2(unittest.TestCase):

    def setUp(self):
        """Set up data sets."""

        # "pyramid" with density peak at (2,2)
        d1 = [(x,y) for x in range(5) for y in range(5)]
        d1.extend([
            (2,2), (2,2),
            (2,1), (1,2), (2,3), (3,2)
            ])
        self.pyramid = np.array(d1)

        # "slope" with highest density at (4,4)
        d2 = []
        for idx in xrange(1,5):
            d2.extend([(x,y) for x in range(idx,5) for y in range(idx,5)])
        self.slope = np.array(d2)

    # Test normal use case behaviors

    def test_pyramid_1_gated_data_1(self):
        bins = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        np.testing.assert_array_equal(
            fc.gate.density2d(
                self.pyramid, bins=bins, gate_fraction=11.0/31, sigma=0.0),
            np.array([
                [1,2],
                [2,1],
                [2,2],
                [2,3],
                [3,2],
                [2,2],
                [2,2],
                [2,1],
                [1,2],
                [2,3],
                [3,2],
                ])
            )

    def test_pyramid_1_gated_data_2(self):
        bins = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        np.testing.assert_array_equal(
            fc.gate.density2d(
                self.pyramid, bins=bins, gate_fraction=11.0/31, sigma=0.0,
                full_output=True).gated_data,
            np.array([
                [1,2],
                [2,1],
                [2,2],
                [2,3],
                [3,2],
                [2,2],
                [2,2],
                [2,1],
                [1,2],
                [2,3],
                [3,2],
                ])
            )

    def test_pyramid_1_mask(self):
        bins = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        np.testing.assert_array_equal(
            fc.gate.density2d(
                self.pyramid, bins=bins, gate_fraction=11.0/31, sigma=0.0,
                full_output=True).mask,
            np.array([0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,1,0,0,0,0,0,0,0,
                1,1,1,1,1,1], dtype=bool)
            )

    def test_pyramid_2_gated_data_1(self):
        bins = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        np.testing.assert_array_equal(
            fc.gate.density2d(
                self.pyramid, bins=bins, gate_fraction=3.0/31, sigma=0.0),
            np.array([
                [2,2],
                [2,2],
                [2,2],
                ])
            )

    def test_pyramid_2_gated_data_2(self):
        bins = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        np.testing.assert_array_equal(
            fc.gate.density2d(
                self.pyramid, bins=bins, gate_fraction=3.0/31, sigma=0.0,
                full_output=True).gated_data,
            np.array([
                [2,2],
                [2,2],
                [2,2],
                ])
            )

    def test_pyramid_2_mask(self):
        bins = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        np.testing.assert_array_equal(
            fc.gate.density2d(
                self.pyramid, bins=bins, gate_fraction=3.0/31, sigma=0.0,
                full_output=True).mask,
            np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,
                1,1,0,0,0,0], dtype=bool)
            )

    def test_slope_1_gated_data_1(self):
        bins = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        np.testing.assert_array_equal(
            fc.gate.density2d(
                self.slope, bins=bins, gate_fraction=4.0/30, sigma=0.0),
            np.array([
                [4,4],
                [4,4],
                [4,4],
                [4,4],
                ])
            )

    def test_slope_1_gated_data_2(self):
        bins = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        np.testing.assert_array_equal(
            fc.gate.density2d(
                self.slope, bins=bins, gate_fraction=4.0/30, sigma=0.0,
                full_output=True).gated_data,
            np.array([
                [4,4],
                [4,4],
                [4,4],
                [4,4],
                ])
            )

    def test_slope_1_mask(self):
        bins = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        np.testing.assert_array_equal(
            fc.gate.density2d(
                self.slope, bins=bins, gate_fraction=4.0/30, sigma=0.0,
                full_output=True).mask,
            np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,
                0,0,0,1,1], dtype=bool)
            )

    def test_slope_2_gated_data_1(self):
        bins = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        np.testing.assert_array_equal(
            fc.gate.density2d(
                self.slope, bins=bins, gate_fraction=13.0/30, sigma=0.0),
            np.array([
                [3,3],
                [3,4],
                [4,3],
                [4,4],
                [3,3],
                [3,4],
                [4,3],
                [4,4],
                [3,3],
                [3,4],
                [4,3],
                [4,4],
                [4,4],
                ])
            )

    def test_slope_2_gated_data_2(self):
        bins = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        np.testing.assert_array_equal(
            fc.gate.density2d(
                self.slope, bins=bins, gate_fraction=13.0/30, sigma=0.0,
                full_output=True).gated_data,
            np.array([
                [3,3],
                [3,4],
                [4,3],
                [4,4],
                [3,3],
                [3,4],
                [4,3],
                [4,4],
                [3,3],
                [3,4],
                [4,3],
                [4,4],
                [4,4],
                ])
            )

    def test_slope_2_mask(self):
        bins = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        np.testing.assert_array_equal(
            fc.gate.density2d(
                self.slope, bins=bins, gate_fraction=13.0/30, sigma=0.0,
                full_output=True).mask,
            np.array([0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,1,1,0,1,1,
                1,1,1,1,1], dtype=bool)
            )

    # Confirm everything gets through with 1.0 gate_fraction

    def test_gate_fraction_1_gated_data_1(self):
        bins = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        np.testing.assert_array_equal(
            fc.gate.density2d(
                self.pyramid, bins=bins, gate_fraction=1.0, sigma=0.0),
            self.pyramid
            )

    def test_gate_fraction_1_gated_data_2(self):
        bins = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        np.testing.assert_array_equal(
            fc.gate.density2d(
                self.pyramid, bins=bins, gate_fraction=1.0, sigma=0.0,
                full_output=True).gated_data,
            self.pyramid
            )

    def test_gate_fraction_1_mask(self):
        bins = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        np.testing.assert_array_equal(
            fc.gate.density2d(
                self.pyramid, bins=bins, gate_fraction=1.0, sigma=0.0,
                full_output=True).mask,
            np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                1,1,1,1,1,1], dtype=bool)
            )

    # Confirm nothing gets through with 0.0 gate_fraction

    def test_gate_fraction_2_gated_data_1(self):
        bins = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        np.testing.assert_array_equal(
            fc.gate.density2d(
                self.pyramid, bins=bins, gate_fraction=0.0, sigma=0.0),
            np.array([])
            )

    def test_gate_fraction_2_gated_data_2(self):
        bins = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        np.testing.assert_array_equal(
            fc.gate.density2d(
                self.pyramid, bins=bins, gate_fraction=0.0, sigma=0.0,
                full_output=True).gated_data,
            np.array([])
            )

    def test_gate_fraction_2_mask(self):
        bins = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        np.testing.assert_array_equal(
            fc.gate.density2d(
                self.pyramid, bins=bins, gate_fraction=0.0, sigma=0.0,
                full_output=True).mask,
            np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                0,0,0,0,0,0], dtype=bool)
            )

if __name__ == '__main__':
    unittest.main()
