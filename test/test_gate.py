"""
`gate` module unit tests.

"""

import FlowCal.gate
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

    def test_gated_data_1(self):
        np.testing.assert_array_equal(
            FlowCal.gate.start_end(self.d, num_start=2, num_end=3),
            np.array([
                [3, 9, 4],
                [4, 10, 5],
                [5, 1, 6],
                [6, 2, 7],
                [7, 3, 8],
                ])
            )

    def test_gated_data_2(self):
        np.testing.assert_array_equal(
            FlowCal.gate.start_end(
                self.d, num_start=2, num_end=3, full_output=True).gated_data,
            np.array([
                [3, 9, 4],
                [4, 10, 5],
                [5, 1, 6],
                [6, 2, 7],
                [7, 3, 8],
                ])
            )

    def test_mask(self):
        np.testing.assert_array_equal(
            FlowCal.gate.start_end(
                self.d, num_start=2, num_end=3, full_output=True).mask,
            np.array([0,0,1,1,1,1,1,0,0,0], dtype=bool)
            )

    def test_0_end_gated_data_1(self):
        np.testing.assert_array_equal(
            FlowCal.gate.start_end(self.d, num_start=2, num_end=0),
            np.array([
                [3, 9, 4],
                [4, 10, 5],
                [5, 1, 6],
                [6, 2, 7],
                [7, 3, 8],
                [8, 4, 9],
                [9, 5, 10],
                [10, 6, 1],
                ])
            )

    def test_0_end_gated_data_2(self):
        np.testing.assert_array_equal(
            FlowCal.gate.start_end(
                self.d, num_start=2, num_end=0, full_output=True).gated_data,
            np.array([
                [3, 9, 4],
                [4, 10, 5],
                [5, 1, 6],
                [6, 2, 7],
                [7, 3, 8],
                [8, 4, 9],
                [9, 5, 10],
                [10, 6, 1],
                ])
            )

    def test_0_end_mask(self):
        np.testing.assert_array_equal(
            FlowCal.gate.start_end(
                self.d, num_start=2, num_end=0, full_output=True).mask,
            np.array([0,0,1,1,1,1,1,1,1,1], dtype=bool)
            )

    def test_error(self):
        with self.assertRaises(ValueError):
            FlowCal.gate.start_end(self.d, num_start=5, num_end=7)

class TestHighLowGate(unittest.TestCase):
    
    def setUp(self):
        self.d1 = np.array([list(range(1,11))]).T
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

    ###
    # Test 1D data with combinations of high and low values
    ###

    def test_1d_1_gated_data_1(self):
        np.testing.assert_array_equal(
            FlowCal.gate.high_low(self.d1, high=8, low=2),
            np.array([[3,4,5,6,7]]).T
            )

    def test_1d_1_gated_data_2(self):
        np.testing.assert_array_equal(
            FlowCal.gate.high_low(
                self.d1, high=8, low=2, full_output=True).gated_data,
            np.array([[3,4,5,6,7]]).T
            )

    def test_1d_1_mask(self):
        np.testing.assert_array_equal(
            FlowCal.gate.high_low(self.d1,
                                  high=8,
                                  low=2,
                                  full_output=True).mask,
            np.array([0,0,1,1,1,1,1,0,0,0], dtype=bool)
            )

    def test_1d_2_gated_data_1(self):
        np.testing.assert_array_equal(
            FlowCal.gate.high_low(self.d1, high=11, low=0),
            np.array([[1,2,3,4,5,6,7,8,9,10]]).T
            )

    def test_1d_2_gated_data_2(self):
        np.testing.assert_array_equal(
            FlowCal.gate.high_low(
                self.d1, high=11, low=0, full_output=True).gated_data,
            np.array([[1,2,3,4,5,6,7,8,9,10]]).T
            )

    def test_1d_2_mask(self):
        np.testing.assert_array_equal(
            FlowCal.gate.high_low(self.d1,
                                  high=11,
                                  low=0,
                                  full_output=True).mask,
            np.array([1,1,1,1,1,1,1,1,1,1], dtype=bool)
            )

    # Test that defaults allow all data through

    def test_1d_defaults_gated_data_1(self):
        np.testing.assert_array_equal(
            FlowCal.gate.high_low(self.d1),
            np.array([[1,2,3,4,5,6,7,8,9,10]]).T
            )

    def test_1d_defaults_gated_data_2(self):
        np.testing.assert_array_equal(
            FlowCal.gate.high_low(self.d1, full_output=True).gated_data,
            np.array([[1,2,3,4,5,6,7,8,9,10]]).T
            )

    def test_1d_defaults_mask(self):
        np.testing.assert_array_equal(
            FlowCal.gate.high_low(self.d1, full_output=True).mask,
            np.array([1,1,1,1,1,1,1,1,1,1], dtype=bool)
            )

    ###
    # Test multi-dimensional data with combinations of high and low values
    ###

    def test_2d_1_gated_data_1(self):
        np.testing.assert_array_equal(
            FlowCal.gate.high_low(self.d2, high=10, low=1),
            np.array([
                [2, 8, 3],
                [3, 9, 4],
                [6, 2, 7],
                [7, 3, 8],
                [8, 4, 9],
                ])
            )

    def test_2d_1_gated_data_2(self):
        np.testing.assert_array_equal(
            FlowCal.gate.high_low(
                self.d2, high=10, low=1, full_output=True).gated_data,
            np.array([
                [2, 8, 3],
                [3, 9, 4],
                [6, 2, 7],
                [7, 3, 8],
                [8, 4, 9],
                ])
            )

    def test_2d_1_mask(self):
        np.testing.assert_array_equal(
            FlowCal.gate.high_low(self.d2,
                                  high=10,
                                  low=1,
                                  full_output=True).mask,
            np.array([0,1,1,0,0,1,1,1,0,0], dtype=bool)
            )

    def test_2d_2_gated_data_1(self):
        np.testing.assert_array_equal(
            FlowCal.gate.high_low(self.d2, high=11, low=1),
            np.array([
                [2, 8, 3],
                [3, 9, 4],
                [4, 10, 5],
                [6, 2, 7],
                [7, 3, 8],
                [8, 4, 9],
                [9, 5, 10],
                ])
            )

    def test_2d_2_gated_data_2(self):
        np.testing.assert_array_equal(
            FlowCal.gate.high_low(self.d2,
                                  high=11,
                                  low=1,
                                  full_output=True).gated_data,
            np.array([
                [2, 8, 3],
                [3, 9, 4],
                [4, 10, 5],
                [6, 2, 7],
                [7, 3, 8],
                [8, 4, 9],
                [9, 5, 10],
                ])
            )

    def test_2d_2_mask(self):
        np.testing.assert_array_equal(
            FlowCal.gate.high_low(self.d2,
                                  high=11,
                                  low=1,
                                  full_output=True).mask,
            np.array([0,1,1,1,0,1,1,1,1,0], dtype=bool)
            )

    def test_2d_3_gated_data_1(self):
        np.testing.assert_array_equal(
            FlowCal.gate.high_low(self.d2, high=10, low=0),
            np.array([
                [1, 7, 2],
                [2, 8, 3],
                [3, 9, 4],
                [5, 1, 6],
                [6, 2, 7],
                [7, 3, 8],
                [8, 4, 9],
                ])
            )

    def test_2d_3_gated_data_2(self):
        np.testing.assert_array_equal(
            FlowCal.gate.high_low(
                self.d2, high=10, low=0, full_output=True).gated_data,
            np.array([
                [1, 7, 2],
                [2, 8, 3],
                [3, 9, 4],
                [5, 1, 6],
                [6, 2, 7],
                [7, 3, 8],
                [8, 4, 9],
                ])
            )

    def test_2d_3_mask(self):
        np.testing.assert_array_equal(
            FlowCal.gate.high_low(self.d2,
                                  high=10,
                                  low=0,
                                  full_output=True).mask,
            np.array([1,1,1,0,1,1,1,1,0,0], dtype=bool)
            )

    def test_2d_4_gated_data_1(self):
        np.testing.assert_array_equal(
            FlowCal.gate.high_low(self.d2, high=11, low=0),
            np.array([
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
            )

    def test_2d_4_gated_data_2(self):
        np.testing.assert_array_equal(
            FlowCal.gate.high_low(
                self.d2, high=11, low=0, full_output=True).gated_data,
            np.array([
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
            )

    def test_2d_4_mask(self):
        np.testing.assert_array_equal(
            FlowCal.gate.high_low(self.d2,
                                  high=11,
                                  low=0,
                                  full_output=True).mask,
            np.array([1,1,1,1,1,1,1,1,1,1], dtype=bool)
            )

    def test_2d_5_gated_data_1(self):
        np.testing.assert_array_equal(
            FlowCal.gate.high_low(self.d2, high=9, low=2),
            np.array([
                [7, 3, 8],
                ])
            )

    def test_2d_5_gated_data_2(self):
        np.testing.assert_array_equal(
            FlowCal.gate.high_low(
                self.d2, high=9, low=2, full_output=True).gated_data,
            np.array([
                [7, 3, 8],
                ])
            )

    def test_2d_5_mask(self):
        np.testing.assert_array_equal(
            FlowCal.gate.high_low(self.d2,
                                  high=9,
                                  low=2,
                                  full_output=True).mask,
            np.array([0,0,0,0,0,0,1,0,0,0], dtype=bool)
            )

    def test_2d_6_gated_data_1(self):
        np.testing.assert_array_equal(
            FlowCal.gate.high_low(self.d2, channels=1, high=9, low=2),
            np.array([
                [1, 7, 2],
                [2, 8, 3],
                [7, 3, 8],
                [8, 4, 9],
                [9, 5, 10],
                [10, 6, 1],
                ])
            )

    def test_2d_6_gated_data_2(self):
        np.testing.assert_array_equal(
            FlowCal.gate.high_low(
                self.d2, channels=1, high=9, low=2, full_output=True
                ).gated_data,
            np.array([
                [1, 7, 2],
                [2, 8, 3],
                [7, 3, 8],
                [8, 4, 9],
                [9, 5, 10],
                [10, 6, 1],
                ])
            )

    def test_2d_6_mask(self):
        np.testing.assert_array_equal(
            FlowCal.gate.high_low(
                self.d2, channels=1, high=9, low=2, full_output=True).mask,
            np.array([1,1,0,0,0,0,1,1,1,1], dtype=bool)
            )

    # Test channels

    def test_2d_channels_gated_data_1(self):
        np.testing.assert_array_equal(
            FlowCal.gate.high_low(self.d2, channels=0, high=10, low=1),
            np.array([
                [2, 8, 3],
                [3, 9, 4],
                [4, 10, 5],
                [5, 1, 6],
                [6, 2, 7],
                [7, 3, 8],
                [8, 4, 9],
                [9, 5, 10],
                ])
            )

    def test_2d_channels_gated_data_2(self):
        np.testing.assert_array_equal(
            FlowCal.gate.high_low(
                self.d2, channels=0, high=10, low=1, full_output=True
                ).gated_data,
            np.array([
                [2, 8, 3],
                [3, 9, 4],
                [4, 10, 5],
                [5, 1, 6],
                [6, 2, 7],
                [7, 3, 8],
                [8, 4, 9],
                [9, 5, 10],
                ])
            )

    def test_2d_channels_mask(self):
        np.testing.assert_array_equal(
            FlowCal.gate.high_low(
                self.d2, channels=0, high=10, low=1, full_output=True).mask,
            np.array([0,1,1,1,1,1,1,1,1,0], dtype=bool)
            )

    # Test that defaults allow all data through

    def test_2d_defaults_1_gated_data_1(self):
        np.testing.assert_array_equal(
            FlowCal.gate.high_low(self.d2),
            np.array([
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
            )

    def test_2d_defaults_1_gated_data_2(self):
        np.testing.assert_array_equal(
            FlowCal.gate.high_low(self.d2, full_output=True).gated_data,
            np.array([
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
            )

    def test_2d_defaults_1_mask(self):
        np.testing.assert_array_equal(
            FlowCal.gate.high_low(self.d2, full_output=True).mask,
            np.array([1,1,1,1,1,1,1,1,1,1], dtype=bool)
            )

    def test_2d_defaults_2_gated_data_1(self):
        np.testing.assert_array_equal(
            FlowCal.gate.high_low(self.d2, channels=0),
            np.array([
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
            )

    def test_2d_defaults_2_gated_data_2(self):
        np.testing.assert_array_equal(
            FlowCal.gate.high_low(
                self.d2, channels=0, full_output=True).gated_data,
            np.array([
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
            )

    def test_2d_defaults_2_mask(self):
        np.testing.assert_array_equal(
            FlowCal.gate.high_low(self.d2,
                                  channels=0,
                                  full_output=True).mask,
            np.array([1,1,1,1,1,1,1,1,1,1], dtype=bool)
            )
        
class TestDensity2dGate1(unittest.TestCase):
    
    def setUp(self):
        """
        Testing proper result of density gating.

        This function applied the density2d gate to Data003.fcs with a gating
        fraction of 0.3. The result is compared to the (previously calculated)
        output of gate.density2d at d23ec66f9039bbe104ff05ede0e3600b9a550078
        using the following command:
        FlowCal.gate.density2d(FlowCal.io.FCSData('Data003.fcs'),
                               channels = ['FSC', 'SSC'],
                               gate_fraction = 0.3)[0]
        """
        self.ungated_data = FlowCal.io.FCSData('test/Data003.fcs')
        self.gated_data = np.load('test/Data003_gate_density2d.npy')

    def test_density2d(self):
        gated_data = FlowCal.gate.density2d(self.ungated_data,
                                            channels = ['FSC', 'SSC'],
                                            gate_fraction = 0.3,
                                            xscale='linear',
                                            yscale='linear')
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
        for idx in range(1,5):
            d2.extend([(x,y) for x in range(idx,5) for y in range(idx,5)])
        self.slope = np.array(d2)

    ###
    # Test normal use case behaviors
    ###

    def test_pyramid_1_gated_data_1(self):
        bins = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        np.testing.assert_array_equal(
            FlowCal.gate.density2d(
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
            FlowCal.gate.density2d(
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
            FlowCal.gate.density2d(
                self.pyramid, bins=bins, gate_fraction=11.0/31, sigma=0.0,
                full_output=True).mask,
            np.array([0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,1,0,0,0,0,0,0,0,
                1,1,1,1,1,1], dtype=bool)
            )

    def test_pyramid_2_gated_data_1(self):
        bins = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        np.testing.assert_array_equal(
            FlowCal.gate.density2d(
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
            FlowCal.gate.density2d(
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
            FlowCal.gate.density2d(
                self.pyramid, bins=bins, gate_fraction=3.0/31, sigma=0.0,
                full_output=True).mask,
            np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,
                1,1,0,0,0,0], dtype=bool)
            )

    def test_slope_1_gated_data_1(self):
        bins = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        np.testing.assert_array_equal(
            FlowCal.gate.density2d(
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
            FlowCal.gate.density2d(
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
            FlowCal.gate.density2d(
                self.slope, bins=bins, gate_fraction=4.0/30, sigma=0.0,
                full_output=True).mask,
            np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,
                0,0,0,1,1], dtype=bool)
            )

    def test_slope_2_gated_data_1(self):
        bins = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        np.testing.assert_array_equal(
            FlowCal.gate.density2d(
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
            FlowCal.gate.density2d(
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
            FlowCal.gate.density2d(
                self.slope, bins=bins, gate_fraction=13.0/30, sigma=0.0,
                full_output=True).mask,
            np.array([0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,1,1,0,1,1,
                1,1,1,1,1], dtype=bool)
            )

    def test_slope_3_gated_data_1(self):
        bins = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        np.testing.assert_array_equal(
            FlowCal.gate.density2d(
                self.slope, bins=bins, gate_fraction=23.0/30, sigma=0.0),
            np.array([
                [2,2],
                [2,3],
                [2,4],
                [3,2],
                [3,3],
                [3,4],
                [4,2],
                [4,3],
                [4,4],
                [2,2],
                [2,3],
                [2,4],
                [3,2],
                [3,3],
                [3,4],
                [4,2],
                [4,3],
                [4,4],
                [3,3],
                [3,4],
                [4,3],
                [4,4],
                [4,4],
                ])
            )

    def test_slope_3_gated_data_2(self):
        bins = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        np.testing.assert_array_equal(
            FlowCal.gate.density2d(
                self.slope, bins=bins, gate_fraction=23.0/30, sigma=0.0,
                full_output=True).gated_data,
            np.array([
                [2,2],
                [2,3],
                [2,4],
                [3,2],
                [3,3],
                [3,4],
                [4,2],
                [4,3],
                [4,4],
                [2,2],
                [2,3],
                [2,4],
                [3,2],
                [3,3],
                [3,4],
                [4,2],
                [4,3],
                [4,4],
                [3,3],
                [3,4],
                [4,3],
                [4,4],
                [4,4],
                ])
            )

    def test_slope_3_mask(self):
        bins = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        np.testing.assert_array_equal(
            FlowCal.gate.density2d(
                self.slope, bins=bins, gate_fraction=23.0/30, sigma=0.0,
                full_output=True).mask,
            np.array([0,0,0,0,0,1,1,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,
                1,1,1,1,1], dtype=bool)
            )

    ###
    # Test edge cases
    ###

    # Confirm everything gets through with 1.0 gate_fraction

    def test_gate_fraction_1_gated_data_1(self):
        bins = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        np.testing.assert_array_equal(
            FlowCal.gate.density2d(
                self.pyramid, bins=bins, gate_fraction=1.0, sigma=0.0),
            self.pyramid
            )

    def test_gate_fraction_1_gated_data_2(self):
        bins = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        np.testing.assert_array_equal(
            FlowCal.gate.density2d(
                self.pyramid, bins=bins, gate_fraction=1.0, sigma=0.0,
                full_output=True).gated_data,
            self.pyramid
            )

    def test_gate_fraction_1_mask(self):
        bins = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        np.testing.assert_array_equal(
            FlowCal.gate.density2d(
                self.pyramid, bins=bins, gate_fraction=1.0, sigma=0.0,
                full_output=True).mask,
            np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                1,1,1,1,1,1], dtype=bool)
            )

    # Confirm nothing gets through with 0.0 gate_fraction

    def test_gate_fraction_2_gated_data_1(self):
        bins = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        np.testing.assert_array_equal(
            FlowCal.gate.density2d(
                self.pyramid, bins=bins, gate_fraction=0.0, sigma=0.0),
            np.empty((0,2), dtype=self.pyramid.dtype)     # empty 2D array
            )

    def test_gate_fraction_2_gated_data_2(self):
        bins = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        np.testing.assert_array_equal(
            FlowCal.gate.density2d(
                self.pyramid, bins=bins, gate_fraction=0.0, sigma=0.0,
                full_output=True).gated_data,
            np.empty((0,2), dtype=self.pyramid.dtype)     # empty 2D array
            )

    def test_gate_fraction_2_mask(self):
        bins = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        np.testing.assert_array_equal(
            FlowCal.gate.density2d(
                self.pyramid, bins=bins, gate_fraction=0.0, sigma=0.0,
                full_output=True).mask,
            np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                0,0,0,0,0,0], dtype=bool)
            )

    # Test error when gate_fraction outside [0,1]

    def test_gate_fraction_2_error_negative_gate_fraction(self):
        bins = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        self.assertRaises(
            ValueError,
            FlowCal.gate.density2d,
            self.pyramid,
            bins=bins,
            gate_fraction=-0.1,
            sigma=0.0,
            )

    def test_gate_fraction_2_error_large_gate_fraction(self):
        bins = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        self.assertRaises(
            ValueError,
            FlowCal.gate.density2d,
            self.pyramid,
            bins=bins,
            gate_fraction=1.1,
            sigma=0.0,
            )

    # Test implicit gating (when values exist outside specified bins)
    #
    # The expected behavior is that density2d() should mimic
    # np.histogram2d()'s behavior: values outside the specified bins are
    # ignored (in the context of a gate function, this means they are
    # implicitly gated out).

    def test_implicit_gating_1_gated_data_1(self):
        bins = [0.5, 1.5, 2.5, 3.5]
        np.testing.assert_array_equal(
            FlowCal.gate.density2d(
                self.pyramid, bins=bins, gate_fraction=3.0/15, sigma=0.0),
            np.array([
                [2,2],
                [2,2],
                [2,2],
                ])
            )

    def test_implicit_gating_1_gated_data_2(self):
        bins = [0.5, 1.5, 2.5, 3.5]
        np.testing.assert_array_equal(
            FlowCal.gate.density2d(
                self.pyramid, bins=bins, gate_fraction=3.0/15, sigma=0.0,
                full_output=True).gated_data,
            np.array([
                [2,2],
                [2,2],
                [2,2],
                ])
            )

    def test_implicit_gating_1_mask(self):
        bins = [0.5, 1.5, 2.5, 3.5]
        np.testing.assert_array_equal(
            FlowCal.gate.density2d(
                self.pyramid, bins=bins, gate_fraction=3.0/15, sigma=0.0,
                full_output=True).mask,
            np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,
                1,1,0,0,0,0], dtype=bool)
            )

    def test_implicit_gating_2_gated_data_1(self):
        bins = [0.5, 1.5, 2.5, 3.5]
        np.testing.assert_array_equal(
            FlowCal.gate.density2d(
                self.pyramid, bins=bins, gate_fraction=11.0/15, sigma=0.0),
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

    def test_implicit_gating_2_gated_data_2(self):
        bins = [0.5, 1.5, 2.5, 3.5]
        np.testing.assert_array_equal(
            FlowCal.gate.density2d(
                self.pyramid, bins=bins, gate_fraction=11.0/15, sigma=0.0,
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

    def test_implicit_gating_2_mask(self):
        bins = [0.5, 1.5, 2.5, 3.5]
        np.testing.assert_array_equal(
            FlowCal.gate.density2d(
                self.pyramid, bins=bins, gate_fraction=11.0/15, sigma=0.0,
                full_output=True).mask,
            np.array([0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,1,0,0,0,0,0,0,0,
                1,1,1,1,1,1], dtype=bool)
            )

    def test_implicit_gating_3_gated_data_1(self):
        bins = [1.5, 2.5, 3.5]
        np.testing.assert_array_equal(
            FlowCal.gate.density2d(
                self.slope, bins=bins, gate_fraction=3.0/9, sigma=0.0),
            np.array([
                [3,3],
                [3,3],
                [3,3],
                ])
            )

    def test_implicit_gating_3_gated_data_2(self):
        bins = [1.5, 2.5, 3.5]
        np.testing.assert_array_equal(
            FlowCal.gate.density2d(
                self.slope, bins=bins, gate_fraction=3.0/9, sigma=0.0,
                full_output=True).gated_data,
            np.array([
                [3,3],
                [3,3],
                [3,3],
                ])
            )

    def test_implicit_gating_3_mask(self):
        bins = [1.5, 2.5, 3.5]
        np.testing.assert_array_equal(
            FlowCal.gate.density2d(
                self.slope, bins=bins, gate_fraction=3.0/9, sigma=0.0,
                full_output=True).mask,
            np.array([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,
                1,0,0,0,0], dtype=bool)
            )

    def test_implicit_gating_4_gated_data_1(self):
        bins = [0.5, 1.5, 2.5, 3.5, 4.5]
        np.testing.assert_array_equal(
            FlowCal.gate.density2d(
                self.pyramid, bins=bins, gate_fraction=3.0/22, sigma=0.0),
            np.array([
                [2,2],
                [2,2],
                [2,2],
                ])
            )

    def test_implicit_gating_4_gated_data_2(self):
        bins = [0.5, 1.5, 2.5, 3.5, 4.5]
        np.testing.assert_array_equal(
            FlowCal.gate.density2d(
                self.pyramid, bins=bins, gate_fraction=3.0/22, sigma=0.0,
                full_output=True).gated_data,
            np.array([
                [2,2],
                [2,2],
                [2,2],
                ])
            )

    def test_implicit_gating_4_mask(self):
        bins = [0.5, 1.5, 2.5, 3.5, 4.5]
        np.testing.assert_array_equal(
            FlowCal.gate.density2d(
                self.pyramid, bins=bins, gate_fraction=3.0/22, sigma=0.0,
                full_output=True).mask,
            np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,
                1,1,0,0,0,0], dtype=bool)
            )

    # Test sub-binning (multiple values per bin)

    def test_sub_bin_1_gated_data_1(self):
        bins = [0.5, 2.5, 4.5]
        np.testing.assert_array_equal(
            FlowCal.gate.density2d(
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

    def test_sub_bin_1_gated_data_2(self):
        bins = [0.5, 2.5, 4.5]
        np.testing.assert_array_equal(
            FlowCal.gate.density2d(
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

    def test_sub_bin_1_mask(self):
        bins = [0.5, 2.5, 4.5]
        np.testing.assert_array_equal(
            FlowCal.gate.density2d(
                self.slope, bins=bins, gate_fraction=13.0/30, sigma=0.0,
                full_output=True).mask,
            np.array([0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,1,1,0,1,1,
                1,1,1,1,1], dtype=bool)
            )

    # Test bins edge case (when bin edges = values)
    #
    # Again, the expected behavior is that density2d() should mimic
    # np.histogram2d()'s behavior which will group the right-most two values
    # together in the same bin (since the last bins interval is fully closed,
    # as opposed to all other bins intervals which are half-open).

    def test_bins_edge_case_1_gated_data_1(self):
        bins = list(range(5))
        np.testing.assert_array_equal(
            FlowCal.gate.density2d(
                self.pyramid, bins=bins, gate_fraction=4.0/31, sigma=0.0),
            np.array([
                [3,3],
                [3,4],
                [4,3],
                [4,4],
                ])
            )

    def test_bins_edge_case_1_gated_data_2(self):
        bins = list(range(5))
        np.testing.assert_array_equal(
            FlowCal.gate.density2d(
                self.pyramid, bins=bins, gate_fraction=4.0/31, sigma=0.0,
                full_output=True).gated_data,
            np.array([
                [3,3],
                [3,4],
                [4,3],
                [4,4],
                ])
            )

    def test_bins_edge_case_1_mask(self):
        bins = list(range(5))
        np.testing.assert_array_equal(
            FlowCal.gate.density2d(
                self.pyramid, bins=bins, gate_fraction=4.0/31, sigma=0.0,
                full_output=True).mask,
            np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1,
                0,0,0,0,0,0], dtype=bool)
            )

    def test_bins_edge_case_2_gated_data_1(self):
        bins = list(range(5))
        np.testing.assert_array_equal(
            FlowCal.gate.density2d(
                self.pyramid, bins=bins, gate_fraction=13.0/31, sigma=0.0),
            np.array([
                [2,2],
                [2,3],
                [2,4],
                [3,2],
                [3,3],
                [3,4],
                [4,2],
                [4,3],
                [4,4],
                [2,2],
                [2,2],
                [2,3],
                [3,2],
                ])
            )

    def test_bins_edge_case_2_gated_data_2(self):
        bins = list(range(5))
        np.testing.assert_array_equal(
            FlowCal.gate.density2d(
                self.pyramid, bins=bins, gate_fraction=13.0/31, sigma=0.0,
                full_output=True).gated_data,
            np.array([
                [2,2],
                [2,3],
                [2,4],
                [3,2],
                [3,3],
                [3,4],
                [4,2],
                [4,3],
                [4,4],
                [2,2],
                [2,2],
                [2,3],
                [3,2],
                ])
            )

    def test_bins_edge_case_2_mask(self):
        bins = list(range(5))
        np.testing.assert_array_equal(
            FlowCal.gate.density2d(
                self.pyramid, bins=bins, gate_fraction=13.0/31, sigma=0.0,
                full_output=True).mask,
            np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,
                1,1,0,0,1,1], dtype=bool)
            )

    def test_bins_edge_case_3_gated_data_1(self):
        bins = [1,2,3]
        np.testing.assert_array_equal(
            FlowCal.gate.density2d(
                self.slope, bins=bins, gate_fraction=9.0/14, sigma=0.0),
            np.array([
                [2,2],
                [2,3],
                [3,2],
                [3,3],
                [2,2],
                [2,3],
                [3,2],
                [3,3],
                [3,3],
                ])
            )

    def test_bins_edge_case_3_gated_data_2(self):
        bins = [1,2,3]
        np.testing.assert_array_equal(
            FlowCal.gate.density2d(
                self.slope, bins=bins, gate_fraction=9.0/14, sigma=0.0,
                full_output=True).gated_data,
            np.array([
                [2,2],
                [2,3],
                [3,2],
                [3,3],
                [2,2],
                [2,3],
                [3,2],
                [3,3],
                [3,3],
                ])
            )

    def test_bins_edge_case_3_mask(self):
        bins = [1,2,3]
        np.testing.assert_array_equal(
            FlowCal.gate.density2d(
                self.slope, bins=bins, gate_fraction=9.0/14, sigma=0.0,
                full_output=True).mask,
            np.array([0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,1,1,0,1,1,0,0,0,0,
                1,0,0,0,0], dtype=bool)
            )

    def test_bins_edge_case_4_gated_data_1(self):
        bins = [1,2,3]
        np.testing.assert_array_equal(
            FlowCal.gate.density2d(
                self.slope, bins=bins, gate_fraction=13.0/14, sigma=0.0),
            np.array([
                [1,2],
                [1,3],
                [2,1],
                [2,2],
                [2,3],
                [3,1],
                [3,2],
                [3,3],
                [2,2],
                [2,3],
                [3,2],
                [3,3],
                [3,3],
                ])
            )

    def test_bins_edge_case_4_gated_data_2(self):
        bins = [1,2,3]
        np.testing.assert_array_equal(
            FlowCal.gate.density2d(
                self.slope, bins=bins, gate_fraction=13.0/14, sigma=0.0,
                full_output=True).gated_data,
            np.array([
                [1,2],
                [1,3],
                [2,1],
                [2,2],
                [2,3],
                [3,1],
                [3,2],
                [3,3],
                [2,2],
                [2,3],
                [3,2],
                [3,3],
                [3,3],
                ])
            )

    def test_bins_edge_case_4_mask(self):
        bins = [1,2,3]
        np.testing.assert_array_equal(
            FlowCal.gate.density2d(
                self.slope, bins=bins, gate_fraction=13.0/14, sigma=0.0,
                full_output=True).mask,
            np.array([0,1,1,0,1,1,1,0,1,1,1,0,0,0,0,0,1,1,0,1,1,0,0,0,0,
                1,0,0,0,0], dtype=bool)
            )

if __name__ == '__main__':
    unittest.main()
