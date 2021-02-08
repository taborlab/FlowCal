#!/usr/bin/python
#
# test_transform.py - Unit tests for transform module
#
# Author: Sebastian M. Castillo-Hair (smc9@rice.edu)
# Date: 7/1/2015
#
# Requires:
#   * FlowCal.io
#   * FlowCal.transform
#   * numpy
#

import FlowCal.io
import FlowCal.transform
import numpy as np
import unittest

class TestRFIArray(unittest.TestCase):
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

    def test_rfi_original_integrity(self):
        db = self.d.copy()
        dt = FlowCal.transform.to_rfi(self.d,
                                      channels=[0,1],
                                      amplification_type=[(0,0), (0,0)],
                                      amplifier_gain=[1.0, 1.0],
                                      resolution=[1024, 1024],)
        np.testing.assert_array_equal(self.d, db)

    def test_rfi_arg_error_amplification_type_absent(self):
        with self.assertRaises(ValueError):
            FlowCal.transform.to_rfi(self.d,
                                     channels=[0,1])

    def test_rfi_arg_error_amplification_type_length(self):
        with self.assertRaises(ValueError):
            FlowCal.transform.to_rfi(self.d,
                                     channels=[0,1],
                                     amplification_type=[(4,1), (4,1), (4,1)])

    def test_rfi_arg_error_resolution_absent(self):
        with self.assertRaises(ValueError):
            FlowCal.transform.to_rfi(self.d,
                                     channels=[0,1],
                                     amplification_type=[(4,1), (4,1)])

    def test_rfi_arg_error_resolution_length(self):
        with self.assertRaises(ValueError):
            FlowCal.transform.to_rfi(self.d,
                                     channels=[0,1],
                                     amplification_type=[(4,1), (4,1)],
                                     resolution=[1024])

    def test_rfi_arg_error_amplifier_gain_length(self):
        with self.assertRaises(ValueError):
            FlowCal.transform.to_rfi(self.d,
                                     channels=[0,1],
                                     amplification_type=[(0,0), (0,0)],
                                     amplifier_gain=[3,4,4])

    def test_rfi_1d_log_1(self):
        dt = FlowCal.transform.to_rfi(self.d,
                                      channels=1,
                                      amplification_type=(4, 1),
                                      resolution=1024)
        np.testing.assert_array_equal(dt[:,0], self.d[:,0])
        np.testing.assert_array_equal(dt[:,1], 10**(self.d[:,1]/256.0))
        np.testing.assert_array_equal(dt[:,2], self.d[:,2])

    def test_rfi_1d_log_2(self):
        dt = FlowCal.transform.to_rfi(self.d,
                                      channels=2,
                                      amplification_type=(2, 0.01),
                                      amplifier_gain=5.0,
                                      resolution=256)
        np.testing.assert_array_equal(dt[:,0], self.d[:,0])
        np.testing.assert_array_equal(dt[:,1], self.d[:,1])
        np.testing.assert_array_equal(dt[:,2], 0.01*10**(self.d[:,2]/128.0))

    def test_rfi_1d_linear_1(self):
        dt = FlowCal.transform.to_rfi(self.d,
                                      channels=2,
                                      amplification_type=(0, 0),
                                      amplifier_gain=None,
                                      resolution=256)
        np.testing.assert_array_equal(dt[:,0], self.d[:,0])
        np.testing.assert_array_equal(dt[:,1], self.d[:,1])
        np.testing.assert_array_equal(dt[:,2], self.d[:,2])

    def test_rfi_1d_linear_2(self):
        dt = FlowCal.transform.to_rfi(self.d,
                                      channels=1,
                                      amplification_type=(0, 0),
                                      amplifier_gain=5.0,
                                      resolution=256)
        np.testing.assert_array_equal(dt[:,0], self.d[:,0])
        np.testing.assert_array_equal(dt[:,1], self.d[:,1]/5.0)
        np.testing.assert_array_equal(dt[:,2], self.d[:,2])

    def test_rfi_2d_log_1(self):
        dt = FlowCal.transform.to_rfi(self.d,
                                      channels=[1,2],
                                      amplification_type=[(4, 1), (2, 0.01)],
                                      resolution=[1024, 256])
        np.testing.assert_array_equal(dt[:,0], self.d[:,0])
        np.testing.assert_array_equal(dt[:,1], 10**(self.d[:,1]/256.0))
        np.testing.assert_array_equal(dt[:,2], 0.01*10**(self.d[:,2]/128.0))

    def test_rfi_2d_mixed_1(self):
        dt = FlowCal.transform.to_rfi(self.d,
                                      channels=[1,2],
                                      amplification_type=[(4, 1), (0, 0)],
                                      amplifier_gain=[4., None],
                                      resolution=[1024, 1024])
        np.testing.assert_array_equal(dt[:,0], self.d[:,0])
        np.testing.assert_array_equal(dt[:,1], 10**(self.d[:,1]/256.0))
        np.testing.assert_array_equal(dt[:,2], self.d[:,2])

    def test_rfi_2d_mixed_2(self):
        dt = FlowCal.transform.to_rfi(self.d,
                                      channels=[1,2],
                                      amplification_type=[(4, 1), (0, 0)],
                                      amplifier_gain=[4., 10.],
                                      resolution=[1024, 1024])
        np.testing.assert_array_equal(dt[:,0], self.d[:,0])
        np.testing.assert_array_equal(dt[:,1], 10**(self.d[:,1]/256.0))
        np.testing.assert_array_equal(dt[:,2], self.d[:,2]/10.)

    def test_rfi_default_channel_1(self):
        dt = FlowCal.transform.to_rfi(self.d,
                                      amplification_type=[(4,1)]*3,
                                      amplifier_gain=[4., 5., 10.],
                                      resolution=[1024]*3)
        np.testing.assert_array_equal(dt, 10**(self.d/256.0))

    def test_rfi_default_channel_2(self):
        dt = FlowCal.transform.to_rfi(self.d,
                                      amplification_type=[(0,0)]*3,
                                      amplifier_gain=[10., 100., 0.01],
                                      resolution=[1024]*3)
        np.testing.assert_array_equal(dt, self.d/np.array([10., 100., 0.01]))

class TestRFIFCSLog(unittest.TestCase):
    def setUp(self):
        self.channel_names = ['FSC-H', 'SSC-H', 'FL1-H', 
                                'FL2-H', 'FL3-H', 'Time']
        self.d = FlowCal.io.FCSData('test/Data001.fcs')
        self.n_samples = self.d.shape[0]

    def test_rfi_original_integrity(self):
        db = self.d.copy()
        dt = FlowCal.transform.to_rfi(self.d,
                                      channels=['FSC-H', 'SSC-H'],
                                      amplification_type=[(4,1), (4,1)],
                                      resolution=[1024, 1024])
        np.testing.assert_array_equal(self.d, db)

    def test_rfi_arg_error_amplification_type_length(self):
        with self.assertRaises(ValueError):
            FlowCal.transform.to_rfi(self.d,
                                     channels=[0,1],
                                     amplification_type=[(4,1), (4,1), (4,1)])

    def test_rfi_arg_error_resolution_length(self):
        with self.assertRaises(ValueError):
            FlowCal.transform.to_rfi(self.d,
                                     channels=[0,1],
                                     amplification_type=[(4,1), (4,1)],
                                     resolution=[1024])

    def test_rfi_arg_error_amplifier_gain_length(self):
        with self.assertRaises(ValueError):
            FlowCal.transform.to_rfi(self.d,
                                     channels=[0,1],
                                     amplification_type=[(0,0), (0,0)],
                                     amplifier_gain=[3,4,4])

    def test_rfi_1d_log_1(self):
        dt = FlowCal.transform.to_rfi(self.d,
                                      channels='FL1-H',
                                      amplification_type=(4, 0.01),
                                      resolution=512)
        np.testing.assert_array_equal(dt[:,'FSC-H'], self.d[:,'FSC-H'])
        np.testing.assert_array_equal(dt[:,'SSC-H'], self.d[:,'SSC-H'])
        np.testing.assert_array_equal(dt[:,'FL1-H'],
                                      0.01*10**(self.d[:,'FL1-H']/128.0))
        np.testing.assert_array_equal(dt[:,'FL2-H'], self.d[:,'FL2-H'])
        np.testing.assert_array_equal(dt[:,'FL3-H'], self.d[:,'FL3-H'])
        np.testing.assert_array_equal(dt[:,'Time'], self.d[:,'Time'])

    def test_rfi_1d_log_2(self):
        dt = FlowCal.transform.to_rfi(self.d,
                                      channels=2,
                                      amplification_type=(2, 0.01),
                                      amplifier_gain=50.,
                                      resolution=512)
        np.testing.assert_array_equal(dt[:,'FSC-H'], self.d[:,'FSC-H'])
        np.testing.assert_array_equal(dt[:,'SSC-H'], self.d[:,'SSC-H'])
        np.testing.assert_array_equal(dt[:,'FL1-H'],
                                      0.01*10**(self.d[:,'FL1-H']/256.0))
        np.testing.assert_array_equal(dt[:,'FL2-H'], self.d[:,'FL2-H'])
        np.testing.assert_array_equal(dt[:,'FL3-H'], self.d[:,'FL3-H'])
        np.testing.assert_array_equal(dt[:,'Time'], self.d[:,'Time'])

    def test_rfi_1d_linear_1(self):
        dt = FlowCal.transform.to_rfi(self.d,
                                      channels=2,
                                      amplification_type=(0, 0),
                                      amplifier_gain=50.,
                                      resolution=512)
        np.testing.assert_array_equal(dt[:,'FSC-H'], self.d[:,'FSC-H'])
        np.testing.assert_array_equal(dt[:,'SSC-H'], self.d[:,'SSC-H'])
        np.testing.assert_array_equal(dt[:,'FL1-H'], self.d[:,'FL1-H']/50.)
        np.testing.assert_array_equal(dt[:,'FL2-H'], self.d[:,'FL2-H'])
        np.testing.assert_array_equal(dt[:,'FL3-H'], self.d[:,'FL3-H'])
        np.testing.assert_array_equal(dt[:,'Time'], self.d[:,'Time'])

    def test_rfi_1d_defaults(self):
        dt = FlowCal.transform.to_rfi(self.d,
                                      channels='FL1-H')
        np.testing.assert_array_equal(dt[:,'FSC-H'], self.d[:,'FSC-H'])
        np.testing.assert_array_equal(dt[:,'SSC-H'], self.d[:,'SSC-H'])
        np.testing.assert_array_equal(dt[:,'FL1-H'],
                                      10**(self.d[:,'FL1-H']/256.0))
        np.testing.assert_array_equal(dt[:,'FL2-H'], self.d[:,'FL2-H'])
        np.testing.assert_array_equal(dt[:,'FL3-H'], self.d[:,'FL3-H'])
        np.testing.assert_array_equal(dt[:,'Time'], self.d[:,'Time'])

    def test_rfi_2d_log_1(self):
        dt = FlowCal.transform.to_rfi(self.d,
                                      channels=['FL1-H', 'FL3-H'],
                                      amplification_type=[(4, 0.01), (2, 1)],
                                      resolution=[512, 2048])
        np.testing.assert_array_equal(dt[:,'FSC-H'], self.d[:,'FSC-H'])
        np.testing.assert_array_equal(dt[:,'SSC-H'], self.d[:,'SSC-H'])
        np.testing.assert_array_equal(dt[:,'FL1-H'],
                                      0.01*10**(self.d[:,'FL1-H']/128.0))
        np.testing.assert_array_equal(dt[:,'FL2-H'], self.d[:,'FL2-H'])
        np.testing.assert_array_equal(dt[:,'FL3-H'],
                                      10**(self.d[:,'FL3-H']/1024.))
        np.testing.assert_array_equal(dt[:,'Time'], self.d[:,'Time'])

    def test_rfi_2d_mixed_1(self):
        dt = FlowCal.transform.to_rfi(self.d,
                                      channels=[2, 4],
                                      amplification_type=[(4, 0.01), (0, 0)],
                                      resolution=[512, 1024])
        np.testing.assert_array_equal(dt[:,'FSC-H'], self.d[:,'FSC-H'])
        np.testing.assert_array_equal(dt[:,'SSC-H'], self.d[:,'SSC-H'])
        np.testing.assert_array_equal(dt[:,'FL1-H'],
                                      0.01*10**(self.d[:,'FL1-H']/128.0))
        np.testing.assert_array_equal(dt[:,'FL2-H'], self.d[:,'FL2-H'])
        np.testing.assert_array_equal(dt[:,'FL3-H'], self.d[:,'FL3-H'])
        np.testing.assert_array_equal(dt[:,'Time'], self.d[:,'Time'])

    def test_rfi_2d_mixed_2(self):
        dt = FlowCal.transform.to_rfi(self.d,
                                      channels=[2, 4],
                                      amplification_type=[(4, 0.01), (0, 0)],
                                      amplifier_gain=[5., None],
                                      resolution=[512, 1024])
        np.testing.assert_array_equal(dt[:,'FSC-H'], self.d[:,'FSC-H'])
        np.testing.assert_array_equal(dt[:,'SSC-H'], self.d[:,'SSC-H'])
        np.testing.assert_array_equal(dt[:,'FL1-H'],
                                      0.01*10**(self.d[:,'FL1-H']/128.0))
        np.testing.assert_array_equal(dt[:,'FL2-H'], self.d[:,'FL2-H'])
        np.testing.assert_array_equal(dt[:,'FL3-H'], self.d[:,'FL3-H'])
        np.testing.assert_array_equal(dt[:,'Time'], self.d[:,'Time'])

    def test_rfi_2d_mixed_3(self):
        dt = FlowCal.transform.to_rfi(self.d,
                                      channels=[2, 4],
                                      amplification_type=[(4, 0.01), (0, 0)],
                                      amplifier_gain=[5., 10.],
                                      resolution=[512, 1024])
        np.testing.assert_array_equal(dt[:,'FSC-H'], self.d[:,'FSC-H'])
        np.testing.assert_array_equal(dt[:,'SSC-H'], self.d[:,'SSC-H'])
        np.testing.assert_array_equal(dt[:,'FL1-H'],
                                      0.01*10**(self.d[:,'FL1-H']/128.0))
        np.testing.assert_array_equal(dt[:,'FL2-H'], self.d[:,'FL2-H'])
        np.testing.assert_array_equal(dt[:,'FL3-H'], self.d[:,'FL3-H']/10.)
        np.testing.assert_array_equal(dt[:,'Time'], self.d[:,'Time'])

    def test_rfi_2d_defaults(self):
        dt = FlowCal.transform.to_rfi(self.d,
                                      channels=['FL1-H', 'FL3-H'])
        np.testing.assert_array_equal(dt[:,'FSC-H'], self.d[:,'FSC-H'])
        np.testing.assert_array_equal(dt[:,'SSC-H'], self.d[:,'SSC-H'])
        np.testing.assert_array_equal(dt[:,'FL1-H'],
                                      10**(self.d[:,'FL1-H']/256.0))
        np.testing.assert_array_equal(dt[:,'FL2-H'], self.d[:,'FL2-H'])
        np.testing.assert_array_equal(dt[:,'FL3-H'],
                                      10**(self.d[:,'FL3-H']/256.))
        np.testing.assert_array_equal(dt[:,'Time'], self.d[:,'Time'])

    def test_rfi_2d_range(self):
        dt = FlowCal.transform.to_rfi(self.d,
                                      channels=['FL1-H', 'FL3-H'],
                                      resolution=[512, 2048],
                                      amplification_type=[(4, 0.01), (2, 1)])
        np.testing.assert_array_equal(
            dt.range('FSC-H'),
            self.d.range('FSC-H'))
        np.testing.assert_array_equal(
            dt.range('SSC-H'),
            self.d.range('SSC-H'))
        np.testing.assert_array_equal(
            dt.range('FL1-H'),
            [0.01*10**(r/128.0) for r in self.d.range('FL1-H')])
        np.testing.assert_array_equal(
            dt.range('FL2-H'),
            self.d.range('FL2-H'))
        np.testing.assert_array_equal(
            dt.range('FL3-H'),
            [10**(r/1024.0) for r in self.d.range('FL3-H')])

    def test_rfi_default_channel(self):
        # Leave time channel out
        channels = ['FSC-H', 'SSC-H', 'FL1-H', 'FL2-H', 'FL3-H']
        dt = FlowCal.transform.to_rfi(self.d[:, channels])
        np.testing.assert_array_equal(dt[:,'FSC-H'], self.d[:,'FSC-H'])
        np.testing.assert_array_equal(dt[:,'SSC-H'], self.d[:,'SSC-H'])
        np.testing.assert_array_equal(dt[:,'FL1-H'],
                                      10**(self.d[:,'FL1-H']/256.0))
        np.testing.assert_array_equal(dt[:,'FL2-H'],
                                      10**(self.d[:,'FL2-H']/256.0))
        np.testing.assert_array_equal(dt[:,'FL3-H'],
                                      10**(self.d[:,'FL3-H']/256.0))

class TestRFIFCSLinear(unittest.TestCase):
    def setUp(self):
        self.channel_names = ['FSC-A',
                              'FSC-H',
                              'FSC-W',
                              'SSC-A',
                              'SSC-H',
                              'SSC-W',
                              'FSC PMT-A',
                              'FSC PMT-H',
                              'FSC PMT-W',
                              'GFP-A',
                              'GFP-H',
                              'mCherry-A',
                              'mCherry-H',
                              'Time']
        self.d = FlowCal.io.FCSData('test/Data004.fcs')
        self.n_samples = self.d.shape[0]

    def test_rfi_original_integrity(self):
        db = self.d.copy()
        dt = FlowCal.transform.to_rfi(self.d,
                                      channels=['FSC-H', 'SSC-H'])
        np.testing.assert_array_equal(self.d, db)

    def test_rfi_arg_error_amplification_type_length(self):
        with self.assertRaises(ValueError):
            FlowCal.transform.to_rfi(self.d,
                                     channels=[0,1],
                                     amplification_type=[(4,1), (4,1), (4,1)])

    def test_rfi_arg_error_resolution_length(self):
        with self.assertRaises(ValueError):
            FlowCal.transform.to_rfi(self.d,
                                     channels=[0,1],
                                     amplification_type=[(4,1), (4,1)],
                                     resolution=[1024])

    def test_rfi_arg_error_amplifier_gain_length(self):
        with self.assertRaises(ValueError):
            FlowCal.transform.to_rfi(self.d,
                                     channels=[0,1],
                                     amplification_type=[(0,0), (0,0)],
                                     amplifier_gain=[3,4,4])

    def test_rfi_1d_defaults(self):
        dt = FlowCal.transform.to_rfi(self.d,
                                      channels=['GFP-A', 'mCherry-A', 'Time'])
        np.testing.assert_array_equal(dt[:,'FSC-A'], self.d[:,'FSC-A'])
        np.testing.assert_array_equal(dt[:,'FSC-H'], self.d[:,'FSC-H'])
        np.testing.assert_array_equal(dt[:,'FSC-W'], self.d[:,'FSC-W'])
        np.testing.assert_array_equal(dt[:,'SSC-A'], self.d[:,'SSC-A'])
        np.testing.assert_array_equal(dt[:,'SSC-H'], self.d[:,'SSC-H'])
        np.testing.assert_array_equal(dt[:,'SSC-W'], self.d[:,'SSC-W'])
        np.testing.assert_array_equal(dt[:,'FSC PMT-A'], self.d[:,'FSC PMT-A'])
        np.testing.assert_array_equal(dt[:,'FSC PMT-H'], self.d[:,'FSC PMT-H'])
        np.testing.assert_array_equal(dt[:,'FSC PMT-W'], self.d[:,'FSC PMT-W'])
        np.testing.assert_array_equal(dt[:,'GFP-A'], self.d[:,'GFP-A'])
        np.testing.assert_array_equal(dt[:,'GFP-H'], self.d[:,'GFP-H'])
        np.testing.assert_array_equal(dt[:,'mCherry-A'], self.d[:,'mCherry-A'])
        np.testing.assert_array_equal(dt[:,'mCherry-H'], self.d[:,'mCherry-H'])
        np.testing.assert_array_almost_equal(dt[:,'Time'],
                                             self.d[:,'Time']/0.01,
                                             decimal=3)

    def test_rfi_1d_defaults_2(self):
        dt = FlowCal.transform.to_rfi(self.d,
                                      channels=['GFP-A', 'mCherry-A', 'Time'],
                                      amplifier_gain=[2., 1., 0.01])
        np.testing.assert_array_equal(dt[:,'FSC-A'], self.d[:,'FSC-A'])
        np.testing.assert_array_equal(dt[:,'FSC-H'], self.d[:,'FSC-H'])
        np.testing.assert_array_equal(dt[:,'FSC-W'], self.d[:,'FSC-W'])
        np.testing.assert_array_equal(dt[:,'SSC-A'], self.d[:,'SSC-A'])
        np.testing.assert_array_equal(dt[:,'SSC-H'], self.d[:,'SSC-H'])
        np.testing.assert_array_equal(dt[:,'SSC-W'], self.d[:,'SSC-W'])
        np.testing.assert_array_equal(dt[:,'FSC PMT-A'], self.d[:,'FSC PMT-A'])
        np.testing.assert_array_equal(dt[:,'FSC PMT-H'], self.d[:,'FSC PMT-H'])
        np.testing.assert_array_equal(dt[:,'FSC PMT-W'], self.d[:,'FSC PMT-W'])
        np.testing.assert_array_equal(dt[:,'GFP-A'], self.d[:,'GFP-A']/2.)
        np.testing.assert_array_equal(dt[:,'GFP-H'], self.d[:,'GFP-H'])
        np.testing.assert_array_equal(dt[:,'mCherry-A'], self.d[:,'mCherry-A'])
        np.testing.assert_array_equal(dt[:,'mCherry-H'], self.d[:,'mCherry-H'])
        np.testing.assert_array_almost_equal(dt[:,'Time'],
                                             self.d[:,'Time']/0.01,
                                             decimal=3)

class TestMefArray(unittest.TestCase):
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
        self.sc0 = lambda x: x + 10
        self.sc1 = lambda x: x**2
        self.sc2 = lambda x: np.log(x)

    def test_mef_length_error(self):
        self.assertRaises(ValueError, FlowCal.transform.to_mef, 
                            self.d, 1, [self.sc1], [1,2])

    def test_mef_channel_error(self):
        self.assertRaises(ValueError, FlowCal.transform.to_mef, 
                            self.d, 0, [self.sc1, self.sc2], [1,2])

    def test_mef_1d_1(self):
        dt = FlowCal.transform.to_mef(
            self.d, 1, [self.sc1, self.sc2], [1,2])
        np.testing.assert_array_equal(dt[:,0], self.d[:,0])
        np.testing.assert_array_equal(dt[:,1], self.d[:,1]**2.)
        np.testing.assert_array_equal(dt[:,2], self.d[:,2])

    def test_mef_1d_2(self):
        dt = FlowCal.transform.to_mef(
            self.d, 2, [self.sc1, self.sc2], [1,2])
        np.testing.assert_array_equal(dt[:,0], self.d[:,0])
        np.testing.assert_array_equal(dt[:,1], self.d[:,1])
        np.testing.assert_array_equal(dt[:,2], np.log(self.d[:,2]))

    def test_mef_2d(self):
        dt = FlowCal.transform.to_mef(
            self.d, [1,2], [self.sc1, self.sc2], [1,2])
        np.testing.assert_array_equal(dt[:,0], self.d[:,0])
        np.testing.assert_array_equal(dt[:,1], self.d[:,1]**2.)
        np.testing.assert_array_equal(dt[:,2], np.log(self.d[:,2]))

    def test_mef_default_channel(self):
        dt = FlowCal.transform.to_mef(
            self.d, None, [self.sc1, self.sc2], [1,2])
        np.testing.assert_array_equal(dt[:,0], self.d[:,0])
        np.testing.assert_array_equal(dt[:,1], self.d[:,1]**2.)
        np.testing.assert_array_equal(dt[:,2], np.log(self.d[:,2]))

    def test_mef_default_sc_channel(self):
        dt = FlowCal.transform.to_mef(
            self.d, None, [self.sc0, self.sc1, self.sc2], None)
        np.testing.assert_array_equal(dt[:,0], self.d[:,0] + 10)
        np.testing.assert_array_equal(dt[:,1], self.d[:,1]**2.)
        np.testing.assert_array_equal(dt[:,2], np.log(self.d[:,2]))

    def test_mef_default_sc_channel_error(self):
        self.assertRaises(ValueError, FlowCal.transform.to_mef, 
                            self.d, None, [self.sc1, self.sc2], None)

class TestMefFCS(unittest.TestCase):
    def setUp(self):
        self.channel_names = ['FSC-H', 'SSC-H', 'FL1-H', 
                                'FL2-H', 'FL3-H', 'Time']
        self.d = FlowCal.io.FCSData('test/Data001.fcs')
        self.n_samples = self.d.shape[0]
        self.sc0 = lambda x: x + 10
        self.sc1 = lambda x: x**2
        self.sc2 = lambda x: np.log(x + 1)

    def test_mef_length_error(self):
        self.assertRaises(ValueError, FlowCal.transform.to_mef, 
                    self.d, 'FL1-H', [self.sc1], ['FL1-H','FL3-H'])

    def test_mef_channel_error(self):
        self.assertRaises(ValueError, FlowCal.transform.to_mef, 
                    self.d, 'FSC-H', [self.sc1, self.sc2], ['FL1-H','FL3-H'])

    def test_mef_1d_1(self):
        dt = FlowCal.transform.to_mef(
            self.d, 'FL1-H', [self.sc1, self.sc2], ['FL1-H','FL3-H'])
        np.testing.assert_array_equal(dt[:,0], self.d[:,0])
        np.testing.assert_array_equal(dt[:,1], self.d[:,1])
        np.testing.assert_array_equal(dt[:,2], self.d[:,2]**2.)
        np.testing.assert_array_equal(dt[:,3], self.d[:,3])
        np.testing.assert_array_equal(dt[:,4], self.d[:,4])

    def test_mef_1d_2(self):
        dt = FlowCal.transform.to_mef(
            self.d, 'FL3-H', [self.sc1, self.sc2], ['FL1-H','FL3-H'])
        np.testing.assert_array_equal(dt[:,0], self.d[:,0])
        np.testing.assert_array_equal(dt[:,1], self.d[:,1])
        np.testing.assert_array_equal(dt[:,2], self.d[:,2])
        np.testing.assert_array_equal(dt[:,3], self.d[:,3])
        np.testing.assert_array_equal(dt[:,4], 
            np.log(self.d[:,4].astype(np.float64) + 1))

    def test_mef_2d(self):
        dt = FlowCal.transform.to_mef(
            self.d, ['FL1-H','FL3-H'], [self.sc1, self.sc2], ['FL1-H','FL3-H'])
        np.testing.assert_array_equal(dt[:,0], self.d[:,0])
        np.testing.assert_array_equal(dt[:,1], self.d[:,1])
        np.testing.assert_array_equal(dt[:,2], self.d[:,2]**2.)
        np.testing.assert_array_equal(dt[:,3], self.d[:,3])
        np.testing.assert_array_equal(dt[:,4], 
            np.log(self.d[:,4].astype(np.float64) + 1))

    def test_mef_default_channel(self):
        dt = FlowCal.transform.to_mef(
            self.d, None, [self.sc1, self.sc2], ['FL1-H','FL3-H'])
        np.testing.assert_array_equal(dt[:,0], self.d[:,0])
        np.testing.assert_array_equal(dt[:,1], self.d[:,1])
        np.testing.assert_array_equal(dt[:,2], self.d[:,2]**2.)
        np.testing.assert_array_equal(dt[:,3], self.d[:,3])
        np.testing.assert_array_equal(dt[:,4], 
            np.log(self.d[:,4].astype(np.float64) + 1))

    def test_mef_bins_channels(self):
        dt = FlowCal.transform.to_mef(
            self.d, ['FL1-H','FL3-H'], [self.sc1, self.sc2], ['FL1-H','FL3-H'])
        vit = [dt.range(i) for i in range(5)]
        vo = [np.array([0, 1023]),
              np.array([0, 1023]),
              np.array([0, 1023])**2,
              np.array([0, 1023]),
              np.log(np.array([0, 1023]) + 1),
              ]
        np.testing.assert_array_equal(vit, vo)

class TestCompensateArray(unittest.TestCase):
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
        self.a0 = np.array([1, 0.5, 2.1])
        self.A = np.array([[1, 0.3, 0.2], [0.6, 1, 0.5], [0.7, 0.8, 1]])
        self.dt_complete = np.linalg.solve(self.A, (self.d - self.a0).T).T

    def test_length_error_1(self):
        self.assertRaises(ValueError, FlowCal.transform.to_compensated, 
                    self.d, [0, 1, 2], self.a0, self.A, [0, 1])

    def test_length_error_2(self):
        self.assertRaises(ValueError, FlowCal.transform.to_compensated, 
                    self.d, [0, 1, 2], self.a0[0:1], self.A, [0, 1, 2])

    def test_length_error_3(self):
        self.assertRaises(ValueError, FlowCal.transform.to_compensated, 
                    self.d, [0, 1, 2], self.a0, self.A[0:2,0:2], [0, 1, 2])

    def test_channel_error(self):
        self.assertRaises(ValueError, FlowCal.transform.to_compensated, 
                    self.d, [0, 2], self.a0[0:2], self.A[0:2,0:2], [0, 1])

    def test_compensation_full(self):
        dt = FlowCal.transform.to_compensated(
            self.d, [0, 1, 2], self.a0, self.A, [0, 1, 2])
        np.testing.assert_array_almost_equal(
            dt,
            np.linalg.solve(self.A, (self.d - self.a0).T).T)

    def test_compensation_full_channels_none(self):
        dt = FlowCal.transform.to_compensated(
            self.d, None, self.a0, self.A, [0, 1, 2])
        np.testing.assert_array_almost_equal(dt, self.dt_complete)

    def test_compensation_full_comp_channels_none(self):
        dt = FlowCal.transform.to_compensated(
            self.d, None, self.a0, self.A, None)
        np.testing.assert_array_almost_equal(dt, self.dt_complete)

    def test_compensation_subset_channels_1(self):
        dt = FlowCal.transform.to_compensated(
            self.d, [0], self.a0, self.A, [0, 1, 2])
        np.testing.assert_array_almost_equal(dt[:,0], self.dt_complete[:,0])
        np.testing.assert_array_almost_equal(dt[:,[1,2]], self.d[:,[1,2]])

    def test_compensation_subset_channels_2(self):
        dt = FlowCal.transform.to_compensated(
            self.d, [0, 1], self.a0, self.A, [0, 1, 2])
        np.testing.assert_array_almost_equal(
            dt[:,[0, 1]], self.dt_complete[:,[0, 1]])
        np.testing.assert_array_almost_equal(dt[:,2], self.d[:,2])

    def test_compensation_subset_channels_3(self):
        dt = FlowCal.transform.to_compensated(
            self.d, [0, 2], self.a0, self.A, [0, 1, 2])
        np.testing.assert_array_almost_equal(
            dt[:,[0, 2]], self.dt_complete[:,[0, 2]])
        np.testing.assert_array_almost_equal(dt[:,1], self.d[:,1])

class TestCompensateFCS(unittest.TestCase):
    def setUp(self):
        # Data
        self.d = FlowCal.io.FCSData('test/Data001.fcs')
        self.channel_names = ['FSC-H', 'SSC-H', 'FL1-H', 
                                'FL2-H', 'FL3-H', 'Time']
        self.n_samples = self.d.shape[0]
        # Compensation coefficients
        self.a0 = np.array([1, 0.5, 2.1])
        self.A = np.array([[1, 0.3, 0.2], [0.6, 1, 0.5], [0.7, 0.8, 1]])
        # Expected results
        self.dt_complete = np.linalg.solve(
            self.A,
            (self.d[:, ['FL1-H', 'FL2-H', 'FL3-H']] - self.a0).T,
        ).T
        range_initial = self.d.range(['FL1-H', 'FL2-H', 'FL3-H'])
        self.range_complete = np.linalg.solve(
            self.A,
            (np.array(range_initial).T - self.a0).T,
        )

    def test_length_error_1(self):
        self.assertRaises(
            ValueError,
            FlowCal.transform.to_compensated, 
            self.d,
            ['FL1-H', 'FL2-H', 'FL3-H'],
            self.a0,
            self.A,
            ['FL1-H', 'FL2-H'],
        )

    def test_length_error_2(self):
        self.assertRaises(
            ValueError,
            FlowCal.transform.to_compensated, 
            self.d,
            ['FL1-H', 'FL2-H', 'FL3-H'],
            self.a0[0:1],
            self.A,
            ['FL1-H', 'FL2-H', 'FL3-H'],
        )

    def test_length_error_3(self):
        self.assertRaises(
            ValueError,
            FlowCal.transform.to_compensated, 
            self.d,
            ['FL1-H', 'FL2-H', 'FL3-H'],
            self.a0,
            self.A[0:2,0:2],
            ['FL1-H', 'FL2-H', 'FL3-H'],
        )

    def test_channel_error(self):
        self.assertRaises(
            ValueError,
            FlowCal.transform.to_compensated, 
            self.d,
            ['FL1-H', 'FL3-H'],
            self.a0[0:2],
            self.A[0:2,0:2],
            ['FL1-H', 'FL2-H'],
        )

    def test_compensation_full(self):
        dt = FlowCal.transform.to_compensated(
            self.d,
            ['FL1-H', 'FL2-H', 'FL3-H'],
            self.a0,
            self.A,
            ['FL1-H', 'FL2-H', 'FL3-H'],
        )
        # Check events
        np.testing.assert_array_almost_equal(
            dt[:, ['FL1-H', 'FL2-H', 'FL3-H']],
            self.dt_complete,
        )
        np.testing.assert_array_almost_equal(
            dt[:, ['FSC-H', 'SSC-H', 'Time']],
            self.d[:, ['FSC-H', 'SSC-H', 'Time']],
        )
        # Check range
        np.testing.assert_array_almost_equal(
            dt.range(['FL1-H', 'FL2-H', 'FL3-H']),
            self.range_complete,
        )
        np.testing.assert_array_almost_equal(
            dt.range(['FSC-H', 'SSC-H', 'Time']),
            self.d.range(['FSC-H', 'SSC-H', 'Time']),
        )

    def test_compensation_full_channels_none(self):
        dt = FlowCal.transform.to_compensated(
            self.d,
            None,
            self.a0,
            self.A,
            ['FL1-H', 'FL2-H', 'FL3-H'],
        )
        # Check events
        np.testing.assert_array_almost_equal(
            dt[:, ['FL1-H', 'FL2-H', 'FL3-H']],
            self.dt_complete,
        )
        np.testing.assert_array_almost_equal(
            dt[:, ['FSC-H', 'SSC-H', 'Time']],
            self.d[:, ['FSC-H', 'SSC-H', 'Time']],
        )
        # Check range
        np.testing.assert_array_almost_equal(
            dt.range(['FL1-H', 'FL2-H', 'FL3-H']),
            self.range_complete,
        )
        np.testing.assert_array_almost_equal(
            dt.range(['FSC-H', 'SSC-H', 'Time']),
            self.d.range(['FSC-H', 'SSC-H', 'Time']),
        )

    def test_compensation_subset_channels_1(self):
        dt = FlowCal.transform.to_compensated(
            self.d,
            ['FL1-H'],
            self.a0,
            self.A,
            ['FL1-H', 'FL2-H', 'FL3-H'],
        )
        # Check events
        np.testing.assert_array_almost_equal(
            dt[:, 'FL1-H'],
            self.dt_complete[:, 0],
        )
        np.testing.assert_array_almost_equal(
            dt[:, ['FSC-H', 'SSC-H', 'FL2-H', 'FL3-H', 'Time']],
            self.d[:, ['FSC-H', 'SSC-H', 'FL2-H', 'FL3-H', 'Time']],
        )
        # Check range
        np.testing.assert_array_almost_equal(
            dt.range('FL1-H'),
            self.range_complete[0,:],
        )
        np.testing.assert_array_almost_equal(
            dt.range(['FSC-H', 'SSC-H', 'FL2-H', 'FL3-H', 'Time']),
            self.d.range(['FSC-H', 'SSC-H', 'FL2-H', 'FL3-H', 'Time']),
        )

    def test_compensation_subset_channels_2(self):
        dt = FlowCal.transform.to_compensated(
            self.d,
            ['FL1-H', 'FL2-H'],
            self.a0,
            self.A,
            ['FL1-H', 'FL2-H', 'FL3-H'],
        )
        # Check events
        np.testing.assert_array_almost_equal(
            dt[:, ['FL1-H', 'FL2-H']],
            self.dt_complete[:, [0, 1]],
        )
        np.testing.assert_array_almost_equal(
            dt[:, ['FSC-H', 'SSC-H', 'FL3-H', 'Time']],
            self.d[:, ['FSC-H', 'SSC-H', 'FL3-H', 'Time']],
        )
        # Check range
        np.testing.assert_array_almost_equal(
            dt.range(['FL1-H', 'FL2-H']),
            self.range_complete[[0, 1], :],
        )
        np.testing.assert_array_almost_equal(
            dt.range(['FSC-H', 'SSC-H', 'FL3-H', 'Time']),
            self.d.range(['FSC-H', 'SSC-H', 'FL3-H', 'Time']),
        )

    def test_compensation_subset_channels_3(self):
        dt = FlowCal.transform.to_compensated(
            self.d,
            ['FL1-H', 'FL3-H'],
            self.a0,
            self.A,
            ['FL1-H', 'FL2-H', 'FL3-H'],
        )
        # Check events
        np.testing.assert_array_almost_equal(
            dt[:, ['FL1-H', 'FL3-H']],
            self.dt_complete[:, [0, 2]],
        )
        np.testing.assert_array_almost_equal(
            dt[:, ['FSC-H', 'SSC-H', 'FL2-H', 'Time']],
            self.d[:, ['FSC-H', 'SSC-H', 'FL2-H', 'Time']],
        )
        # Check range
        np.testing.assert_array_almost_equal(
            dt.range(['FL1-H', 'FL3-H']),
            self.range_complete[[0, 2], :],
        )
        np.testing.assert_array_almost_equal(
            dt.range(['FSC-H', 'SSC-H', 'FL2-H', 'Time']),
            self.d.range(['FSC-H', 'SSC-H', 'FL2-H', 'Time']),
        )

if __name__ == '__main__':
    unittest.main()
