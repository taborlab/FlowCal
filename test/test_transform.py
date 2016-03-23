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
                                      max_range=[1024, 1024],
                                      amplification_type=[(0,0), (0,0)])
        np.testing.assert_array_equal(self.d, db)

    def test_rfi_arg_error_max_range(self):
        self.assertRaises(ValueError, FlowCal.transform.to_rfi, 
                          self.d, [0,1], None, [(0,0), (0,0)])

    def test_rfi_arg_error_amplification_type(self):
        self.assertRaises(ValueError, FlowCal.transform.to_rfi, 
                          self.d, [0,1], [1024, 1024], None)

    def test_rfi_length_error_max_range(self):
        self.assertRaises(ValueError, FlowCal.transform.to_rfi, 
                          self.d, [0,1], [1024], [(0,0), (0,0)])

    def test_rfi_length_error_amplification_type(self):
        self.assertRaises(ValueError, FlowCal.transform.to_rfi, 
                          self.d, [0,1], [1024, 1024], [(0,0), (0,0), (4,1)])

    def test_rfi_1d_1(self):
        dt = FlowCal.transform.to_rfi(self.d,
                                      channels=1,
                                      max_range=1024,
                                      amplification_type=(4, 1))
        np.testing.assert_array_equal(dt[:,0], self.d[:,0])
        np.testing.assert_array_equal(dt[:,1], 10**(self.d[:,1]/256.0))
        np.testing.assert_array_equal(dt[:,2], self.d[:,2])

    def test_rfi_1d_2(self):
        dt = FlowCal.transform.to_rfi(self.d,
                                      channels=2,
                                      max_range=256,
                                      amplification_type=(2, 0.01))
        np.testing.assert_array_equal(dt[:,0], self.d[:,0])
        np.testing.assert_array_equal(dt[:,1], self.d[:,1])
        np.testing.assert_array_equal(dt[:,2], 0.01*10**(self.d[:,2]/128.0))

    def test_rfi_2d_1(self):
        dt = FlowCal.transform.to_rfi(self.d,
                                      channels=[1,2],
                                      max_range=[1024, 256],
                                      amplification_type=[(4, 1), (2, 0.01)])
        np.testing.assert_array_equal(dt[:,0], self.d[:,0])
        np.testing.assert_array_equal(dt[:,1], 10**(self.d[:,1]/256.0))
        np.testing.assert_array_equal(dt[:,2], 0.01*10**(self.d[:,2]/128.0))

    def test_rfi_2d_2(self):
        dt = FlowCal.transform.to_rfi(self.d,
                                      channels=[1,2],
                                      max_range=[1024, 1024],
                                      amplification_type=[(4, 1), (0, 0)])
        np.testing.assert_array_equal(dt[:,0], self.d[:,0])
        np.testing.assert_array_equal(dt[:,1], 10**(self.d[:,1]/256.0))
        np.testing.assert_array_equal(dt[:,2], self.d[:,2])

    def test_rfi_default_channel(self):
        dt = FlowCal.transform.to_rfi(self.d,
                                      max_range=[1024]*3,
                                      amplification_type=[(4,1)]*3)
        np.testing.assert_array_equal(dt, 10**(self.d/256.0))

class TestRFIFCS(unittest.TestCase):
    def setUp(self):
        self.channel_names = ['FSC-H', 'SSC-H', 'FL1-H', 
                                'FL2-H', 'FL3-H', 'Time']
        self.d = FlowCal.io.FCSData('test/Data001.fcs')
        self.n_samples = self.d.shape[0]

    def test_rfi_original_integrity(self):
        db = self.d.copy()
        dt = FlowCal.transform.to_rfi(self.d,
                                      channels=['FSC-H', 'SSC-H'],
                                      max_range=[1024, 1024],
                                      amplification_type=[(4,1), (4,1)])
        np.testing.assert_array_equal(self.d, db)

    def test_rfi_length_error_max_range(self):
        self.assertRaises(ValueError,
                          FlowCal.transform.to_rfi, 
                          self.d,
                          ['FSC-H', 'SSC-H'],
                          [1024],
                          [(4,1), (4,1)])

    def test_rfi_length_error_amplification_type(self):
        self.assertRaises(ValueError,
                          FlowCal.transform.to_rfi, 
                          self.d,
                          ['FSC-H', 'SSC-H'],
                          [1024, 1024],
                          [(4,1), (4,1), (0,0)])

    def test_rfi_1d_1(self):
        dt = FlowCal.transform.to_rfi(self.d,
                                      channels='FL1-H',
                                      max_range=512,
                                      amplification_type=(4, 0.01))
        np.testing.assert_array_equal(dt[:,'FSC-H'], self.d[:,'FSC-H'])
        np.testing.assert_array_equal(dt[:,'SSC-H'], self.d[:,'SSC-H'])
        np.testing.assert_array_equal(dt[:,'FL1-H'],
                                      0.01*10**(self.d[:,'FL1-H']/128.0))
        np.testing.assert_array_equal(dt[:,'FL2-H'], self.d[:,'FL2-H'])
        np.testing.assert_array_equal(dt[:,'FL3-H'], self.d[:,'FL3-H'])
        np.testing.assert_array_equal(dt[:,'Time'], self.d[:,'Time'])

    def test_rfi_1d_2(self):
        dt = FlowCal.transform.to_rfi(self.d,
                                      channels=2,
                                      max_range=512,
                                      amplification_type=(2, 0.01))
        np.testing.assert_array_equal(dt[:,'FSC-H'], self.d[:,'FSC-H'])
        np.testing.assert_array_equal(dt[:,'SSC-H'], self.d[:,'SSC-H'])
        np.testing.assert_array_equal(dt[:,'FL1-H'],
                                      0.01*10**(self.d[:,'FL1-H']/256.0))
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

    def test_rfi_2d_1(self):
        dt = FlowCal.transform.to_rfi(self.d,
                                      channels=['FL1-H', 'FL3-H'],
                                      max_range=[512, 2048],
                                      amplification_type=[(4, 0.01), (2, 1)])
        np.testing.assert_array_equal(dt[:,'FSC-H'], self.d[:,'FSC-H'])
        np.testing.assert_array_equal(dt[:,'SSC-H'], self.d[:,'SSC-H'])
        np.testing.assert_array_equal(dt[:,'FL1-H'],
                                      0.01*10**(self.d[:,'FL1-H']/128.0))
        np.testing.assert_array_equal(dt[:,'FL2-H'], self.d[:,'FL2-H'])
        np.testing.assert_array_equal(dt[:,'FL3-H'],
                                      10**(self.d[:,'FL3-H']/1024.))
        np.testing.assert_array_equal(dt[:,'Time'], self.d[:,'Time'])

    def test_rfi_2d_2(self):
        dt = FlowCal.transform.to_rfi(self.d,
                                      channels=[2, 4],
                                      max_range=[512, 1024],
                                      amplification_type=[(4, 0.01), (0, 0)])
        np.testing.assert_array_equal(dt[:,'FSC-H'], self.d[:,'FSC-H'])
        np.testing.assert_array_equal(dt[:,'SSC-H'], self.d[:,'SSC-H'])
        np.testing.assert_array_equal(dt[:,'FL1-H'],
                                      0.01*10**(self.d[:,'FL1-H']/128.0))
        np.testing.assert_array_equal(dt[:,'FL2-H'], self.d[:,'FL2-H'])
        np.testing.assert_array_equal(dt[:,'FL3-H'], self.d[:,'FL3-H'])
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

    def test_rfi_2d_domain(self):
        dt = FlowCal.transform.to_rfi(self.d,
                                      channels=['FL1-H', 'FL3-H'],
                                      max_range=[512, 2048],
                                      amplification_type=[(4, 0.01), (2, 1)])
        np.testing.assert_array_equal(dt.domain('FSC-H'),
                                      self.d.domain('FSC-H'))
        np.testing.assert_array_equal(dt.domain('SSC-H'),
                                      self.d.domain('SSC-H'))
        np.testing.assert_array_equal(dt.domain('FL1-H'),
                                      0.01*10**(self.d.domain('FL1-H')/128.0))
        np.testing.assert_array_equal(dt.domain('FL2-H'),
                                      self.d.domain('FL2-H'))
        np.testing.assert_array_equal(dt.domain('FL3-H'),
                                      10**(self.d.domain('FL3-H')/1024.0))

    def test_rfi_2d_bin_edges(self):
        dt = FlowCal.transform.to_rfi(self.d,
                                      channels=['FL1-H', 'FL3-H'],
                                      max_range=[512, 2048],
                                      amplification_type=[(4, 0.01), (2, 1)])
        np.testing.assert_array_equal(dt.hist_bin_edges('FSC-H'),
                                      self.d.hist_bin_edges('FSC-H'))
        np.testing.assert_array_equal(dt.hist_bin_edges('SSC-H'),
                                      self.d.hist_bin_edges('SSC-H'))
        np.testing.assert_array_equal(dt.hist_bin_edges('FL1-H'),
                                      0.01*10**(self.d.hist_bin_edges('FL1-H')/128.0))
        np.testing.assert_array_equal(dt.hist_bin_edges('FL2-H'),
                                      self.d.hist_bin_edges('FL2-H'))
        np.testing.assert_array_equal(dt.hist_bin_edges('FL3-H'),
                                      10**(self.d.hist_bin_edges('FL3-H')/1024.0))

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
        vit = [dt.domain(i) for i in range(5)]
        eit = [dt.hist_bin_edges(i) for i in range(5)]
        vo = [np.arange(1024),
              np.arange(1024),
              np.arange(1024)**2,
              np.arange(1024),
              np.log(np.arange(1024) + 1),
              ]
        eo = [np.arange(-0.5, 1024.5, 1.0),
              np.arange(-0.5, 1024.5, 1.0),
              np.arange(-0.5, 1024.5, 1.0)**2,
              np.arange(-0.5, 1024.5, 1.0),
              np.log(np.arange(-0.5, 1024.5, 1.0) + 1),
              ]
        np.testing.assert_array_equal(vit, vo)
        np.testing.assert_array_equal(eit, eo)

if __name__ == '__main__':
    unittest.main()
