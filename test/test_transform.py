#!/usr/bin/python
#
# test_transform.py - Unit tests for transform module
#
# Author: Sebastian M. Castillo-Hair (smc9@rice.edu)
# Date: 7/1/2015
#
# Requires:
#   * fc.io
#   * fc.transform
#   * numpy
#

import fc.io
import fc.transform
import numpy as np
import unittest

class TestExponentiateArray(unittest.TestCase):
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

    def test_transform_original_integrity(self):
        db = self.d.copy()
        dt = fc.transform.exponentiate(self.d)
        np.testing.assert_array_equal(self.d, db)

    def test_transform_all(self):
        dt = fc.transform.exponentiate(self.d)
        np.testing.assert_array_equal(dt, 10**(self.d/256.0))

    def test_transform_channel(self):
        dt = fc.transform.exponentiate(self.d, channels = 1)
        np.testing.assert_array_equal(dt[:,0], self.d[:,0])
        np.testing.assert_array_equal(dt[:,1], 10**(self.d[:,1]/256.0))
        np.testing.assert_array_equal(dt[:,2], self.d[:,2])

    def test_transform_channels(self):
        dt = fc.transform.exponentiate(self.d, channels = [1,2])
        np.testing.assert_array_equal(dt[:,0], self.d[:,0])
        np.testing.assert_array_equal(dt[:,1], 10**(self.d[:,1]/256.0))
        np.testing.assert_array_equal(dt[:,2], 10**(self.d[:,2]/256.0))

class TestExponentiateFCS(unittest.TestCase):
    def setUp(self):
        self.channel_names = ['FSC-H', 'SSC-H', 'FL1-H', 
                                'FL2-H', 'FL3-H', 'Time']
        self.filename = 'test/Data001.fcs'
        self.d = fc.io.FCSData(self.filename)
        self.n_samples = self.d.shape[0]

    def test_transform_original_integrity(self):
        db = self.d.copy()
        dt = fc.transform.exponentiate(self.d)
        np.testing.assert_array_equal(self.d, db)

    def test_transform_all(self):
        dt = fc.transform.exponentiate(self.d)
        np.testing.assert_array_equal(dt, 10**(self.d/256.0))

    def test_transform_channel(self):
        dt = fc.transform.exponentiate(self.d, channels = 1)
        np.testing.assert_array_equal(dt[:,0], self.d[:,0])
        np.testing.assert_array_equal(dt[:,1], 10**(self.d[:,1]/256.0))
        np.testing.assert_array_equal(dt[:,2], self.d[:,2])
        np.testing.assert_array_equal(dt[:,3], self.d[:,3])
        np.testing.assert_array_equal(dt[:,4], self.d[:,4])
        np.testing.assert_array_equal(dt[:,5], self.d[:,5])

    def test_transform_channels(self):
        dt = fc.transform.exponentiate(self.d, channels = [1,2,4])
        np.testing.assert_array_equal(dt[:,0], self.d[:,0])
        np.testing.assert_array_equal(dt[:,1], 10**(self.d[:,1]/256.0))
        np.testing.assert_array_equal(dt[:,2], 10**(self.d[:,2]/256.0))
        np.testing.assert_array_equal(dt[:,3], self.d[:,3])
        np.testing.assert_array_equal(dt[:,4], 10**(self.d[:,4]/256.0))
        np.testing.assert_array_equal(dt[:,5], self.d[:,5])

    def test_transform_bins_original_integrity(self):
        dt = fc.transform.exponentiate(self.d)
        vi = [self.d.domain(i) for i in range(5)]
        ei = [self.d.hist_bin_edges(i) for i in range(5)]
        vo = [range(1024)]*5
        eo = [np.arange(-0.5, 1024.5, 1.0)]*5
        np.testing.assert_array_equal(vi, vo)
        np.testing.assert_array_equal(ei, eo)

    def test_transform_bins_all(self):
        dt = fc.transform.exponentiate(self.d)
        vit = [dt.domain(i) for i in range(5)]
        eit = [dt.hist_bin_edges(i) for i in range(5)]
        vo = [np.logspace(0., 1023/256., 1024)]*5
        eo = [np.logspace(-0.5/256., 1023.5/256., 1025)]*5
        np.testing.assert_array_equal(vit, vo)
        np.testing.assert_array_equal(eit, eo)

    def test_transform_bins_channel(self):
        dt = fc.transform.exponentiate(self.d, channels = 1)
        vit = [dt.domain(i) for i in range(5)]
        eit = [dt.hist_bin_edges(i) for i in range(5)]
        vo = [np.arange(1024),
              np.logspace(0., 1023/256., 1024),
              np.arange(1024),
              np.arange(1024),
              np.arange(1024),
              ]
        eo = [np.arange(-0.5, 1024.5, 1.0),
              np.logspace(-0.5/256., 1023.5/256., 1025),
              np.arange(-0.5, 1024.5, 1.0),
              np.arange(-0.5, 1024.5, 1.0),
              np.arange(-0.5, 1024.5, 1.0),
              ]
        np.testing.assert_array_equal(vit, vo)
        np.testing.assert_array_equal(eit, eo)

    def test_transform_bins_channels(self):
        dt = fc.transform.exponentiate(self.d, channels = [1,2,4])
        vit = [dt.domain(i) for i in range(5)]
        eit = [dt.hist_bin_edges(i) for i in range(5)]
        vo = [np.arange(1024),
              np.logspace(0., 1023/256., 1024),
              np.logspace(0., 1023/256., 1024),
              np.arange(1024),
              np.logspace(0., 1023/256., 1024),
              ]
        eo = [np.arange(-0.5, 1024.5, 1.0),
              np.logspace(-0.5/256., 1023.5/256., 1025),
              np.logspace(-0.5/256., 1023.5/256., 1025),
              np.arange(-0.5, 1024.5, 1.0),
              np.logspace(-0.5/256., 1023.5/256., 1025),
              ]
        np.testing.assert_array_equal(vit, vo)
        np.testing.assert_array_equal(eit, eo)
        pass

    def test_transform_channels_str(self):
        dt = fc.transform.exponentiate(self.d, channels = ['SSC-H', 
                                                        'FL1-H', 'FL3-H'])
        np.testing.assert_array_equal(dt[:,0], self.d[:,0])
        np.testing.assert_array_equal(dt[:,1], 10**(self.d[:,1]/256.0))
        np.testing.assert_array_equal(dt[:,2], 10**(self.d[:,2]/256.0))
        np.testing.assert_array_equal(dt[:,3], self.d[:,3])
        np.testing.assert_array_equal(dt[:,4], 10**(self.d[:,4]/256.0))
        np.testing.assert_array_equal(dt[:,5], self.d[:,5])

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
        self.assertRaises(ValueError, fc.transform.to_mef, 
                            self.d, 1, [self.sc1], [1,2])

    def test_mef_channel_error(self):
        self.assertRaises(ValueError, fc.transform.to_mef, 
                            self.d, 0, [self.sc1, self.sc2], [1,2])

    def test_mef_1d_1(self):
        dt = fc.transform.to_mef(self.d, 1, [self.sc1, self.sc2], [1,2])
        np.testing.assert_array_equal(dt[:,0], self.d[:,0])
        np.testing.assert_array_equal(dt[:,1], self.d[:,1]**2.)
        np.testing.assert_array_equal(dt[:,2], self.d[:,2])

    def test_mef_1d_2(self):
        dt = fc.transform.to_mef(self.d, 2, [self.sc1, self.sc2], [1,2])
        np.testing.assert_array_equal(dt[:,0], self.d[:,0])
        np.testing.assert_array_equal(dt[:,1], self.d[:,1])
        np.testing.assert_array_equal(dt[:,2], np.log(self.d[:,2]))

    def test_mef_2d(self):
        dt = fc.transform.to_mef(self.d, [1,2], [self.sc1, self.sc2], [1,2])
        np.testing.assert_array_equal(dt[:,0], self.d[:,0])
        np.testing.assert_array_equal(dt[:,1], self.d[:,1]**2.)
        np.testing.assert_array_equal(dt[:,2], np.log(self.d[:,2]))

    def test_mef_default_channel(self):
        dt = fc.transform.to_mef(self.d, None, [self.sc1, self.sc2], [1,2])
        np.testing.assert_array_equal(dt[:,0], self.d[:,0])
        np.testing.assert_array_equal(dt[:,1], self.d[:,1]**2.)
        np.testing.assert_array_equal(dt[:,2], np.log(self.d[:,2]))

    def test_mef_default_sc_channel(self):
        dt = fc.transform.to_mef(self.d, None, 
            [self.sc0, self.sc1, self.sc2], None)
        np.testing.assert_array_equal(dt[:,0], self.d[:,0] + 10)
        np.testing.assert_array_equal(dt[:,1], self.d[:,1]**2.)
        np.testing.assert_array_equal(dt[:,2], np.log(self.d[:,2]))

    def test_mef_default_sc_channel_error(self):
        self.assertRaises(ValueError, fc.transform.to_mef, 
                            self.d, None, [self.sc1, self.sc2], None)

class TestMefFCS(unittest.TestCase):
    def setUp(self):
        self.channel_names = ['FSC-H', 'SSC-H', 'FL1-H', 
                                'FL2-H', 'FL3-H', 'Time']
        self.d = fc.io.FCSData('test/Data001.fcs')
        self.n_samples = self.d.shape[0]
        self.sc0 = lambda x: x + 10
        self.sc1 = lambda x: x**2
        self.sc2 = lambda x: np.log(x + 1)

    def test_mef_length_error(self):
        self.assertRaises(ValueError, fc.transform.to_mef, 
                    self.d, 'FL1-H', [self.sc1], ['FL1-H','FL3-H'])

    def test_mef_channel_error(self):
        self.assertRaises(ValueError, fc.transform.to_mef, 
                    self.d, 'FSC-H', [self.sc1, self.sc2], ['FL1-H','FL3-H'])

    def test_mef_1d_1(self):
        dt = fc.transform.to_mef(self.d, 'FL1-H', 
            [self.sc1, self.sc2], ['FL1-H','FL3-H'])
        np.testing.assert_array_equal(dt[:,0], self.d[:,0])
        np.testing.assert_array_equal(dt[:,1], self.d[:,1])
        np.testing.assert_array_equal(dt[:,2], self.d[:,2]**2.)
        np.testing.assert_array_equal(dt[:,3], self.d[:,3])
        np.testing.assert_array_equal(dt[:,4], self.d[:,4])

    def test_mef_1d_2(self):
        dt = fc.transform.to_mef(self.d, 'FL3-H', 
            [self.sc1, self.sc2], ['FL1-H','FL3-H'])
        np.testing.assert_array_equal(dt[:,0], self.d[:,0])
        np.testing.assert_array_equal(dt[:,1], self.d[:,1])
        np.testing.assert_array_equal(dt[:,2], self.d[:,2])
        np.testing.assert_array_equal(dt[:,3], self.d[:,3])
        np.testing.assert_array_equal(dt[:,4], 
            np.log(self.d[:,4].astype(np.float64) + 1))

    def test_mef_2d(self):
        dt = fc.transform.to_mef(self.d, ['FL1-H','FL3-H'], 
            [self.sc1, self.sc2], ['FL1-H','FL3-H'])
        np.testing.assert_array_equal(dt[:,0], self.d[:,0])
        np.testing.assert_array_equal(dt[:,1], self.d[:,1])
        np.testing.assert_array_equal(dt[:,2], self.d[:,2]**2.)
        np.testing.assert_array_equal(dt[:,3], self.d[:,3])
        np.testing.assert_array_equal(dt[:,4], 
            np.log(self.d[:,4].astype(np.float64) + 1))

    def test_mef_default_channel(self):
        dt = fc.transform.to_mef(self.d, None, 
            [self.sc1, self.sc2], ['FL1-H','FL3-H'])
        np.testing.assert_array_equal(dt[:,0], self.d[:,0])
        np.testing.assert_array_equal(dt[:,1], self.d[:,1])
        np.testing.assert_array_equal(dt[:,2], self.d[:,2]**2.)
        np.testing.assert_array_equal(dt[:,3], self.d[:,3])
        np.testing.assert_array_equal(dt[:,4], 
            np.log(self.d[:,4].astype(np.float64) + 1))

    def test_mef_bins_channels(self):
        dt = fc.transform.to_mef(self.d, ['FL1-H','FL3-H'], 
            [self.sc1, self.sc2], ['FL1-H','FL3-H'])
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
