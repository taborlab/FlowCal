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
import numpy
import unittest

class TestExponentiateArray(unittest.TestCase):
    def setUp(self):
        self.d = numpy.array([
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
        numpy.testing.assert_array_equal(self.d, db)

    def test_transform_all(self):
        dt = fc.transform.exponentiate(self.d)
        numpy.testing.assert_array_equal(dt, 10**(self.d/256.0))

    def test_transform_channel(self):
        dt = fc.transform.exponentiate(self.d, channels = 1)
        numpy.testing.assert_array_equal(dt[:,0], self.d[:,0])
        numpy.testing.assert_array_equal(dt[:,1], 10**(self.d[:,1]/256.0))
        numpy.testing.assert_array_equal(dt[:,2], self.d[:,2])

    def test_transform_channels(self):
        dt = fc.transform.exponentiate(self.d, channels = [1,2])
        numpy.testing.assert_array_equal(dt[:,0], self.d[:,0])
        numpy.testing.assert_array_equal(dt[:,1], 10**(self.d[:,1]/256.0))
        numpy.testing.assert_array_equal(dt[:,2], 10**(self.d[:,2]/256.0))

class TestExponentiateFCS(unittest.TestCase):
    def setUp(self):
        self.channel_names = ['FSC-H', 'SSC-H', 'FL1-H', 
                                'FL2-H', 'FL3-H', 'Time']
        self.filename = 'test/Data.001'
        self.d = fc.io.FCSData(self.filename)
        self.n_samples = self.d.shape[0]

    def test_transform_original_integrity(self):
        db = self.d.copy()
        dt = fc.transform.exponentiate(self.d)
        numpy.testing.assert_array_equal(self.d, db)

    def test_transform_all(self):
        dt = fc.transform.exponentiate(self.d)
        numpy.testing.assert_array_equal(dt, 10**(self.d/256.0))

    def test_transform_channel(self):
        dt = fc.transform.exponentiate(self.d, channels = 1)
        numpy.testing.assert_array_equal(dt[:,0], self.d[:,0])
        numpy.testing.assert_array_equal(dt[:,1], 10**(self.d[:,1]/256.0))
        numpy.testing.assert_array_equal(dt[:,2], self.d[:,2])
        numpy.testing.assert_array_equal(dt[:,3], self.d[:,3])
        numpy.testing.assert_array_equal(dt[:,4], self.d[:,4])
        numpy.testing.assert_array_equal(dt[:,5], self.d[:,5])

    def test_transform_channels(self):
        dt = fc.transform.exponentiate(self.d, channels = [1,2,4])
        numpy.testing.assert_array_equal(dt[:,0], self.d[:,0])
        numpy.testing.assert_array_equal(dt[:,1], 10**(self.d[:,1]/256.0))
        numpy.testing.assert_array_equal(dt[:,2], 10**(self.d[:,2]/256.0))
        numpy.testing.assert_array_equal(dt[:,3], self.d[:,3])
        numpy.testing.assert_array_equal(dt[:,4], 10**(self.d[:,4]/256.0))
        numpy.testing.assert_array_equal(dt[:,5], self.d[:,5])

    def test_transform_range_original_integrity(self):
        dt = fc.transform.exponentiate(self.d)
        ci = [self.d.channel_info[i]['range'] for i in range(5)]
        co = [[0,1023,1024]]*5
        self.assertEqual(ci, co)

    def test_transform_range_all(self):
        dt = fc.transform.exponentiate(self.d)
        cit = [dt.channel_info[i]['range'] for i in range(5)]
        co = [[1.,10**(1023/256.),1024]]*5
        self.assertEqual(cit, co)

    def test_transform_range_channel(self):
        dt = fc.transform.exponentiate(self.d, channels = 1)
        cit = [dt.channel_info[i]['range'] for i in range(5)]
        co = [[0,1023,1024],
              [1.,10**(1023/256.),1024],
              [0,1023,1024],
              [0,1023,1024],
              [0,1023,1024]]
        self.assertEqual(cit, co)

    def test_transform_range_channels(self):
        dt = fc.transform.exponentiate(self.d, channels = [1,2,4])
        cit = [dt.channel_info[i]['range'] for i in range(5)]
        co = [[0,1023,1024],
              [1.,10**(1023/256.),1024],
              [1.,10**(1023/256.),1024],
              [0,1023,1024],
              [1.,10**(1023/256.),1024]]
        self.assertEqual(cit, co)

    def test_transform_bins_original_integrity(self):
        dt = fc.transform.exponentiate(self.d)
        vi = [self.d.channel_info[i]['bin_vals'] for i in range(5)]
        ei = [self.d.channel_info[i]['bin_edges'] for i in range(5)]
        vo = [range(1024)]*5
        eo = [numpy.arange(-0.5, 1024.5, 1.0)]*5
        numpy.testing.assert_array_equal(vi, vo)
        numpy.testing.assert_array_equal(ei, eo)

    def test_transform_bins_all(self):
        dt = fc.transform.exponentiate(self.d)
        vit = [dt.channel_info[i]['bin_vals'] for i in range(5)]
        eit = [dt.channel_info[i]['bin_edges'] for i in range(5)]
        vo = [numpy.logspace(0., 1023/256., 1024)]*5
        eo = [numpy.logspace(-0.5/256., 1023.5/256., 1025)]*5
        numpy.testing.assert_array_equal(vit, vo)
        numpy.testing.assert_array_equal(eit, eo)

    def test_transform_bins_channel(self):
        dt = fc.transform.exponentiate(self.d, channels = 1)
        vit = [dt.channel_info[i]['bin_vals'] for i in range(5)]
        eit = [dt.channel_info[i]['bin_edges'] for i in range(5)]
        vo = [numpy.arange(1024),
              numpy.logspace(0., 1023/256., 1024),
              numpy.arange(1024),
              numpy.arange(1024),
              numpy.arange(1024),
              ]
        eo = [numpy.arange(-0.5, 1024.5, 1.0),
              numpy.logspace(-0.5/256., 1023.5/256., 1025),
              numpy.arange(-0.5, 1024.5, 1.0),
              numpy.arange(-0.5, 1024.5, 1.0),
              numpy.arange(-0.5, 1024.5, 1.0),
              ]
        numpy.testing.assert_array_equal(vit, vo)
        numpy.testing.assert_array_equal(eit, eo)

    def test_transform_bins_channels(self):
        dt = fc.transform.exponentiate(self.d, channels = [1,2,4])
        vit = [dt.channel_info[i]['bin_vals'] for i in range(5)]
        eit = [dt.channel_info[i]['bin_edges'] for i in range(5)]
        vo = [numpy.arange(1024),
              numpy.logspace(0., 1023/256., 1024),
              numpy.logspace(0., 1023/256., 1024),
              numpy.arange(1024),
              numpy.logspace(0., 1023/256., 1024),
              ]
        eo = [numpy.arange(-0.5, 1024.5, 1.0),
              numpy.logspace(-0.5/256., 1023.5/256., 1025),
              numpy.logspace(-0.5/256., 1023.5/256., 1025),
              numpy.arange(-0.5, 1024.5, 1.0),
              numpy.logspace(-0.5/256., 1023.5/256., 1025),
              ]
        numpy.testing.assert_array_equal(vit, vo)
        numpy.testing.assert_array_equal(eit, eo)
        pass

    def test_transform_channels_str(self):
        dt = fc.transform.exponentiate(self.d, channels = ['SSC-H', 
                                                        'FL1-H', 'FL3-H'])
        numpy.testing.assert_array_equal(dt[:,0], self.d[:,0])
        numpy.testing.assert_array_equal(dt[:,1], 10**(self.d[:,1]/256.0))
        numpy.testing.assert_array_equal(dt[:,2], 10**(self.d[:,2]/256.0))
        numpy.testing.assert_array_equal(dt[:,3], self.d[:,3])
        numpy.testing.assert_array_equal(dt[:,4], 10**(self.d[:,4]/256.0))
        numpy.testing.assert_array_equal(dt[:,5], self.d[:,5])

class TestMefArray(unittest.TestCase):
    def setUp(self):
        self.d = numpy.array([
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
        self.sc2 = lambda x: numpy.log(x)

    def test_mef_length_error(self):
        self.assertRaises(AssertionError, fc.transform.to_mef, 
                            self.d, 1, [self.sc1], [1,2])

    def test_mef_channel_error(self):
        self.assertRaises(ValueError, fc.transform.to_mef, 
                            self.d, 0, [self.sc1, self.sc2], [1,2])

    def test_mef_1d_1(self):
        dt = fc.transform.to_mef(self.d, 1, [self.sc1, self.sc2], [1,2])
        numpy.testing.assert_array_equal(dt[:,0], self.d[:,0])
        numpy.testing.assert_array_equal(dt[:,1], self.d[:,1]**2.)
        numpy.testing.assert_array_equal(dt[:,2], self.d[:,2])

    def test_mef_1d_2(self):
        dt = fc.transform.to_mef(self.d, 2, [self.sc1, self.sc2], [1,2])
        numpy.testing.assert_array_equal(dt[:,0], self.d[:,0])
        numpy.testing.assert_array_equal(dt[:,1], self.d[:,1])
        numpy.testing.assert_array_equal(dt[:,2], numpy.log(self.d[:,2]))

    def test_mef_2d(self):
        dt = fc.transform.to_mef(self.d, [1,2], [self.sc1, self.sc2], [1,2])
        numpy.testing.assert_array_equal(dt[:,0], self.d[:,0])
        numpy.testing.assert_array_equal(dt[:,1], self.d[:,1]**2.)
        numpy.testing.assert_array_equal(dt[:,2], numpy.log(self.d[:,2]))

    def test_mef_default_channel(self):
        dt = fc.transform.to_mef(self.d, None, [self.sc1, self.sc2], [1,2])
        numpy.testing.assert_array_equal(dt[:,0], self.d[:,0])
        numpy.testing.assert_array_equal(dt[:,1], self.d[:,1]**2.)
        numpy.testing.assert_array_equal(dt[:,2], numpy.log(self.d[:,2]))

    def test_mef_default_sc_channel(self):
        dt = fc.transform.to_mef(self.d, None, 
            [self.sc0, self.sc1, self.sc2], None)
        numpy.testing.assert_array_equal(dt[:,0], self.d[:,0] + 10)
        numpy.testing.assert_array_equal(dt[:,1], self.d[:,1]**2.)
        numpy.testing.assert_array_equal(dt[:,2], numpy.log(self.d[:,2]))

    def test_mef_default_sc_channel_error(self):
        self.assertRaises(AssertionError, fc.transform.to_mef, 
                            self.d, None, [self.sc1, self.sc2], None)

class TestMefFCS(unittest.TestCase):
    def setUp(self):
        self.channel_names = ['FSC-H', 'SSC-H', 'FL1-H', 
                                'FL2-H', 'FL3-H', 'Time']
        self.d = fc.io.FCSData('test/Data.001')
        self.n_samples = self.d.shape[0]
        self.sc0 = lambda x: x + 10
        self.sc1 = lambda x: x**2
        self.sc2 = lambda x: numpy.log(x + 1)

    def test_mef_length_error(self):
        self.assertRaises(AssertionError, fc.transform.to_mef, 
                    self.d, 'FL1-H', [self.sc1], ['FL1-H','FL3-H'])

    def test_mef_channel_error(self):
        self.assertRaises(ValueError, fc.transform.to_mef, 
                    self.d, 'FSC-H', [self.sc1, self.sc2], ['FL1-H','FL3-H'])

    def test_mef_1d_1(self):
        dt = fc.transform.to_mef(self.d, 'FL1-H', 
            [self.sc1, self.sc2], ['FL1-H','FL3-H'])
        numpy.testing.assert_array_equal(dt[:,0], self.d[:,0])
        numpy.testing.assert_array_equal(dt[:,1], self.d[:,1])
        numpy.testing.assert_array_equal(dt[:,2], self.d[:,2]**2.)
        numpy.testing.assert_array_equal(dt[:,3], self.d[:,3])
        numpy.testing.assert_array_equal(dt[:,4], self.d[:,4])

    def test_mef_1d_2(self):
        dt = fc.transform.to_mef(self.d, 'FL3-H', 
            [self.sc1, self.sc2], ['FL1-H','FL3-H'])
        numpy.testing.assert_array_equal(dt[:,0], self.d[:,0])
        numpy.testing.assert_array_equal(dt[:,1], self.d[:,1])
        numpy.testing.assert_array_equal(dt[:,2], self.d[:,2])
        numpy.testing.assert_array_equal(dt[:,3], self.d[:,3])
        numpy.testing.assert_array_equal(dt[:,4], 
            numpy.log(self.d[:,4].astype(numpy.float64) + 1))

    def test_mef_2d(self):
        dt = fc.transform.to_mef(self.d, ['FL1-H','FL3-H'], 
            [self.sc1, self.sc2], ['FL1-H','FL3-H'])
        numpy.testing.assert_array_equal(dt[:,0], self.d[:,0])
        numpy.testing.assert_array_equal(dt[:,1], self.d[:,1])
        numpy.testing.assert_array_equal(dt[:,2], self.d[:,2]**2.)
        numpy.testing.assert_array_equal(dt[:,3], self.d[:,3])
        numpy.testing.assert_array_equal(dt[:,4], 
            numpy.log(self.d[:,4].astype(numpy.float64) + 1))

    def test_mef_default_channel(self):
        dt = fc.transform.to_mef(self.d, None, 
            [self.sc1, self.sc2], ['FL1-H','FL3-H'])
        numpy.testing.assert_array_equal(dt[:,0], self.d[:,0])
        numpy.testing.assert_array_equal(dt[:,1], self.d[:,1])
        numpy.testing.assert_array_equal(dt[:,2], self.d[:,2]**2.)
        numpy.testing.assert_array_equal(dt[:,3], self.d[:,3])
        numpy.testing.assert_array_equal(dt[:,4], 
            numpy.log(self.d[:,4].astype(numpy.float64) + 1))

    def test_mef_range_channels(self):
        dt = fc.transform.to_mef(self.d, ['FL1-H','FL3-H'], 
            [self.sc1, self.sc2], ['FL1-H','FL3-H'])
        cit = [dt.channel_info[i]['range'] for i in range(5)]
        co = [[0, 1023, 1024],
              [0, 1023, 1024],
              [0**2, 1023**2, 1024],
              [0, 1023, 1024],
              [numpy.log(0 + 1), numpy.log(1023 + 1), 1024]]
        self.assertEqual(cit, co)

    def test_mef_bins_channels(self):
        dt = fc.transform.to_mef(self.d, ['FL1-H','FL3-H'], 
            [self.sc1, self.sc2], ['FL1-H','FL3-H'])
        vit = [dt.channel_info[i]['bin_vals'] for i in range(5)]
        eit = [dt.channel_info[i]['bin_edges'] for i in range(5)]
        vo = [numpy.arange(1024),
              numpy.arange(1024),
              numpy.arange(1024)**2,
              numpy.arange(1024),
              numpy.log(numpy.arange(1024) + 1),
              ]
        eo = [numpy.arange(-0.5, 1024.5, 1.0),
              numpy.arange(-0.5, 1024.5, 1.0),
              numpy.arange(-0.5, 1024.5, 1.0)**2,
              numpy.arange(-0.5, 1024.5, 1.0),
              numpy.log(numpy.arange(-0.5, 1024.5, 1.0) + 1),
              ]
        numpy.testing.assert_array_equal(vit, vo)
        numpy.testing.assert_array_equal(eit, eo)

if __name__ == '__main__':
    unittest.main()
