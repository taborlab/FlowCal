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
        self.d = fc.io.TaborLabFCSData(self.filename)
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

    def test_transform_channels_str(self):
        dt = fc.transform.exponentiate(self.d, channels = ['SSC-H', 
                                                        'FL1-H', 'FL3-H'])
        numpy.testing.assert_array_equal(dt[:,0], self.d[:,0])
        numpy.testing.assert_array_equal(dt[:,1], 10**(self.d[:,1]/256.0))
        numpy.testing.assert_array_equal(dt[:,2], 10**(self.d[:,2]/256.0))
        numpy.testing.assert_array_equal(dt[:,3], self.d[:,3])
        numpy.testing.assert_array_equal(dt[:,4], 10**(self.d[:,4]/256.0))
        numpy.testing.assert_array_equal(dt[:,5], self.d[:,5])

if __name__ == '__main__':
    unittest.main()
