#!/usr/bin/python
#
# test_io.py - Unit tests for io module
#
# Author: Sebastian M. Castillo-Hair (smc9@rice.edu)
# Date: 8/10/2015
#
# Requires:
#   * fc.io
#   * numpy
#

import fc.io
import numpy as np
import unittest

'''
Files to test:
    - Data001.fcs: FCS 2.0 from CellQuest Pro 5.1.1 / BD FACScan Flow Cytometer
    - Data002.fcs: FCS 2.0 from FACSDiva 6.1.3 / BD FACSCanto II Flow Cytometer
    - Data003.fcs: FCS 3.0 from FlowJo Collectors Edition 7.5 / 
                    BD FACScan Flow Cytometer
    - Data004.fcs: FCS 3.0 including floating-point data
'''
filenames = ['test/Data001.fcs',
            'test/Data002.fcs',
            'test/Data003.fcs',
            'test/Data004.fcs',
            ]

class TestFCSDataLoading(unittest.TestCase):
    def setUp(self):
        pass

    def test_loading_1(self):
        '''
        Testing proper loading from FCS file (2.0, CellQuest Pro).
        '''
        d = fc.io.FCSData(filenames[0])
        self.assertEqual(d.shape, (20949, 6))
        self.assertEqual(len(d.channel_info), 6)
        self.assertEqual(d.channels,
            ['FSC-H',
             'SSC-H',
             'FL1-H',
             'FL2-H',
             'FL3-H',
             'Time'])

    def test_loading_2(self):
        '''
        Testing proper loading from FCS file (2.0, FACSDiva).
        '''
        d = fc.io.FCSData(filenames[1])
        self.assertEqual(d.shape, (20000, 9))
        self.assertEqual(len(d.channel_info), 9)
        self.assertEqual(d.channels,
            ['FSC-A',
             'SSC-A',
             'FITC-A',
             'PE-A',
             'PerCP-Cy5-5-A',
             'PE-Cy7-A',
             'APC-A',
             'APC-Cy7-A',
             'Time',
            ])

    def test_loading_3(self):
        '''
        Testing proper loading from FCS file (3.0, FlowJo).
        '''
        d = fc.io.FCSData(filenames[2])
        self.assertEqual(d.shape, (25000, 8))
        self.assertEqual(len(d.channel_info), 8)
        self.assertEqual(d.channels,
            ['TIME',
             'FSC',
             'SSC',
             'FL1',
             'FL2',
             'FL3',
             'FSCW',
             'FSCA',
            ])

    def test_loading_4(self):
        '''
        Testing proper loading from FCS file (3.0, Floating-point).
        '''
        d = fc.io.FCSData(filenames[3])
        self.assertEqual(d.shape, (50000, 14))
        self.assertEqual(len(d.channel_info), 14)
        self.assertEqual(d.channels,
            ['FSC-A',
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
            'Time',
            ])

class TestFCSMetadata(unittest.TestCase):
    def setUp(self):
        pass
        
    def test_metadata_default(self):
        '''
        Test proper initialization of default metadata.
        '''
        d = fc.io.FCSData(filenames[0])
        self.assertEqual(d.metadata, {})

    def test_metadata_explicit(self):
        '''
        Test proper initialization of explicit metadata.
        '''
        d = fc.io.FCSData(filenames[0], {'l2': 4, 'a': 'r'})
        self.assertEqual(d.metadata, {'l2': 4, 'a': 'r'})

class TestTaborLabFCSAttributes(unittest.TestCase):
    def setUp(self):
        self.d = fc.io.FCSData(filenames[0])
        self.n_samples = self.d.shape[0]

    def test_range(self):
        '''
        Testing proper loading of range information
        '''
        self.assertEqual(self.d.channel_info[0]['range'], [0, 1023, 1024])
        self.assertEqual(self.d.channel_info[1]['range'], [0, 1023, 1024])
        self.assertEqual(self.d.channel_info[2]['range'], [0, 1023, 1024])
        self.assertEqual(self.d.channel_info[3]['range'], [0, 1023, 1024])
        self.assertEqual(self.d.channel_info[4]['range'], [0, 1023, 1024])

    def test_bins(self):
        '''
        Testing proper creation of bins
        '''
        # Bin values
        np.testing.assert_array_equal(self.d.channel_info[0]['bin_vals'], 
            np.arange(1024))
        np.testing.assert_array_equal(self.d.channel_info[1]['bin_vals'], 
            np.arange(1024))
        np.testing.assert_array_equal(self.d.channel_info[2]['bin_vals'], 
            np.arange(1024))
        np.testing.assert_array_equal(self.d.channel_info[3]['bin_vals'], 
            np.arange(1024))
        np.testing.assert_array_equal(self.d.channel_info[4]['bin_vals'], 
            np.arange(1024))
        # Bin edges
        np.testing.assert_array_equal(self.d.channel_info[0]['bin_edges'], 
            np.arange(1025) - 0.5)
        np.testing.assert_array_equal(self.d.channel_info[1]['bin_edges'], 
            np.arange(1025) - 0.5)
        np.testing.assert_array_equal(self.d.channel_info[2]['bin_edges'], 
            np.arange(1025) - 0.5)
        np.testing.assert_array_equal(self.d.channel_info[3]['bin_edges'], 
            np.arange(1025) - 0.5)
        np.testing.assert_array_equal(self.d.channel_info[4]['bin_edges'], 
            np.arange(1025) - 0.5)

    def test_str(self):
        '''
        Testing string representation.
        '''
        self.assertEqual(str(self.d), 'Data001.fcs')

    def test_time_step(self):
        '''
        Testing of the time step
        '''
        # Data.001 is a FCS2.0 file, use the timeticks parameter.
        # We have previously looked at self.d.text['TIMETICKS']) to determine
        # the correct output for this file.
        self.assertEqual(self.d.time_step, 0.2)

    def test_acquisition_time_event(self):
        '''
        Testing acquisition time
        '''
        # Data.001 has the time channel, so the acquisition time should be
        # calculated using the event list.
        # We have previously looked at self.d[0, 'Time'] and self.d[-1, 'Time']
        # to determine the correct output for this file.
        self.assertEqual(self.d.acquisition_time, 74.8)

    def test_acquisition_time_btim_etim(self):
        '''
        Testing acquisition time using the btim/etim method
        '''
        # Data.001 has the time channel, so we will remove it so that the
        # BTIM and ETIM keyword arguments are used.
        # We have previously looked at d.text['$BTIM'] and d.text['$ETIM'] to
        # determine the correct output for this file.
        d = self.d[:,['FSC-H', 'SSC-H', 'FL1-H', 'FL2-H', 'FL3-H']]
        self.assertEqual(d.acquisition_time, 77)

class TestTaborLabFCSAttributes3(unittest.TestCase):
    def setUp(self):
        self.d = fc.io.FCSData(filenames[2])
        self.n_samples = self.d.shape[0]

    def test_range(self):
        '''
        Testing proper loading of range information
        '''
        self.assertEqual(self.d.channel_info[1]['range'], [0, 1023, 1024])
        self.assertEqual(self.d.channel_info[2]['range'], [0, 1023, 1024])
        self.assertEqual(self.d.channel_info[3]['range'], [0, 1023, 1024])
        self.assertEqual(self.d.channel_info[4]['range'], [0, 1023, 1024])
        self.assertEqual(self.d.channel_info[5]['range'], [0, 1023, 1024])

    def test_bins(self):
        '''
        Testing proper creation of bins
        '''
        # Bin values
        np.testing.assert_array_equal(self.d.channel_info[1]['bin_vals'], 
            np.arange(1024))
        np.testing.assert_array_equal(self.d.channel_info[2]['bin_vals'], 
            np.arange(1024))
        np.testing.assert_array_equal(self.d.channel_info[3]['bin_vals'], 
            np.arange(1024))
        np.testing.assert_array_equal(self.d.channel_info[4]['bin_vals'], 
            np.arange(1024))
        np.testing.assert_array_equal(self.d.channel_info[5]['bin_vals'], 
            np.arange(1024))
        # Bin edges
        np.testing.assert_array_equal(self.d.channel_info[1]['bin_edges'], 
            np.arange(1025) - 0.5)
        np.testing.assert_array_equal(self.d.channel_info[2]['bin_edges'], 
            np.arange(1025) - 0.5)
        np.testing.assert_array_equal(self.d.channel_info[3]['bin_edges'], 
            np.arange(1025) - 0.5)
        np.testing.assert_array_equal(self.d.channel_info[4]['bin_edges'], 
            np.arange(1025) - 0.5)
        np.testing.assert_array_equal(self.d.channel_info[5]['bin_edges'], 
            np.arange(1025) - 0.5)

    def test_str(self):
        '''
        Testing string representation.
        '''
        self.assertEqual(str(self.d), 'Data003.fcs')

    def test_time_step(self):
        '''
        Testing of the time step
        '''
        # Data.003 is a FCS3.0 file, use the $TIMESTEP parameter.
        # We have previously looked at self.d.text['$TIMESTEP']) to determine
        # the correct output for this file.
        self.assertEqual(self.d.time_step, 0.1)

    def test_acquisition_time_event(self):
        '''
        Testing acquisition time
        '''
        # Data.003 has the time channel, so the acquisition time should be
        # calculated using the event list.
        # We have previously looked at self.d[0, 'TIME'] and self.d[-1, 'TIME']
        # to determine the correct output for this file.
        self.assertEqual(self.d.acquisition_time, 134.8)

    def test_acquisition_time_btim_etim(self):
        '''
        Testing acquisition time using the btim/etim method
        '''
        # Data.001 has the time channel, so we will remove it so that the
        # BTIM and ETIM keyword arguments are used.
        # We have previously looked at d.text['$BTIM'] and d.text['$ETIM'] to
        # determine the correct output for this file.
        d = self.d[:,['FSC', 'SSC', 'FL1', 'FL2', 'FL3']]
        self.assertEqual(d.acquisition_time, 156)


class TestFCSDataSlicing(unittest.TestCase):
    def setUp(self):
        self.d = fc.io.FCSData(filenames[0], {'l2': 4, 'a': 'r'})
        self.n_samples = self.d.shape[0]

    def test_1d_slicing_with_scalar(self):
        '''
        Testing the 1D slicing with a scalar of a FCSData object.
        '''
        ds = self.d[1]
        self.assertIsInstance(ds, fc.io.FCSData)
        self.assertEqual(ds.shape, (6,))
        self.assertEqual(ds.channels,
            ['FSC-H', 'SSC-H', 'FL1-H', 'FL2-H', 'FL3-H', 'Time'])
        self.assertEqual(len(ds.channel_info), 6)

    def test_1d_slicing_with_list(self):
        '''
        Testing the 1D slicing with a list of a FCSData object.
        '''
        ds = self.d[range(10)]
        self.assertIsInstance(ds, fc.io.FCSData)
        self.assertEqual(ds.shape, (10,6))
        self.assertEqual(ds.channels,
            ['FSC-H', 'SSC-H', 'FL1-H', 'FL2-H', 'FL3-H', 'Time'])
        self.assertEqual(len(ds.channel_info), 6)

    def test_slicing_channel_with_int(self):
        '''
        Testing the channel slicing with an int of a FCSData object.
        '''
        ds = self.d[:,2]
        self.assertIsInstance(ds, fc.io.FCSData)
        self.assertEqual(ds.shape, (self.n_samples,))
        self.assertEqual(ds.channels, ['FL1-H'])
        self.assertEqual(len(ds.channel_info), 1)

    def test_slicing_channel_with_string(self):
        '''
        Testing the channel slicing with a string of a FCSData object.
        '''
        ds = self.d[:,'SSC-H']
        self.assertIsInstance(ds, fc.io.FCSData)
        self.assertEqual(ds.shape, (self.n_samples,))
        self.assertEqual(ds.channels, ['SSC-H'])
        self.assertEqual(len(ds.channel_info), 1)

    def test_slicing_channel_with_int_array(self):
        '''
        Testing the channel slicing with an int array of a FCSData 
        object.
        '''
        ds = self.d[:,[1,3]]
        self.assertIsInstance(ds, fc.io.FCSData)
        self.assertEqual(ds.shape, (self.n_samples,2))
        self.assertEqual(ds.channels, ['SSC-H', 'FL2-H'])
        self.assertEqual(len(ds.channel_info), 2)

    def test_slicing_channel_with_string_array(self):
        '''
        Testing the channel slicing with a string array of a FCSData 
        object.
        '''
        ds = self.d[:,['FSC-H', 'FL3-H']]
        self.assertIsInstance(ds, fc.io.FCSData)
        self.assertEqual(ds.shape, (self.n_samples,2))
        self.assertEqual(ds.channels, ['FSC-H', 'FL3-H'])
        self.assertEqual(len(ds.channel_info), 2)

    def test_slicing_sample(self):
        '''
        Testing the sample slicing of a FCSData object.
        '''
        ds = self.d[:1000]
        self.assertIsInstance(ds, fc.io.FCSData)
        self.assertEqual(ds.shape, (1000,6))
        self.assertEqual(ds.channels,
            ['FSC-H', 'SSC-H', 'FL1-H', 'FL2-H', 'FL3-H', 'Time'])
        self.assertEqual(len(ds.channel_info), 6)

    def test_2d_slicing(self):
        '''
        Testing 2D slicing of a FCSData object.
        '''
        ds = self.d[:1000,['SSC-H', 'FL3-H']]
        self.assertIsInstance(ds, fc.io.FCSData)
        self.assertEqual(ds.shape, (1000,2))
        self.assertEqual(ds.channels, ['SSC-H', 'FL3-H'])
        self.assertEqual(len(ds.channel_info), 2)

    def test_mask_slicing(self):
        '''
        Testing mask slicing of a FCSData object.
        '''
        m = self.d[:,1]>500
        ds = self.d[m,:]
        self.assertIsInstance(ds, fc.io.FCSData)
        self.assertEqual(ds.channels,
            ['FSC-H', 'SSC-H', 'FL1-H', 'FL2-H', 'FL3-H', 'Time'])
        self.assertEqual(len(ds.channel_info), 6)

    def test_none_slicing_1(self):
        '''
        Testing slicing with None on the first dimension of a FCSData 
        object.
        '''
        ds = self.d[None,[0,2]]
        self.assertIsInstance(ds, np.ndarray)

    def test_none_slicing_2(self):
        '''
        Testing slicing with None on the second dimension of a FCSData 
        object.
        '''
        ds = self.d[:,None]
        self.assertIsInstance(ds, np.ndarray)

    def test_2d_slicing_assignment(self):
        '''
        Test assignment to FCSData using slicing
        '''
        ds = self.d.copy()
        ds[:,[1,2]] = 5
        self.assertIsInstance(ds, fc.io.FCSData)
        self.assertEqual(ds.channels,
            ['FSC-H', 'SSC-H', 'FL1-H', 'FL2-H', 'FL3-H', 'Time'])
        np.testing.assert_array_equal(ds[:,0], self.d[:,0])
        np.testing.assert_array_equal(ds[:,1], 5)
        np.testing.assert_array_equal(ds[:,2], 5)
        np.testing.assert_array_equal(ds[:,3], self.d[:,3])
        np.testing.assert_array_equal(ds[:,4], self.d[:,4])

    def test_2d_slicing_assignment_string(self):
        '''
        Test assignment to FCSData using slicing with channel names
        '''
        ds = self.d.copy()
        ds[:,['SSC-H', 'FL1-H']] = 5
        self.assertIsInstance(ds, fc.io.FCSData)
        self.assertEqual(ds.channels,
            ['FSC-H', 'SSC-H', 'FL1-H', 'FL2-H', 'FL3-H', 'Time'])
        np.testing.assert_array_equal(ds[:,0], self.d[:,0])
        np.testing.assert_array_equal(ds[:,1], 5)
        np.testing.assert_array_equal(ds[:,2], 5)
        np.testing.assert_array_equal(ds[:,3], self.d[:,3])
        np.testing.assert_array_equal(ds[:,4], self.d[:,4])

    def test_metadata_slicing(self):
        '''
        Testing preservation of metadata upon slicing.
        '''
        ds = self.d[:1000,['SSC-H', 'FL3-H']]
        self.assertIsInstance(ds.metadata, dict)
        self.assertEqual(ds.metadata, {'l2': 4, 'a': 'r'})


class TestFCSDataOperations(unittest.TestCase):
    def setUp(self):
        self.d = fc.io.FCSData(filenames[0], {'l2': 4, 'a': 'r'})
        self.n_samples = self.d.shape[0]

    def test_sum_integer(self):
        '''
        Testing that scalar + FCSData is consistent with scalar + ndarray
        '''
        ds = self.d + 3
        ds_array = self.d.view(np.ndarray) + 3
        self.assertIsInstance(ds, fc.io.FCSData)
        np.testing.assert_array_equal(ds, ds_array)
        
    def test_sqrt(self):
        '''
        Testing that the sqrt(FCSData) is consistent with sqrt(ndarray)
        '''
        ds = np.sqrt(self.d)
        ds_array = np.sqrt(self.d.view(np.ndarray))
        self.assertIsInstance(ds, fc.io.FCSData)
        np.testing.assert_array_equal(ds, ds_array)

    def test_sum(self):
        '''
        Testing that the sum(FCSData) is consistent with sum(ndarray)
        '''
        s = np.sum(self.d)
        s_array = np.sum(self.d.view(np.ndarray))
        self.assertEqual(s, s_array)
        self.assertEqual(type(s), type(s_array))

    def test_mean(self):
        '''
        Testing that the mean(FCSData) is consistent with mean(ndarray)
        '''
        m = np.mean(self.d)
        m_array = np.mean(self.d.view(np.ndarray))
        self.assertEqual(m, m_array)
        self.assertEqual(type(m), type(m_array))

    def test_median(self):
        '''
        Testing that the median(FCSData) is consistent with median(ndarray)
        '''
        m = np.median(self.d)
        m_array = np.median(self.d.view(np.ndarray))
        self.assertEqual(m, m_array)
        self.assertEqual(type(m), type(m_array))

    def test_std(self):
        '''
        Testing that the std(FCSData) is consistent with std(ndarray)
        '''
        s = np.std(self.d)
        s_array = np.std(self.d.view(np.ndarray))
        self.assertEqual(s, s_array)
        self.assertEqual(type(s), type(s_array))

    def test_mean_axis(self):
        '''
        Testing that the mean along the axis 0 of a FCSData object 
        returns a FCSData object.
        '''
        m = np.mean(self.d, axis = 0)
        m_array = np.mean(self.d.view(np.ndarray), axis = 0)
        self.assertIsInstance(m, fc.io.FCSData)
        self.assertEqual(m.shape, m_array.shape)
        np.testing.assert_array_equal(m, m_array)

    def test_metadata_sqrt(self):
        '''
        Testing preservation of metadata after taking the square root.
        '''
        ds = np.sqrt(self.d)
        self.assertIsInstance(ds.metadata, dict)
        self.assertEqual(ds.metadata, {'l2': 4, 'a': 'r'})
        
if __name__ == '__main__':
    unittest.main()
