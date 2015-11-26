"""
Unit tests for the `io` module.

"""

import datetime
import unittest

import numpy as np

import fc.io

"""
Files to test:
    - Data001.fcs: FCS 2.0 from CellQuest Pro 5.1.1 / BD FACScan Flow Cytometer
    - Data002.fcs: FCS 2.0 from FACSDiva 6.1.3 / BD FACSCanto II Flow Cytometer
    - Data003.fcs: FCS 3.0 from FlowJo Collectors Edition 7.5 / 
                    BD FACScan Flow Cytometer
    - Data004.fcs: FCS 3.0 including floating-point data
"""
filenames = ['test/Data001.fcs',
            'test/Data002.fcs',
            'test/Data003.fcs',
            'test/Data004.fcs',
            ]

class TestFCSDataLoading(unittest.TestCase):
    def setUp(self):
        pass

    def test_loading_1(self):
        """
        Testing proper loading from FCS file (2.0, CellQuest Pro).

        """
        d = fc.io.FCSData(filenames[0])
        self.assertEqual(d.shape, (20949, 6))
        self.assertEqual(d.channels,
            ('FSC-H',
             'SSC-H',
             'FL1-H',
             'FL2-H',
             'FL3-H',
             'Time'))

    def test_loading_2(self):
        """
        Testing proper loading from FCS file (2.0, FACSDiva).

        """
        d = fc.io.FCSData(filenames[1])
        self.assertEqual(d.shape, (20000, 9))
        self.assertEqual(d.channels,
            ('FSC-A',
             'SSC-A',
             'FITC-A',
             'PE-A',
             'PerCP-Cy5-5-A',
             'PE-Cy7-A',
             'APC-A',
             'APC-Cy7-A',
             'Time',
            ))

    def test_loading_3(self):
        """
        Testing proper loading from FCS file (3.0, FlowJo).

        """
        d = fc.io.FCSData(filenames[2])
        self.assertEqual(d.shape, (25000, 8))
        self.assertEqual(d.channels,
            ('TIME',
             'FSC',
             'SSC',
             'FL1',
             'FL2',
             'FL3',
             'FSCW',
             'FSCA',
            ))

    def test_loading_4(self):
        """
        Testing proper loading from FCS file (3.0, Floating-point).

        """
        d = fc.io.FCSData(filenames[3])
        self.assertEqual(d.shape, (50000, 14))
        self.assertEqual(d.channels,
            ('FSC-A',
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
             ))

class TestFCSMetadata(unittest.TestCase):
    def setUp(self):
        pass
        
    def test_metadata_default(self):
        """
        Test proper initialization of default metadata.

        """
        d = fc.io.FCSData(filenames[0])
        self.assertEqual(d.metadata, {})

    def test_metadata_explicit(self):
        """
        Test proper initialization of explicit metadata.

        """
        d = fc.io.FCSData(filenames[0], {'l2': 4, 'a': 'r'})
        self.assertEqual(d.metadata, {'l2': 4, 'a': 'r'})

class TestFCSParseTimeString(unittest.TestCase):
    def test_parse_none(self):
        """
        Test that _parse_time_string() returns None when the input is None.

        """
        t = fc.io.FCSData._parse_time_string(None)
        self.assertEqual(t, None)

    def test_parse_fcs2(self):
        """
        Test that _parse_time_string() interprets the FCS2.0 time format.

        """
        t = fc.io.FCSData._parse_time_string("20:15:43")
        self.assertEqual(t, datetime.time(20, 15, 43))

    def test_parse_fcs3(self):
        """
        Test that _parse_time_string() interprets the FCS3.0 time format.

        """
        t = fc.io.FCSData._parse_time_string("20:15:43:20")
        self.assertEqual(t, datetime.time(20, 15, 43, 333333))

    def test_parse_fcs3_1(self):
        """
        Test that _parse_time_string() interprets the FCS3.1 time format.

        """
        t = fc.io.FCSData._parse_time_string("20:15:43.27")
        self.assertEqual(t, datetime.time(20, 15, 43, 270000))

    def test_parse_undefined(self):
        """
        Test that _parse_time_string() returns None for undefined inputs.

        """
        t = fc.io.FCSData._parse_time_string("i'm undefined")
        self.assertEqual(t, None)

class TestFCSAttributesDetectorVoltage(unittest.TestCase):
    """
    Test correct extraction, functioning, and slicing of detector_voltage.

    We have previously looked at the contents of the $PnV attribute for
    the test files ('BD$WORD{12 + n}' for Data.001) and identified the
    correct detector voltage output as follows:
        - Data.001: [1, 460, 400, 900, 999, 100]
        - Data.002: [305, 319, 478, 470, 583, 854, 800, 620, None]
        - Data.003: [None, 10.0, 460, 501, 501, 501, None, None]
        - Data.004: [250, 250, 250, 340, 340, 340, 550, 550, 550, 650, 650,
                     575, 575, None]

    """
    def setUp(self):
        self.d = [fc.io.FCSData(filenames[i]) for i in range(4)]
        # for di in self.d:
        #     print di
        #     print [di.text.get('$P{}N'.format(j + 1))
        #            for j in range(len(di.channels))]
        #     print [di.text.get('$P{}V'.format(j + 1))
        #            for j in range(len(di.channels))]

    def test_attribute(self):
        """
        Testing correct reporting of detector voltage.

        """
        self.assertEqual(self.d[0].detector_voltage(),
                         [1, 460, 400, 900, 999, 100])
        self.assertEqual(self.d[1].detector_voltage(),
                         [305, 319, 478, 470, 583, 854, 800, 620, None])
        self.assertEqual(self.d[2].detector_voltage(),
                         [None, 10.0, 460, 501, 501, 501, None, None])
        self.assertEqual(self.d[3].detector_voltage(),
                         [250, 250, 250, 340, 340, 340, 550, 550, 550, 650, 650,
                          575, 575, None])

    def test_attribute_single(self):
        """
        Testing correct reporting of detector voltage for a single channel.

        """
        self.assertEqual(self.d[0].detector_voltage('FSC-H'), 1)
        self.assertEqual(self.d[1].detector_voltage('FITC-A'), 478)
        self.assertEqual(self.d[2].detector_voltage('SSC'), 460)
        self.assertEqual(self.d[3].detector_voltage('GFP-A'), 650)

    def test_attribute_many(self):
        """
        Testing correct reporting of detector voltage for many channels.

        """
        self.assertEqual(self.d[0].detector_voltage(['SSC-H',
                                                     'FL2-H',
                                                     'FL3-H']),
                         [460, 900, 999])
        self.assertEqual(self.d[1].detector_voltage(['FITC-A',
                                                     'PE-A',
                                                     'PE-Cy7-A']),
                         [478, 470, 854])
        self.assertEqual(self.d[2].detector_voltage(['FSC',
                                                     'SSC',
                                                     'FL1']),
                         [10, 460, 501])
        self.assertEqual(self.d[3].detector_voltage(['FSC PMT-A',
                                                     'FSC PMT-H',
                                                     'FSC PMT-W']),
                         [550, 550, 550])

    def test_slice_single_str(self):
        """
        Testing correct reporting of detector voltage after slicing.

        """
        self.assertEqual(self.d[0][:, 'FSC-H'].detector_voltage(), [1])
        self.assertEqual(self.d[1][:, 'FITC-A'].detector_voltage(), [478])
        self.assertEqual(self.d[2][:, 'SSC'].detector_voltage(), [460])
        self.assertEqual(self.d[3][:, 'GFP-A'].detector_voltage(), [650])

    def test_slice_many_str(self):
        """
        Testing correct reporting of detector voltage after slicing.

        """
        self.assertEqual(self.d[0][:, ['SSC-H',
                                       'FL2-H',
                                       'FL3-H']].detector_voltage(),
                         [460, 900, 999])
        self.assertEqual(self.d[1][:, ['FITC-A',
                                       'PE-A',
                                       'PE-Cy7-A']].detector_voltage(),
                         [478, 470, 854])
        self.assertEqual(self.d[2][:, ['FSC',
                                       'SSC',
                                       'FL1']].detector_voltage(),
                         [10, 460, 501])
        self.assertEqual(self.d[3][:, ['FSC PMT-A',
                                       'FSC PMT-H',
                                       'FSC PMT-W']].detector_voltage(),
                         [550, 550, 550])

class TestFCSAttributesAmplifierGain(unittest.TestCase):
    """
    Test correct extraction, functioning, and slicing of amplifier_gain.

    We have previously looked at the contents of the $PnG attribute for
    the test files and identified the correct amplifier gain output as follows:
        - Data.001: [None, None, None, None, None, None]
        - Data.002: [None, None, None, None, None, None, None, None, None]
        - Data.003: [None, None, None, None, None, None, None, None]
        - Data.004: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                     1.0, 1.0, 0.01]

    """
    def setUp(self):
        self.d = [fc.io.FCSData(filenames[i]) for i in range(4)]
        # for di in self.d:
        #     print di
        #     print [di.text.get('$P{}N'.format(j + 1))
        #            for j in range(len(di.channels))]
        #     print [di.text.get('$P{}G'.format(j + 1))
        #            for j in range(len(di.channels))]

    def test_attribute(self):
        """
        Testing correct reporting of amplifier gain.

        """
        self.assertEqual(self.d[0].amplifier_gain(),
                         [None, None, None, None, None, None])
        self.assertEqual(self.d[1].amplifier_gain(),
                         [None, None, None, None, None, None, None, None, None])
        self.assertEqual(self.d[2].amplifier_gain(),
                         [None, None, None, None, None, None, None, None])
        self.assertEqual(self.d[3].amplifier_gain(),
                         [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                          1.0, 1.0, 0.01])

    def test_attribute_single(self):
        """
        Testing correct reporting of amplifier gain for a single channel.

        """
        self.assertEqual(self.d[0].amplifier_gain('FSC-H'), None)
        self.assertEqual(self.d[1].amplifier_gain('FITC-A'), None)
        self.assertEqual(self.d[2].amplifier_gain('SSC'), None)
        self.assertEqual(self.d[3].amplifier_gain('GFP-A'), 1.0)

    def test_attribute_many(self):
        """
        Testing correct reporting of amplifier gain for many channels.

        """
        self.assertEqual(self.d[0].amplifier_gain(['SSC-H',
                                                   'FL2-H',
                                                   'FL3-H']),
                         [None, None, None])
        self.assertEqual(self.d[1].amplifier_gain(['FITC-A',
                                                   'PE-A',
                                                   'PE-Cy7-A']),
                         [None, None, None])
        self.assertEqual(self.d[2].amplifier_gain(['FSC',
                                                   'SSC',
                                                   'FL1']),
                         [None, None, None])
        self.assertEqual(self.d[3].amplifier_gain(['FSC PMT-A',
                                                   'FSC PMT-H',
                                                   'FSC PMT-W']),
                         [1.0, 1.0, 1.0])

    def test_slice_single_str(self):
        """
        Testing correct reporting of amplifier gain after slicing.

        """
        self.assertEqual(self.d[0][:, 'FSC-H'].amplifier_gain(), [None])
        self.assertEqual(self.d[1][:, 'FITC-A'].amplifier_gain(), [None])
        self.assertEqual(self.d[2][:, 'SSC'].amplifier_gain(), [None])
        self.assertEqual(self.d[3][:, 'GFP-A'].amplifier_gain(), [1.0])

    def test_slice_many_str(self):
        """
        Testing correct reporting of amplifier gain after slicing.

        """
        self.assertEqual(self.d[0][:, ['SSC-H',
                                       'FL2-H',
                                       'FL3-H']].amplifier_gain(),
                         [None, None, None])
        self.assertEqual(self.d[1][:, ['FITC-A',
                                       'PE-A',
                                       'PE-Cy7-A']].amplifier_gain(),
                         [None, None, None])
        self.assertEqual(self.d[2][:, ['FSC',
                                       'SSC',
                                       'FL1']].amplifier_gain(),
                         [None, None, None])
        self.assertEqual(self.d[3][:, ['FSC PMT-A',
                                       'FSC PMT-H',
                                       'Time']].amplifier_gain(),
                         [1.0, 1.0, 0.01])

class TestFCSAttributes(unittest.TestCase):
    def setUp(self):
        self.d = fc.io.FCSData(filenames[0])
        self.n_samples = self.d.shape[0]

#     def test_range(self):
#         """
#         Testing proper loading of range information.

#         """
#         self.assertEqual(self.d.channel_info[0]['range'], [0, 1023, 1024])
#         self.assertEqual(self.d.channel_info[1]['range'], [0, 1023, 1024])
#         self.assertEqual(self.d.channel_info[2]['range'], [0, 1023, 1024])
#         self.assertEqual(self.d.channel_info[3]['range'], [0, 1023, 1024])
#         self.assertEqual(self.d.channel_info[4]['range'], [0, 1023, 1024])

#     def test_bins(self):
#         """
#         Testing proper creation of bins.

#         """
#         # Bin values
#         np.testing.assert_array_equal(self.d.channel_info[0]['bin_vals'], 
#             np.arange(1024))
#         np.testing.assert_array_equal(self.d.channel_info[1]['bin_vals'], 
#             np.arange(1024))
#         np.testing.assert_array_equal(self.d.channel_info[2]['bin_vals'], 
#             np.arange(1024))
#         np.testing.assert_array_equal(self.d.channel_info[3]['bin_vals'], 
#             np.arange(1024))
#         np.testing.assert_array_equal(self.d.channel_info[4]['bin_vals'], 
#             np.arange(1024))
#         # Bin edges
#         np.testing.assert_array_equal(self.d.channel_info[0]['bin_edges'], 
#             np.arange(1025) - 0.5)
#         np.testing.assert_array_equal(self.d.channel_info[1]['bin_edges'], 
#             np.arange(1025) - 0.5)
#         np.testing.assert_array_equal(self.d.channel_info[2]['bin_edges'], 
#             np.arange(1025) - 0.5)
#         np.testing.assert_array_equal(self.d.channel_info[3]['bin_edges'], 
#             np.arange(1025) - 0.5)
#         np.testing.assert_array_equal(self.d.channel_info[4]['bin_edges'], 
#             np.arange(1025) - 0.5)

    def test_str(self):
        """
        Testing string representation.

        """
        self.assertEqual(str(self.d), 'Data001.fcs')

    def test_time_step(self):
        """
        Testing of the time step.

        """
        # Data.001 is a FCS2.0 file, use the timeticks parameter.
        # We have previously looked at self.d.text['TIMETICKS']) to determine
        # the correct output for this file.
        self.assertEqual(self.d.time_step, 0.2)

    def test_acquisition_start_time(self):
        """
        Testing of acquisition start time.

        """
        # We have previously looked at the $BTIM and #DATE attributes of
        # Data.001 to determine the correct value of acquisition_start_time.
        time_correct = datetime.datetime(2015, 5, 19, 16, 50, 29)
        self.assertEqual(self.d.acquisition_start_time, time_correct)

    def test_acquisition_end_time(self):
        """
        Testing of acquisition end time.

        """
        # We have previously looked at the $ETIM and #DATE attributes of
        # Data.001 to determine the correct value of acquisition_end_time.
        time_correct = datetime.datetime(2015, 5, 19, 16, 51, 46)
        self.assertEqual(self.d.acquisition_end_time, time_correct)

    def test_acquisition_time_event(self):
        """
        Testing acquisition time.

        """
        # Data.001 has the time channel, so the acquisition time should be
        # calculated using the event list.
        # We have previously looked at self.d[0, 'Time'] and self.d[-1, 'Time']
        # to determine the correct output for this file.
        self.assertEqual(self.d.acquisition_time, 74.8)

    def test_acquisition_time_btim_etim(self):
        """
        Testing acquisition time using the btim/etim method.

        """
        # Data.001 has the time channel, so we will remove it so that the
        # BTIM and ETIM keyword arguments are used.
        # We have previously looked at d.text['$BTIM'] and d.text['$ETIM'] to
        # determine the correct output for this file.
        d = self.d[:,['FSC-H', 'SSC-H', 'FL1-H', 'FL2-H', 'FL3-H']]
        self.assertEqual(d.acquisition_time, 77)

class TestFCSAttributes3(unittest.TestCase):
    def setUp(self):
        self.d = fc.io.FCSData(filenames[2])
        self.n_samples = self.d.shape[0]

#     def test_range(self):
#         """
#         Testing proper loading of range information.

#         """
#         self.assertEqual(self.d.channel_info[1]['range'], [0, 1023, 1024])
#         self.assertEqual(self.d.channel_info[2]['range'], [0, 1023, 1024])
#         self.assertEqual(self.d.channel_info[3]['range'], [0, 1023, 1024])
#         self.assertEqual(self.d.channel_info[4]['range'], [0, 1023, 1024])
#         self.assertEqual(self.d.channel_info[5]['range'], [0, 1023, 1024])

#     def test_bins(self):
#         """
#         Testing proper creation of bins.

#         """
#         # Bin values
#         np.testing.assert_array_equal(self.d.channel_info[1]['bin_vals'], 
#             np.arange(1024))
#         np.testing.assert_array_equal(self.d.channel_info[2]['bin_vals'], 
#             np.arange(1024))
#         np.testing.assert_array_equal(self.d.channel_info[3]['bin_vals'], 
#             np.arange(1024))
#         np.testing.assert_array_equal(self.d.channel_info[4]['bin_vals'], 
#             np.arange(1024))
#         np.testing.assert_array_equal(self.d.channel_info[5]['bin_vals'], 
#             np.arange(1024))
#         # Bin edges
#         np.testing.assert_array_equal(self.d.channel_info[1]['bin_edges'], 
#             np.arange(1025) - 0.5)
#         np.testing.assert_array_equal(self.d.channel_info[2]['bin_edges'], 
#             np.arange(1025) - 0.5)
#         np.testing.assert_array_equal(self.d.channel_info[3]['bin_edges'], 
#             np.arange(1025) - 0.5)
#         np.testing.assert_array_equal(self.d.channel_info[4]['bin_edges'], 
#             np.arange(1025) - 0.5)
#         np.testing.assert_array_equal(self.d.channel_info[5]['bin_edges'], 
#             np.arange(1025) - 0.5)

    def test_str(self):
        """
        Testing string representation.

        """
        self.assertEqual(str(self.d), 'Data003.fcs')

    def test_time_step(self):
        """
        Testing of the time step.

        """
        # Data.003 is a FCS3.0 file, use the $TIMESTEP parameter.
        # We have previously looked at self.d.text['$TIMESTEP']) to determine
        # the correct output for this file.
        self.assertEqual(self.d.time_step, 0.1)

    def test_acquisition_start_time(self):
        """
        Testing of acquisition start time.

        """
        # We have previously looked at the $BTIM and #DATE attributes of
        # Data.003 to determine the correct value of acquisition_start_time.
        time_correct = datetime.datetime(2015, 7, 27, 19, 57, 40)
        self.assertEqual(self.d.acquisition_start_time, time_correct)

    def test_acquisition_end_time(self):
        """
        Testing of acquisition end time.

        """
        # We have previously looked at the $ETIM and #DATE attributes of
        # Data.003 to determine the correct value of acquisition_end_time.
        time_correct = datetime.datetime(2015, 7, 27, 20, 00, 16)
        self.assertEqual(self.d.acquisition_end_time, time_correct)

    def test_acquisition_time_event(self):
        """
        Testing acquisition time.

        """
        # Data.003 has the time channel, so the acquisition time should be
        # calculated using the event list.
        # We have previously looked at self.d[0, 'TIME'] and self.d[-1, 'TIME']
        # to determine the correct output for this file.
        self.assertEqual(self.d.acquisition_time, 134.8)

    def test_acquisition_time_btim_etim(self):
        """
        Testing acquisition time using the btim/etim method.

        """
        # Data.003 has the time channel, so we will remove it so that the
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
        """
        Testing the 1D slicing with a scalar of a FCSData object.

        """
        ds = self.d[1]
        self.assertIsInstance(ds, fc.io.FCSData)
        self.assertEqual(ds.shape, (6,))
        self.assertEqual(ds.channels,
            ('FSC-H', 'SSC-H', 'FL1-H', 'FL2-H', 'FL3-H', 'Time'))

    def test_1d_slicing_with_list(self):
        """
        Testing the 1D slicing with a list of a FCSData object.

        """
        ds = self.d[range(10)]
        self.assertIsInstance(ds, fc.io.FCSData)
        self.assertEqual(ds.shape, (10,6))
        self.assertEqual(ds.channels,
            ('FSC-H', 'SSC-H', 'FL1-H', 'FL2-H', 'FL3-H', 'Time'))

    def test_slicing_channel_with_int(self):
        """
        Testing the channel slicing with an int of a FCSData object.

        """
        ds = self.d[:,2]
        self.assertIsInstance(ds, fc.io.FCSData)
        self.assertEqual(ds.shape, (self.n_samples,))
        self.assertEqual(ds.channels, ('FL1-H',))

    def test_slicing_channel_with_string(self):
        """
        Testing the channel slicing with a string of a FCSData object.

        """
        ds = self.d[:,'SSC-H']
        self.assertIsInstance(ds, fc.io.FCSData)
        self.assertEqual(ds.shape, (self.n_samples,))
        self.assertEqual(ds.channels, ('SSC-H', ))

    def test_slicing_channel_with_int_array(self):
        """
        Testing the channel slicing with an int array of a FCSData 
        object.
        """
        ds = self.d[:,[1,3]]
        self.assertIsInstance(ds, fc.io.FCSData)
        self.assertEqual(ds.shape, (self.n_samples,2))
        self.assertEqual(ds.channels, ('SSC-H', 'FL2-H'))

    def test_slicing_channel_with_string_array(self):
        """
        Testing the channel slicing with a string array of a FCSData 
        object.

        """
        ds = self.d[:,['FSC-H', 'FL3-H']]
        self.assertIsInstance(ds, fc.io.FCSData)
        self.assertEqual(ds.shape, (self.n_samples,2))
        self.assertEqual(ds.channels, ('FSC-H', 'FL3-H'))

    def test_slicing_sample(self):
        """
        Testing the sample slicing of a FCSData object.

        """
        ds = self.d[:1000]
        self.assertIsInstance(ds, fc.io.FCSData)
        self.assertEqual(ds.shape, (1000,6))
        self.assertEqual(ds.channels,
            ('FSC-H', 'SSC-H', 'FL1-H', 'FL2-H', 'FL3-H', 'Time'))

    def test_2d_slicing(self):
        """
        Testing 2D slicing of a FCSData object.

        """
        ds = self.d[:1000,['SSC-H', 'FL3-H']]
        self.assertIsInstance(ds, fc.io.FCSData)
        self.assertEqual(ds.shape, (1000,2))
        self.assertEqual(ds.channels, ('SSC-H', 'FL3-H'))

    def test_mask_slicing(self):
        """
        Testing mask slicing of a FCSData object.

        """
        m = self.d[:,1]>500
        ds = self.d[m,:]
        self.assertIsInstance(ds, fc.io.FCSData)
        self.assertEqual(ds.channels,
            ('FSC-H', 'SSC-H', 'FL1-H', 'FL2-H', 'FL3-H', 'Time'))

    def test_none_slicing_1(self):
        """
        Testing slicing with None on the first dimension of a FCSData 
        object.

        """
        ds = self.d[None,[0,2]]
        self.assertIsInstance(ds, np.ndarray)

    def test_none_slicing_2(self):
        """
        Testing slicing with None on the second dimension of a FCSData 
        object.

        """
        ds = self.d[:,None]
        self.assertIsInstance(ds, np.ndarray)

    def test_2d_slicing_assignment(self):
        """
        Test assignment to FCSData using slicing.

        """
        ds = self.d.copy()
        ds[:,[1,2]] = 5
        self.assertIsInstance(ds, fc.io.FCSData)
        self.assertEqual(ds.channels,
            ('FSC-H', 'SSC-H', 'FL1-H', 'FL2-H', 'FL3-H', 'Time'))
        np.testing.assert_array_equal(ds[:,0], self.d[:,0])
        np.testing.assert_array_equal(ds[:,1], 5)
        np.testing.assert_array_equal(ds[:,2], 5)
        np.testing.assert_array_equal(ds[:,3], self.d[:,3])
        np.testing.assert_array_equal(ds[:,4], self.d[:,4])

    def test_2d_slicing_assignment_string(self):
        """
        Test assignment to FCSData using slicing with channel names.

        """
        ds = self.d.copy()
        ds[:,['SSC-H', 'FL1-H']] = 5
        self.assertIsInstance(ds, fc.io.FCSData)
        self.assertEqual(ds.channels,
            ('FSC-H', 'SSC-H', 'FL1-H', 'FL2-H', 'FL3-H', 'Time'))
        np.testing.assert_array_equal(ds[:,0], self.d[:,0])
        np.testing.assert_array_equal(ds[:,1], 5)
        np.testing.assert_array_equal(ds[:,2], 5)
        np.testing.assert_array_equal(ds[:,3], self.d[:,3])
        np.testing.assert_array_equal(ds[:,4], self.d[:,4])

    def test_metadata_slicing(self):
        """
        Testing preservation of metadata upon slicing.

        """
        ds = self.d[:1000,['SSC-H', 'FL3-H']]
        self.assertIsInstance(ds.metadata, dict)
        self.assertEqual(ds.metadata, {'l2': 4, 'a': 'r'})


class TestFCSDataOperations(unittest.TestCase):
    def setUp(self):
        self.d = fc.io.FCSData(filenames[0], {'l2': 4, 'a': 'r'})
        self.n_samples = self.d.shape[0]

    def test_sum_integer(self):
        """
        Testing that the sum of a FCSData object returns a 
        FCSData object.

        """
        ds = self.d + 3
        self.assertIsInstance(ds, fc.io.FCSData)
        self.assertEqual(ds[254,3] - self.d[254,3], 3)
        
    def test_sqrt(self):
        """
        Testing that the square root of a FCSData object returns a 
        FCSData object.

        """
        ds = np.sqrt(self.d)
        self.assertIsInstance(ds, fc.io.FCSData)
        self.assertEqual(ds[254,3], np.sqrt(self.d[254,3]))

    def test_sum(self):
        """
        Testing that the sum of a FCSData object returns an scalar.

        """
        s = np.sum(self.d)
        self.assertIsInstance(s, np.uint)

    def test_mean(self):
        """
        Testing that the mean of a FCSData object returns an scalar.

        """
        m = np.mean(self.d)
        self.assertIsInstance(m, float)

    def test_std(self):
        """
        Testing that the std of a FCSData object returns an scalar.

        """
        s = np.std(self.d)
        self.assertIsInstance(s, float)

    def test_mean_axis(self):
        """
        Testing that the mean along the axis 0 of a FCSData object 
        returns a FCSData object.

        """
        m = np.mean(self.d, axis = 0)
        self.assertIsInstance(m, fc.io.FCSData)
        self.assertEqual(m.shape, (6,))

    def test_metadata_sqrt(self):
        """
        Testing preservation of metadata after taking the square root.

        """
        ds = np.sqrt(self.d)
        self.assertIsInstance(ds.metadata, dict)
        self.assertEqual(ds.metadata, {'l2': 4, 'a': 'r'})
        
if __name__ == '__main__':
    unittest.main()
