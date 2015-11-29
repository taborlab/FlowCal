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

class TestFCSAttributesAmplificationType(unittest.TestCase):
    """
    Test correct extraction, functioning, and slicing of amplification_type.

    We have previously looked at the contents of the $PnE attribute for
    the test files and identified the correct amplification_type:
        - Data001.fcs: [(0,0), (0,0), (4,1), (4,1), (4,1), (0,0)]
        - Data002.fcs: [(4,1), (4,1), (4,1), (4,1), (4,1), (4,1), (4,1),
                        (4,1), (0,0)]
        - Data003.fcs: [(0,0), (0,0), (0,0), (4,1), (4,1), (4,1), (0,0),
                        (0,0)]
        - Data004.fcs: [(0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0),
                        (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0)]

    """
    def setUp(self):
        self.d = [fc.io.FCSData(filenames[i]) for i in range(4)]

    def test_attribute(self):
        """
        Testing correct reporting of amplification type.

        """
        self.assertEqual(self.d[0].amplification_type(),
                         [(0,0), (0,0), (4,1), (4,1), (4,1), (0,0)])
        self.assertEqual(self.d[1].amplification_type(),
                         [(4,1), (4,1), (4,1), (4,1), (4,1), (4,1), (4,1),
                          (4,1), (0,0)])
        self.assertEqual(self.d[2].amplification_type(),
                         [(0,0), (0,0), (0,0), (4,1), (4,1), (4,1), (0,0),
                          (0,0)])
        self.assertEqual(self.d[3].amplification_type(),
                         [(0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0),
                          (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0)])

    def test_attribute_single(self):
        """
        Testing correct reporting of amp. type for a single channel.

        """
        self.assertEqual(self.d[0].amplification_type('FSC-H'), (0,0))
        self.assertEqual(self.d[1].amplification_type('FITC-A'), (4,1))
        self.assertEqual(self.d[2].amplification_type('SSC'), (0,0))
        self.assertEqual(self.d[3].amplification_type('GFP-A'), (0,0))

    def test_attribute_many(self):
        """
        Testing correct reporting of amp. type for many channels.

        """
        self.assertEqual(self.d[0].amplification_type(['SSC-H',
                                                       'FL2-H',
                                                       'FL3-H']),
                         [(0,0), (4,1), (4,1)])
        self.assertEqual(self.d[1].amplification_type(['FITC-A',
                                                       'PE-A',
                                                       'PE-Cy7-A']),
                         [(4,1), (4,1), (4,1)])
        self.assertEqual(self.d[2].amplification_type(['FSC',
                                                       'SSC',
                                                       'FL1']),
                         [(0,0), (0,0), (4,1)])
        self.assertEqual(self.d[3].amplification_type(['FSC PMT-A',
                                                       'FSC PMT-H',
                                                       'FSC PMT-W']),
                         [(0,0), (0,0), (0,0)])

    def test_slice_single_str(self):
        """
        Testing correct reporting of amp. type after slicing.

        """
        self.assertEqual(self.d[0][:, 'FSC-H'].amplification_type(), [(0,0)])
        self.assertEqual(self.d[1][:, 'FITC-A'].amplification_type(), [(4,1)])
        self.assertEqual(self.d[2][:, 'SSC'].amplification_type(), [(0,0)])
        self.assertEqual(self.d[3][:, 'GFP-A'].amplification_type(), [(0,0)])

    def test_slice_many_str(self):
        """
        Testing correct reporting of amp. type after slicing.

        """
        self.assertEqual(self.d[0][:, ['SSC-H',
                                       'FL2-H',
                                       'FL3-H']].amplification_type(),
                         [(0,0), (4,1), (4,1)])
        self.assertEqual(self.d[1][:, ['FITC-A',
                                       'PE-A',
                                       'PE-Cy7-A']].amplification_type(),
                         [(4,1), (4,1), (4,1)])
        self.assertEqual(self.d[2][:, ['FSC',
                                       'SSC',
                                       'FL1']].amplification_type(),
                         [(0,0), (0,0), (4,1)])
        self.assertEqual(self.d[3][:, ['FSC PMT-A',
                                       'FSC PMT-H',
                                       'FSC PMT-W']].amplification_type(),
                         [(0,0), (0,0), (0,0)])

class TestFCSAttributesDetectorVoltage(unittest.TestCase):
    """
    Test correct extraction, functioning, and slicing of detector_voltage.

    We have previously looked at the contents of the $PnV attribute for
    the test files ('BD$WORD{12 + n}' for Data001.fcs) and identified the
    correct detector voltage output as follows:
        - Data001.fcs: [1, 460, 400, 900, 999, 100]
        - Data002.fcs: [305, 319, 478, 470, 583, 854, 800, 620, None]
        - Data003.fcs: [None, 10.0, 460, 501, 501, 501, None, None]
        - Data004.fcs: [250, 250, 250, 340, 340, 340, 550, 550, 550, 650, 650,
                        575, 575, None]

    """
    def setUp(self):
        self.d = [fc.io.FCSData(filenames[i]) for i in range(4)]

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
    the test files and identified the correct amplifier gain output as
    follows:
        - Data001.fcs: [None, None, None, None, None, None]
        - Data002.fcs: [None, None, None, None, None, None, None, None,
                        None]
        - Data003.fcs: [None, None, None, None, None, None, None, None]
        - Data004.fcs: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 0.01]

    """
    def setUp(self):
        self.d = [fc.io.FCSData(filenames[i]) for i in range(4)]

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

class TestFCSAttributesDomain(unittest.TestCase):
    """
    Test correct extraction, functioning, and slicing of domain.

    We have previously looked at the contents of the $PnR attribute for
    the test files and identified the correct domain:
        - Data001.fcs: [np.arange(1024), np.arange(1024), np.arange(1024),
                        np.arange(1024), np.arange(1024), np.arange(1024)]
        - Data002.fcs: [np.arange(1024), np.arange(1024), np.arange(1024),
                        np.arange(1024), np.arange(1024), np.arange(1024),
                        np.arange(1024), np.arange(1024), np.arange(1024)]
        - Data003.fcs: [np.arange(262144), np.arange(1024), np.arange(1024),
                        np.arange(1024), np.arange(1024), np.arange(1024),
                        np.arange(1024), np.arange(1024)]
        - Data004.fcs: [np.arange(262144), np.arange(262144),
                        np.arange(262144), np.arange(262144),
                        np.arange(262144), np.arange(262144),
                        np.arange(262144), np.arange(262144),
                        np.arange(262144), np.arange(262144),
                        np.arange(262144), np.arange(262144),
                        np.arange(262144), np.arange(262144)]

    """
    def setUp(self):
        self.d = [fc.io.FCSData(filenames[i]) for i in range(4)]

    def assert_list_of_arrays_equal(self, l1, l2):
        self.assertEqual(len(l1), len(l2))
        for a1, a2 in zip(l1, l2):
            np.testing.assert_array_equal(a1, a2)

    def test_attribute(self):
        """
        Testing correct reporting of domain.

        """
        self.assert_list_of_arrays_equal(self.d[0].domain(),
                                         [np.arange(1024.), np.arange(1024.),
                                          np.arange(1024.), np.arange(1024.),
                                          np.arange(1024.), np.arange(1024.)])
        self.assert_list_of_arrays_equal(self.d[1].domain(),
                                         [np.arange(1024.), np.arange(1024.),
                                          np.arange(1024.), np.arange(1024.),
                                          np.arange(1024.), np.arange(1024.),
                                          np.arange(1024.), np.arange(1024.),
                                          np.arange(1024.)])
        self.assert_list_of_arrays_equal(self.d[2].domain(),
                                         [np.arange(262144.), np.arange(1024.),
                                          np.arange(1024.), np.arange(1024.),
                                          np.arange(1024.), np.arange(1024.),
                                          np.arange(1024.), np.arange(1024.)])
        self.assert_list_of_arrays_equal(
            self.d[3].domain(),
            [np.arange(262144.), np.arange(262144.), np.arange(262144.),
             np.arange(262144.), np.arange(262144.), np.arange(262144.), 
             np.arange(262144.), np.arange(262144.), np.arange(262144.),
             np.arange(262144.), np.arange(262144.), np.arange(262144.), 
             np.arange(262144.), np.arange(262144.)])

    def test_attribute_single(self):
        """
        Testing correct reporting of domain for a single channel.

        """
        np.testing.assert_array_equal(self.d[0].domain('FSC-H'),
                                      np.arange(1024))
        np.testing.assert_array_equal(self.d[1].domain('FITC-A'),
                                      np.arange(1024))
        np.testing.assert_array_equal(self.d[2].domain('SSC'),
                                      np.arange(1024))
        np.testing.assert_array_equal(self.d[3].domain('GFP-A'),
                                      np.arange(262144))

    def test_attribute_many(self):
        """
        Testing correct reporting of domain for many channels.

        """
        self.assert_list_of_arrays_equal(self.d[0].domain(['SSC-H',
                                                           'FL2-H',
                                                           'FL3-H']),
                                         [np.arange(1024),
                                          np.arange(1024),
                                          np.arange(1024)])
        self.assert_list_of_arrays_equal(self.d[1].domain(['FITC-A',
                                                           'PE-A',
                                                           'PE-Cy7-A']),
                                         [np.arange(1024),
                                          np.arange(1024),
                                          np.arange(1024)])
        self.assert_list_of_arrays_equal(self.d[2].domain(['FSC',
                                                           'SSC',
                                                           'TIME']),
                                         [np.arange(1024),
                                          np.arange(1024),
                                          np.arange(262144)])
        self.assert_list_of_arrays_equal(self.d[3].domain(['FSC PMT-A',
                                                           'FSC PMT-H',
                                                           'FSC PMT-W']),
                                         [np.arange(262144),
                                          np.arange(262144),
                                          np.arange(262144)])

class TestFCSAttributesHistBinEdges(unittest.TestCase):
    """
    Test correct extraction, functioning, and slicing of bin edges.

    We have previously looked at the contents of the $PnR attribute for
    the test files and identified the correct histogram bin edges:
        - Data001.fcs: [np.arange(1025) - 0.5, np.arange(1025) - 0.5,
                        np.arange(1025) - 0.5, np.arange(1025) - 0.5,
                        np.arange(1025) - 0.5, np.arange(1025) - 0.5]
        - Data002.fcs: [np.arange(1025) - 0.5, np.arange(1025) - 0.5,
                        np.arange(1025) - 0.5, np.arange(1025) - 0.5,
                        np.arange(1025) - 0.5, np.arange(1025) - 0.5,
                        np.arange(1025) - 0.5, np.arange(1025) - 0.5,
                        np.arange(1025) - 0.5]
        - Data003.fcs: [np.arange(262145) - 0.5, np.arange(1025) - 0.5,
                        np.arange(1025) - 0.5, np.arange(1025) - 0.5,
                        np.arange(1025) - 0.5, np.arange(1025) - 0.5,
                        np.arange(1025) - 0.5, np.arange(1025) - 0.5]
        - Data004.fcs: [np.arange(262145) - 0.5, np.arange(262145) - 0.5,
                        np.arange(262145) - 0.5, np.arange(262145) - 0.5,
                        np.arange(262145) - 0.5, np.arange(262145) - 0.5,
                        np.arange(262145) - 0.5, np.arange(262145) - 0.5,
                        np.arange(262145) - 0.5, np.arange(262145) - 0.5,
                        np.arange(262145) - 0.5, np.arange(262145) - 0.5,
                        np.arange(262145) - 0.5, np.arange(262145) - 0.5]

    """
    def setUp(self):
        self.d = [fc.io.FCSData(filenames[i]) for i in range(4)]

    def assert_list_of_arrays_equal(self, l1, l2):
        self.assertEqual(len(l1), len(l2))
        for a1, a2 in zip(l1, l2):
            np.testing.assert_array_equal(a1, a2)

    def test_attribute(self):
        """
        Testing correct reporting of hist_bin_edges.

        """
        self.assert_list_of_arrays_equal(
            self.d[0].hist_bin_edges(),
            [np.arange(1025) - 0.5, np.arange(1025) - 0.5,
             np.arange(1025) - 0.5, np.arange(1025) - 0.5,
             np.arange(1025) - 0.5, np.arange(1025) - 0.5])
        self.assert_list_of_arrays_equal(
            self.d[1].hist_bin_edges(),
            [np.arange(1025) - 0.5, np.arange(1025) - 0.5,
             np.arange(1025) - 0.5, np.arange(1025) - 0.5,
             np.arange(1025) - 0.5, np.arange(1025) - 0.5,
             np.arange(1025) - 0.5, np.arange(1025) - 0.5,
             np.arange(1025) - 0.5])
        self.assert_list_of_arrays_equal(
            self.d[2].hist_bin_edges(),
            [np.arange(262145) - 0.5, np.arange(1025) - 0.5,
             np.arange(1025) - 0.5, np.arange(1025) - 0.5,
             np.arange(1025) - 0.5, np.arange(1025) - 0.5,
             np.arange(1025) - 0.5, np.arange(1025) - 0.5])
        self.assert_list_of_arrays_equal(
            self.d[3].hist_bin_edges(),
            [np.arange(262145) - 0.5, np.arange(262145) - 0.5,
             np.arange(262145) - 0.5, np.arange(262145) - 0.5,
             np.arange(262145) - 0.5, np.arange(262145) - 0.5,
             np.arange(262145) - 0.5, np.arange(262145) - 0.5,
             np.arange(262145) - 0.5, np.arange(262145) - 0.5,
             np.arange(262145) - 0.5, np.arange(262145) - 0.5,
             np.arange(262145) - 0.5, np.arange(262145) - 0.5])

    def test_attribute_single(self):
        """
        Testing correct reporting of hist_bin_edges for a single channel.

        """
        np.testing.assert_array_equal(self.d[0].hist_bin_edges('FSC-H'),
                                      np.arange(1025) - 0.5)
        np.testing.assert_array_equal(self.d[1].hist_bin_edges('FITC-A'),
                                      np.arange(1025) - 0.5)
        np.testing.assert_array_equal(self.d[2].hist_bin_edges('SSC'),
                                      np.arange(1025) - 0.5)
        np.testing.assert_array_equal(self.d[3].hist_bin_edges('GFP-A'),
                                      np.arange(262145) - 0.5)

    def test_attribute_many(self):
        """
        Testing correct reporting of hist_bin_edges for many channels.

        """
        self.assert_list_of_arrays_equal(
            self.d[0].hist_bin_edges(['SSC-H',
                                      'FL2-H',
                                      'FL3-H']),
            [np.arange(1025) - 0.5,
             np.arange(1025) - 0.5,
             np.arange(1025) - 0.5])
        self.assert_list_of_arrays_equal(
            self.d[1].hist_bin_edges(['FITC-A',
                                      'PE-A',
                                      'PE-Cy7-A']),
            [np.arange(1025) - 0.5,
             np.arange(1025) - 0.5,
             np.arange(1025) - 0.5])
        self.assert_list_of_arrays_equal(
            self.d[2].hist_bin_edges(['FSC',
                                      'SSC',
                                      'TIME']),
            [np.arange(1025) - 0.5,
             np.arange(1025) - 0.5,
             np.arange(262145) - 0.5])
        self.assert_list_of_arrays_equal(
            self.d[3].hist_bin_edges(['FSC PMT-A',
                                      'FSC PMT-H',
                                      'FSC PMT-W']),
            [np.arange(262145) - 0.5,
             np.arange(262145) - 0.5,
             np.arange(262145) - 0.5])

    def test_slice_single_str(self):
        """
        Testing correct reporting of hist_bin_edges after slicing.

        """
        self.assert_list_of_arrays_equal(
            self.d[0][:, 'FSC-H'].hist_bin_edges(),
            [np.arange(1025) - 0.5])
        self.assert_list_of_arrays_equal(
            self.d[1][:, 'FITC-A'].hist_bin_edges(),
            [np.arange(1025) - 0.5])
        self.assert_list_of_arrays_equal(
            self.d[2][:, 'SSC'].hist_bin_edges(),
            [np.arange(1025) - 0.5])
        self.assert_list_of_arrays_equal(
            self.d[3][:, 'GFP-A'].hist_bin_edges(),
            [np.arange(262145) - 0.5])

    def test_slice_many_str(self):
        """
        Testing correct reporting of hist_bin_edges after slicing.

        """
        self.assert_list_of_arrays_equal(
            self.d[0][:, ['SSC-H',
                          'FL2-H',
                          'FL3-H']].hist_bin_edges(),
            [np.arange(1025) - 0.5,
             np.arange(1025) - 0.5,
             np.arange(1025) - 0.5])
        self.assert_list_of_arrays_equal(
            self.d[1][:, ['FITC-A',
                          'PE-A',
                          'PE-Cy7-A']].hist_bin_edges(),
            [np.arange(1025) - 0.5,
             np.arange(1025) - 0.5,
             np.arange(1025) - 0.5])
        self.assert_list_of_arrays_equal(
            self.d[2][:, ['FSC',
                          'SSC',
                          'TIME']].hist_bin_edges(),
            [np.arange(1025) - 0.5,
             np.arange(1025) - 0.5,
             np.arange(262145) - 0.5])
        self.assert_list_of_arrays_equal(
            self.d[3][:,['FSC PMT-A',
                         'FSC PMT-H',
                         'FSC PMT-W']].hist_bin_edges(),
            [np.arange(262145) - 0.5,
             np.arange(262145) - 0.5,
             np.arange(262145) - 0.5])

class TestFCSAttributes(unittest.TestCase):
    def setUp(self):
        self.d = fc.io.FCSData(filenames[0])
        self.n_samples = self.d.shape[0]

    def test_str(self):
        """
        Testing string representation.

        """
        self.assertEqual(str(self.d), 'Data001.fcs')

    def test_time_step(self):
        """
        Testing of the time step.

        """
        # Data001.fcs is a FCS2.0 file, use the timeticks parameter.
        # We have previously looked at self.d.text['TIMETICKS']) to determine
        # the correct output for this file.
        self.assertEqual(self.d.time_step, 0.2)

    def test_acquisition_start_time(self):
        """
        Testing of acquisition start time.

        """
        # We have previously looked at the $BTIM and #DATE attributes of
        # Data001.fcs to determine the correct value of acquisition_start_time.
        time_correct = datetime.datetime(2015, 5, 19, 16, 50, 29)
        self.assertEqual(self.d.acquisition_start_time, time_correct)

    def test_acquisition_end_time(self):
        """
        Testing of acquisition end time.

        """
        # We have previously looked at the $ETIM and #DATE attributes of
        # Data001.fcs to determine the correct value of acquisition_end_time.
        time_correct = datetime.datetime(2015, 5, 19, 16, 51, 46)
        self.assertEqual(self.d.acquisition_end_time, time_correct)

    def test_acquisition_time_event(self):
        """
        Testing acquisition time.

        """
        # Data001.fcs has the time channel, so the acquisition time should be
        # calculated using the event list.
        # We have previously looked at self.d[0, 'Time'] and self.d[-1, 'Time']
        # to determine the correct output for this file.
        self.assertEqual(self.d.acquisition_time, 74.8)

    def test_acquisition_time_btim_etim(self):
        """
        Testing acquisition time using the btim/etim method.

        """
        # Data001.fcs has the time channel, so we will remove it so that the
        # BTIM and ETIM keyword arguments are used.
        # We have previously looked at d.text['$BTIM'] and d.text['$ETIM'] to
        # determine the correct output for this file.
        d = self.d[:,['FSC-H', 'SSC-H', 'FL1-H', 'FL2-H', 'FL3-H']]
        self.assertEqual(d.acquisition_time, 77)

class TestFCSAttributes3(unittest.TestCase):
    def setUp(self):
        self.d = fc.io.FCSData(filenames[2])
        self.n_samples = self.d.shape[0]

    def test_str(self):
        """
        Testing string representation.

        """
        self.assertEqual(str(self.d), 'Data003.fcs')

    def test_time_step(self):
        """
        Testing of the time step.

        """
        # Data003.fcs is a FCS3.0 file, use the $TIMESTEP parameter.
        # We have previously looked at self.d.text['$TIMESTEP']) to determine
        # the correct output for this file.
        self.assertEqual(self.d.time_step, 0.1)

    def test_acquisition_start_time(self):
        """
        Testing of acquisition start time.

        """
        # We have previously looked at the $BTIM and #DATE attributes of
        # Data003.fcs to determine the correct value of acquisition_start_time.
        time_correct = datetime.datetime(2015, 7, 27, 19, 57, 40)
        self.assertEqual(self.d.acquisition_start_time, time_correct)

    def test_acquisition_end_time(self):
        """
        Testing of acquisition end time.

        """
        # We have previously looked at the $ETIM and #DATE attributes of
        # Data003.fcs to determine the correct value of acquisition_end_time.
        time_correct = datetime.datetime(2015, 7, 27, 20, 00, 16)
        self.assertEqual(self.d.acquisition_end_time, time_correct)

    def test_acquisition_time_event(self):
        """
        Testing acquisition time.

        """
        # Data003.fcs has the time channel, so the acquisition time should be
        # calculated using the event list.
        # We have previously looked at self.d[0, 'TIME'] and self.d[-1, 'TIME']
        # to determine the correct output for this file.
        self.assertEqual(self.d.acquisition_time, 134.8)

    def test_acquisition_time_btim_etim(self):
        """
        Testing acquisition time using the btim/etim method.

        """
        # Data003.fcs has the time channel, so we will remove it so that the
        # BTIM and ETIM keyword arguments are used.
        # We have previously looked at d.text['$BTIM'] and d.text['$ETIM'] to
        # determine the correct output for this file.
        d = self.d[:,['FSC', 'SSC', 'FL1', 'FL2', 'FL3']]
        self.assertEqual(d.acquisition_time, 156)

class TestFCSDataSlicing(unittest.TestCase):
    def setUp(self):
        self.d = fc.io.FCSData(filenames[0])
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


class TestFCSDataOperations(unittest.TestCase):
    def setUp(self):
        self.d = fc.io.FCSData(filenames[0])
        self.n_samples = self.d.shape[0]

    def test_sum_integer(self):
        """
        Testing that scalar + FCSData is consistent with scalar + ndarray

        """
        ds = self.d + 3
        ds_array = self.d.view(np.ndarray) + 3
        self.assertIsInstance(ds, fc.io.FCSData)
        np.testing.assert_array_equal(ds, ds_array)
        
    def test_sqrt(self):
        """
        Testing that the sqrt(FCSData) is consistent with sqrt(ndarray)

        """
        ds = np.sqrt(self.d)
        ds_array = np.sqrt(self.d.view(np.ndarray))
        self.assertIsInstance(ds, fc.io.FCSData)
        np.testing.assert_array_equal(ds, ds_array)

    def test_sum(self):
        """
        Testing that the sum(FCSData) is consistent with sum(ndarray)

        """
        s = np.sum(self.d)
        s_array = np.sum(self.d.view(np.ndarray))
        self.assertEqual(s, s_array)
        self.assertEqual(type(s), type(s_array))

    def test_mean(self):
        """
        Testing that the mean(FCSData) is consistent with mean(ndarray)

        """
        m = np.mean(self.d)
        m_array = np.mean(self.d.view(np.ndarray))
        self.assertEqual(m, m_array)
        self.assertEqual(type(m), type(m_array))

    def test_median(self):
        """
        Testing that the median(FCSData) is consistent with median(ndarray)

        """
        m = np.median(self.d)
        m_array = np.median(self.d.view(np.ndarray))
        self.assertEqual(m, m_array)
        self.assertEqual(type(m), type(m_array))

    def test_std(self):
        """
        Testing that the std(FCSData) is consistent with std(ndarray)

        """
        s = np.std(self.d)
        s_array = np.std(self.d.view(np.ndarray))
        self.assertEqual(s, s_array)
        self.assertEqual(type(s), type(s_array))

    def test_mean_axis(self):
        """
        Testing that the mean along the axis 0 of a FCSData object 
        returns a FCSData object.

        """
        m = np.mean(self.d, axis = 0)
        m_array = np.mean(self.d.view(np.ndarray), axis = 0)
        self.assertIsInstance(m, fc.io.FCSData)
        self.assertEqual(m.shape, m_array.shape)
        np.testing.assert_array_equal(m, m_array)
        
if __name__ == '__main__':
    unittest.main()
