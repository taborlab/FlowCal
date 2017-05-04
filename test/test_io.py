"""
Unit tests for the `io` module.

"""

import datetime
import unittest
import StringIO

import numpy as np

import FlowCal.io

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
        d = FlowCal.io.FCSData(filenames[0])
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
        d = FlowCal.io.FCSData(filenames[1])
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
        d = FlowCal.io.FCSData(filenames[2])
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
        d = FlowCal.io.FCSData(filenames[3])
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

class TestReadTextSegment(unittest.TestCase):
    """
    Test that TEXT segments are parsed correctly.
    
    """

    ###
    # Primary TEXT Segment Tests
    ###
    
    def test_primary_one_key_value(self):
        """
        Test that typical primary TEXT segment is read correctly.
        
        """
        raw_text_segment = '/k1/v1/'
        delim            = '/'
        text_dict        = {'k1':'v1'}
        buf              = StringIO.StringIO(raw_text_segment)
        self.assertEqual(
            FlowCal.io.read_fcs_text_segment(
                buf=buf,
                begin=0,
                end=len(raw_text_segment)-1),
            (text_dict, delim))

    def test_primary_three_key_value(self):
        """
        Test that typical primary TEXT segment is read correctly.

        """
        raw_text_segment = '/k1/v1/k2/v2/k3/v3/'
        delim            = '/'
        text_dict        = {'k1':'v1','k2':'v2','k3':'v3'}
        buf              = StringIO.StringIO(raw_text_segment)
        self.assertEqual(
            FlowCal.io.read_fcs_text_segment(
                buf=buf,
                begin=0,
                end=len(raw_text_segment)-1),
            (text_dict, delim))

    def test_primary_extra_trailing_chars(self):
        """
        Test that extra trailing characters still parse correctly.

        Test that primary TEXT segment still parses correctly even if there are
        trailing characters after the last instance of the delimiter.

        """
        raw_text_segment = '/k1/v1/     '
        delim            = '/'
        text_dict        = {'k1':'v1'}
        buf              = StringIO.StringIO(raw_text_segment)
        self.assertEqual(
            FlowCal.io.read_fcs_text_segment(
                buf=buf,
                begin=0,
                end=len(raw_text_segment)-1),
            (text_dict, delim))

    def test_primary_delim_fail_1(self):
        """
        Test that primary TEXT segment delimiter mismatch fails.

        Test that a specified delimiter inconsistent with the first character
        of a primary TEXT segment fails to parse.
        
        """
        raw_text_segment = '/k1/v1/'
        delim            = '|'
        buf              = StringIO.StringIO(raw_text_segment)
        self.assertRaises(
            ValueError,
            FlowCal.io.read_fcs_text_segment,
            buf=buf, begin=0, end=len(raw_text_segment)-1, delim=delim)

    def test_primary_delim_fail_2(self):
        """
        Test that primary segment fails if delim is not first character.

        This test fails because 'k' is deduced to be the delimiter, but upon
        splitting on 'k', there are not an even number of pairs remaining.
        
        """
        raw_text_segment = 'k1/v1/'
        delim            = '/'
        buf              = StringIO.StringIO(raw_text_segment)
        self.assertRaises(
            ValueError,
            FlowCal.io.read_fcs_text_segment,
            buf=buf, begin=0, end=len(raw_text_segment)-1)

    def test_primary_delim_fail_3(self):
        """
        Test that primary segment fails if delim is not first character.

        This test fails because, upon specifying the correct delimiter ('/'),
        the delimiter does not match the first character of the primary TEXT
        segment. Note, this same test should succeed for a supplemental TEXT
        segment.

        """
        raw_text_segment = 'k1/v1/'
        delim            = '/'
        buf              = StringIO.StringIO(raw_text_segment)
        self.assertRaises(
            ValueError,
            FlowCal.io.read_fcs_text_segment,
            buf=buf, begin=0, end=len(raw_text_segment)-1, delim=delim)

    def test_primary_bad_segment_1(self):
        """
        Test that read_fcs_text_segment() fails to parse minimal TEXT segment.

        Test that read_fcs_text_segment() fails to parse a minimal TEXT
        segment because it's either classified as having empty keywords and
        values (which is prohibited by the standards), or it is perceived as
        having keywords or values that start with the delimiter (also
        prohibited by the standards).
        """
        raw_text_segment = '///'
        delim            = '/'
        buf              = StringIO.StringIO(raw_text_segment)
        self.assertRaises(
            ValueError,
            FlowCal.io.read_fcs_text_segment,
            buf=buf, begin=0, end=len(raw_text_segment)-1, delim=delim)

    def test_primary_bad_segment_2(self):
        """
        Test that read_fcs_text_segment() fails to parse minimal TEXT segment.

        This test fails to parse because there is no keyword, only a value
        flanked by delimiters.
        
        """
        raw_text_segment = '/str/ing'
        delim            = '/'
        buf              = StringIO.StringIO(raw_text_segment)
        self.assertRaises(
            ValueError,
            FlowCal.io.read_fcs_text_segment,
            buf=buf, begin=0, end=len(raw_text_segment)-1, delim=delim)

    def test_primary_delim_in_keyword(self):
        """
        Test that delimiter in keyword still parses correctly.
        
        """
        raw_text_segment = '/k1/v1/key//2/value2/k3/v3/'
        delim            = '/'
        text_dict        = {'k1':'v1','key/2':'value2','k3':'v3'}
        buf              = StringIO.StringIO(raw_text_segment)
        self.assertEqual(
            FlowCal.io.read_fcs_text_segment(
                buf=buf,
                begin=0,
                end=len(raw_text_segment)-1),
            (text_dict, delim))

    def test_primary_delim_in_value(self):
        """
        Test that delimiter in keyword value still parses correctly.
        
        """
        raw_text_segment = '/k1/v1/key2/value//2/k3/v3/'
        delim            = '/'
        text_dict        = {'k1':'v1','key2':'value/2','k3':'v3'}
        buf              = StringIO.StringIO(raw_text_segment)
        self.assertEqual(
            FlowCal.io.read_fcs_text_segment(
                buf=buf,
                begin=0,
                end=len(raw_text_segment)-1),
            (text_dict, delim))

    def test_primary_delim_in_keyword_fail(self):
        """
        Test that delimiter at start of keyword fails.
        
        """
        raw_text_segment = '/k1/v1///key2/value2/k3/v3/'
        delim            = '/'
        buf              = StringIO.StringIO(raw_text_segment)
        self.assertRaises(
            ValueError,
            FlowCal.io.read_fcs_text_segment,
            buf=buf, begin=0, end=len(raw_text_segment)-1, delim=delim)

    def test_primary_delim_in_value_fail(self):
        """
        Test that delimiter at start of keyword value fails.
        
        """
        raw_text_segment = '/k1/v1/key2///value2/k3/v3/'
        delim            = '/'
        buf              = StringIO.StringIO(raw_text_segment)
        self.assertRaises(
            ValueError,
            FlowCal.io.read_fcs_text_segment,
            buf=buf, begin=0, end=len(raw_text_segment)-1, delim=delim)

    def test_primary_multi_delim_in_keyword(self):
        """
        Test that delimiter in keyword still parses correctly.
        
        """
        raw_text_segment = '/k1/v1/key//////2/value2/k3/v3/'
        delim            = '/'
        text_dict        = {'k1':'v1','key///2':'value2','k3':'v3'}
        buf              = StringIO.StringIO(raw_text_segment)
        self.assertEqual(
            FlowCal.io.read_fcs_text_segment(
                buf=buf,
                begin=0,
                end=len(raw_text_segment)-1),
            (text_dict, delim))
    
    ###
    # Supplemental TEXT Segment Tests
    ###

    #TODO

class TestFCSParseTimeString(unittest.TestCase):
    def test_parse_none(self):
        """
        Test that _parse_time_string() returns None when the input is None.

        """
        t = FlowCal.io.FCSData._parse_time_string(None)
        self.assertEqual(t, None)

    def test_parse_fcs2(self):
        """
        Test that _parse_time_string() interprets the FCS2.0 time format.

        """
        t = FlowCal.io.FCSData._parse_time_string("20:15:43")
        self.assertEqual(t, datetime.time(20, 15, 43))

    def test_parse_fcs3(self):
        """
        Test that _parse_time_string() interprets the FCS3.0 time format.

        """
        t = FlowCal.io.FCSData._parse_time_string("20:15:43:20")
        self.assertEqual(t, datetime.time(20, 15, 43, 333333))

    def test_parse_fcs3_1(self):
        """
        Test that _parse_time_string() interprets the FCS3.1 time format.

        """
        t = FlowCal.io.FCSData._parse_time_string("20:15:43.27")
        self.assertEqual(t, datetime.time(20, 15, 43, 270000))

    def test_parse_undefined(self):
        """
        Test that _parse_time_string() returns None for undefined inputs.

        """
        t = FlowCal.io.FCSData._parse_time_string("i'm undefined")
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
        self.d = [FlowCal.io.FCSData(filenames[i]) for i in range(4)]

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
        self.d = [FlowCal.io.FCSData(filenames[i]) for i in range(4)]

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
        - Data003.fcs: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        - Data004.fcs: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 0.01]

    """
    def setUp(self):
        self.d = [FlowCal.io.FCSData(filenames[i]) for i in range(4)]

    def test_attribute(self):
        """
        Testing correct reporting of amplifier gain.

        """
        self.assertEqual(self.d[0].amplifier_gain(),
                         [None, None, None, None, None, None])
        self.assertEqual(self.d[1].amplifier_gain(),
                         [None, None, None, None, None, None, None, None, None])
        self.assertEqual(self.d[2].amplifier_gain(),
                         [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.assertEqual(self.d[3].amplifier_gain(),
                         [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                          1.0, 1.0, 0.01])

    def test_attribute_single(self):
        """
        Testing correct reporting of amplifier gain for a single channel.

        """
        self.assertEqual(self.d[0].amplifier_gain('FSC-H'), None)
        self.assertEqual(self.d[1].amplifier_gain('FITC-A'), None)
        self.assertEqual(self.d[2].amplifier_gain('SSC'), 1.0)
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
                         [1.0, 1.0, 1.0])
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
        self.assertEqual(self.d[2][:, 'SSC'].amplifier_gain(), [1.0])
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
                         [1.0, 1.0, 1.0])
        self.assertEqual(self.d[3][:, ['FSC PMT-A',
                                       'FSC PMT-H',
                                       'Time']].amplifier_gain(),
                         [1.0, 1.0, 0.01])

class TestFCSAttributesRange(unittest.TestCase):
    """
    Test correct extraction, functioning, and slicing of range.

    We have previously looked at the contents of the $PnR attribute for
    the test files and identified the correct range:
        - Data001.fcs: [[0, 1023], [0, 1023], [0, 1023], [0, 1023],
                        [0, 1023], [0, 1023]]
        - Data002.fcs: [[0, 1023], [0, 1023], [0, 1023], [0, 1023],
                        [0, 1023], [0, 1023], [0, 1023], [0, 1023],
                        [0, 1023]]
        - Data003.fcs: [[0, 262143], [0, 1023], [0, 1023], [0, 1023],
                        [0, 1023], [0, 1023], [0, 1023], [0, 1023]]
        - Data004.fcs: [[0, 262143], [0, 262143], [0, 262143], [0, 262143],
                        [0, 262143], [0, 262143], [0, 262143], [0, 262143],
                        [0, 262143], [0, 262143], [0, 262143], [0, 262143],
                        [0, 262143], [0, 262143]]

    """
    def setUp(self):
        self.d = [FlowCal.io.FCSData(filenames[i]) for i in range(4)]

    def assert_list_of_arrays_equal(self, l1, l2):
        self.assertEqual(len(l1), len(l2))
        for a1, a2 in zip(l1, l2):
            np.testing.assert_array_equal(a1, a2)

    def test_attribute(self):
        """
        Testing correct reporting of range.

        """
        self.assert_list_of_arrays_equal(self.d[0].range(),
                                         [[0, 1023], [0, 1023],
                                          [0, 1023], [0, 1023],
                                          [0, 1023], [0, 1023]])
        self.assert_list_of_arrays_equal(self.d[1].range(),
                                         [[0, 1023], [0, 1023], [0, 1023],
                                          [0, 1023], [0, 1023], [0, 1023],
                                          [0, 1023], [0, 1023], [0, 1023]])
        self.assert_list_of_arrays_equal(self.d[2].range(),
                                         [[0, 262143], [0, 1023], [0, 1023],
                                          [0, 1023],   [0, 1023], [0, 1023],
                                          [0, 1023],   [0, 1023]])
        self.assert_list_of_arrays_equal(self.d[3].range(),
            [[0, 262143], [0, 262143], [0, 262143], [0, 262143], [0, 262143],
             [0, 262143], [0, 262143], [0, 262143], [0, 262143], [0, 262143],
             [0, 262143], [0, 262143], [0, 262143], [0, 262143]])

    def test_attribute_single(self):
        """
        Testing correct reporting of range for a single channel.

        """
        np.testing.assert_array_equal(self.d[0].range('FSC-H'), [0, 1023])
        np.testing.assert_array_equal(self.d[1].range('FITC-A'), [0, 1023])
        np.testing.assert_array_equal(self.d[2].range('SSC'), [0, 1023])
        np.testing.assert_array_equal(self.d[3].range('GFP-A'), [0, 262143])

    def test_attribute_many(self):
        """
        Testing correct reporting of range for many channels.

        """
        self.assert_list_of_arrays_equal(self.d[0].range(['SSC-H',
                                                          'FL2-H',
                                                          'FL3-H']),
                                         [[0, 1023],
                                          [0, 1023],
                                          [0, 1023]])
        self.assert_list_of_arrays_equal(self.d[1].range(['FITC-A',
                                                          'PE-A',
                                                          'PE-Cy7-A']),
                                         [[0, 1023],
                                          [0, 1023],
                                          [0, 1023]])
        self.assert_list_of_arrays_equal(self.d[2].range(['FSC',
                                                          'SSC',
                                                          'TIME']),
                                         [[0, 1023],
                                          [0, 1023],
                                          [0, 262143]])
        self.assert_list_of_arrays_equal(self.d[3].range(['FSC PMT-A',
                                                          'FSC PMT-H',
                                                          'FSC PMT-W']),
                                         [[0, 262143],
                                          [0, 262143],
                                          [0, 262143]])

class TestFCSAttributesResolution(unittest.TestCase):
    """
    Test correct extraction, functioning, and slicing of channel resolution.

    We have previously looked at the contents of the $PnR attribute for
    the test files and identified the correct histogram bin edges:
        - Data001.fcs: [1024, 1024, 1024, 1024, 1024, 1024]
        - Data002.fcs: [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024]
        - Data003.fcs: [262144, 1024, 1024, 1024, 1024, 1024, 1024, 1024]
        - Data004.fcs: [262144, 262144, 262144, 262144, 262144, 262144, 262144,
                        262144, 262144, 262144, 262144, 262144, 262144, 262144]

    """
    def setUp(self):
        self.d = [FlowCal.io.FCSData(filenames[i]) for i in range(4)]

    def test_attribute(self):
        """
        Testing correct reporting of resolution.

        """
        self.assertListEqual(
            self.d[0].resolution(),
            [1024, 1024, 1024, 1024, 1024, 1024])
        self.assertListEqual(
            self.d[1].resolution(),
            [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024])
        self.assertListEqual(
            self.d[2].resolution(),
            [262144, 1024, 1024, 1024, 1024, 1024, 1024, 1024])
        self.assertListEqual(
            self.d[3].resolution(),
            [262144, 262144, 262144, 262144, 262144, 262144, 262144,
             262144, 262144, 262144, 262144, 262144, 262144, 262144])

    def test_attribute_single(self):
        """
        Testing correct reporting of resolution for a single channel.

        """
        self.assertEqual(self.d[0].resolution('FSC-H'), 1024)
        self.assertEqual(self.d[1].resolution('FITC-A'), 1024)
        self.assertEqual(self.d[2].resolution('SSC'), 1024)
        self.assertEqual(self.d[3].resolution('GFP-A'), 262144)

    def test_attribute_many(self):
        """
        Testing correct reporting of resolution for many channels.

        """
        self.assertListEqual(
            self.d[0].resolution(['SSC-H', 'FL2-H', 'FL3-H']),
            [1024, 1024, 1024])
        self.assertListEqual(
            self.d[1].resolution(['FITC-A', 'PE-A', 'PE-Cy7-A']),
            [1024, 1024, 1024])
        self.assertListEqual(
            self.d[2].resolution(['FSC', 'SSC', 'TIME']),
            [1024, 1024, 262144])
        self.assertListEqual(
            self.d[3].resolution(['FSC PMT-A', 'FSC PMT-H', 'FSC PMT-W']),
            [262144, 262144, 262144])

    def test_slice_single_str(self):
        """
        Testing correct reporting of resolution after slicing.

        """
        self.assertListEqual(
            self.d[0][:, 'FSC-H'].resolution(),
            [1024,])
        self.assertListEqual(
            self.d[1][:, 'FITC-A'].resolution(),
            [1024,])
        self.assertListEqual(
            self.d[2][:, 'SSC'].resolution(),
            [1024,])
        self.assertListEqual(
            self.d[3][:, 'GFP-A'].resolution(),
            [262144,])

    def test_slice_many_str(self):
        """
        Testing correct reporting of resolution after slicing.

        """
        self.assertListEqual(
            self.d[0][:, ['SSC-H', 'FL2-H', 'FL3-H']].resolution(),
            [1024, 1024, 1024])
        self.assertListEqual(
            self.d[1][:, ['FITC-A', 'PE-A', 'PE-Cy7-A']].resolution(),
            [1024, 1024, 1024])
        self.assertListEqual(
            self.d[2][:, ['FSC', 'SSC', 'TIME']].resolution(),
            [1024, 1024, 262144])
        self.assertListEqual(
            self.d[3][:,['FSC PMT-A', 'FSC PMT-H', 'FSC PMT-W']].resolution(),
            [262144, 262144, 262144])

class TestFCSHistBins(unittest.TestCase):
    """
    Test correct generation, functioning, and slicing of histogram bins.
    We have previously looked at the contents of the $PnR attribute for
    the test files and identified the correct default histogram bin edges
    for linear scaling:
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
        self.d = [FlowCal.io.FCSData(filenames[i]) for i in range(4)]

    def assert_list_of_arrays_equal(self, l1, l2):
        self.assertEqual(len(l1), len(l2))
        for a1, a2 in zip(l1, l2):
            np.testing.assert_array_equal(a1, a2)

    def test_attribute(self):
        """
        Testing correct reporting of hist_bins.
        """
        self.assert_list_of_arrays_equal(
            self.d[0].hist_bins(scale='linear'),
            [np.arange(1025) - 0.5, np.arange(1025) - 0.5,
             np.arange(1025) - 0.5, np.arange(1025) - 0.5,
             np.arange(1025) - 0.5, np.arange(1025) - 0.5])
        self.assert_list_of_arrays_equal(
            self.d[1].hist_bins(scale='linear'),
            [np.arange(1025) - 0.5, np.arange(1025) - 0.5,
             np.arange(1025) - 0.5, np.arange(1025) - 0.5,
             np.arange(1025) - 0.5, np.arange(1025) - 0.5,
             np.arange(1025) - 0.5, np.arange(1025) - 0.5,
             np.arange(1025) - 0.5])
        self.assert_list_of_arrays_equal(
            self.d[2].hist_bins(scale='linear'),
            [np.arange(262145) - 0.5, np.arange(1025) - 0.5,
             np.arange(1025) - 0.5, np.arange(1025) - 0.5,
             np.arange(1025) - 0.5, np.arange(1025) - 0.5,
             np.arange(1025) - 0.5, np.arange(1025) - 0.5])
        self.assert_list_of_arrays_equal(
            self.d[3].hist_bins(scale='linear'),
            [np.arange(262145) - 0.5, np.arange(262145) - 0.5,
             np.arange(262145) - 0.5, np.arange(262145) - 0.5,
             np.arange(262145) - 0.5, np.arange(262145) - 0.5,
             np.arange(262145) - 0.5, np.arange(262145) - 0.5,
             np.arange(262145) - 0.5, np.arange(262145) - 0.5,
             np.arange(262145) - 0.5, np.arange(262145) - 0.5,
             np.arange(262145) - 0.5, np.arange(262145) - 0.5])

    def test_attribute_single(self):
        """
        Testing correct reporting of hist_bins for a single channel.
        """
        np.testing.assert_array_equal(
            self.d[0].hist_bins('FSC-H', scale='linear'),
            np.arange(1025) - 0.5)
        np.testing.assert_array_equal(
            self.d[1].hist_bins('FITC-A', scale='linear'),
            np.arange(1025) - 0.5)
        np.testing.assert_array_equal(
            self.d[2].hist_bins('SSC', scale='linear'),
            np.arange(1025) - 0.5)
        np.testing.assert_array_equal(
            self.d[3].hist_bins('GFP-A', scale='linear'),
            np.arange(262145) - 0.5)

    def test_attribute_many(self):
        """
        Testing correct reporting of hist_bins for many channels.
        """
        self.assert_list_of_arrays_equal(
            self.d[0].hist_bins(['SSC-H',
                                 'FL2-H',
                                 'FL3-H'], scale='linear'),
            [np.arange(1025) - 0.5,
             np.arange(1025) - 0.5,
             np.arange(1025) - 0.5])
        self.assert_list_of_arrays_equal(
            self.d[1].hist_bins(['FITC-A',
                                 'PE-A',
                                 'PE-Cy7-A'], scale='linear'),
            [np.arange(1025) - 0.5,
             np.arange(1025) - 0.5,
             np.arange(1025) - 0.5])
        self.assert_list_of_arrays_equal(
            self.d[2].hist_bins(['FSC',
                                 'SSC',
                                 'TIME'], scale='linear'),
            [np.arange(1025) - 0.5,
             np.arange(1025) - 0.5,
             np.arange(262145) - 0.5])
        self.assert_list_of_arrays_equal(
            self.d[3].hist_bins(['FSC PMT-A',
                                 'FSC PMT-H',
                                 'FSC PMT-W'], scale='linear'),
            [np.arange(262145) - 0.5,
             np.arange(262145) - 0.5,
             np.arange(262145) - 0.5])

    def test_slice_single_str(self):
        """
        Testing correct reporting of hist_bins after slicing.
        """
        self.assert_list_of_arrays_equal(
            self.d[0][:, 'FSC-H'].hist_bins(scale='linear'),
            [np.arange(1025) - 0.5])
        self.assert_list_of_arrays_equal(
            self.d[1][:, 'FITC-A'].hist_bins(scale='linear'),
            [np.arange(1025) - 0.5])
        self.assert_list_of_arrays_equal(
            self.d[2][:, 'SSC'].hist_bins(scale='linear'),
            [np.arange(1025) - 0.5])
        self.assert_list_of_arrays_equal(
            self.d[3][:, 'GFP-A'].hist_bins(scale='linear'),
            [np.arange(262145) - 0.5])

    def test_slice_many_str(self):
        """
        Testing correct reporting of hist_bins after slicing.
        """
        self.assert_list_of_arrays_equal(
            self.d[0][:, ['SSC-H',
                          'FL2-H',
                          'FL3-H']].hist_bins(scale='linear'),
            [np.arange(1025) - 0.5,
             np.arange(1025) - 0.5,
             np.arange(1025) - 0.5])
        self.assert_list_of_arrays_equal(
            self.d[1][:, ['FITC-A',
                          'PE-A',
                          'PE-Cy7-A']].hist_bins(scale='linear'),
            [np.arange(1025) - 0.5,
             np.arange(1025) - 0.5,
             np.arange(1025) - 0.5])
        self.assert_list_of_arrays_equal(
            self.d[2][:, ['FSC',
                          'SSC',
                          'TIME']].hist_bins(scale='linear'),
            [np.arange(1025) - 0.5,
             np.arange(1025) - 0.5,
             np.arange(262145) - 0.5])
        self.assert_list_of_arrays_equal(
            self.d[3][:,['FSC PMT-A',
                         'FSC PMT-H',
                         'FSC PMT-W']].hist_bins(scale='linear'),
            [np.arange(262145) - 0.5,
             np.arange(262145) - 0.5,
             np.arange(262145) - 0.5])

    def test_nondefault_nbins(self):
        """
        Testing correct generation of hist_bins with a non-default nbins
        """
        for nbins in [128, 256, 512]:
            # Generate proper bins with the method previously used in plot.hist
            bd = np.arange(1025) - 0.5
            xd = np.linspace(0, 1, len(bd))
            xs = np.linspace(0, 1, nbins + 1)
            bins = np.interp(xs, xd, bd)
            # Generate with FCSData.hist_bins and compare
            np.testing.assert_array_equal(
                self.d[0].hist_bins('FSC-H',
                                    nbins=nbins,
                                    scale='linear'), bins)

    def test_nondefault_nbins_many_1(self):
        """
        Testing correct generation of hist_bins with a non-default nbins
        """
        # Generate proper bins with the method previously used in plot.hist
        bd = np.arange(1025) - 0.5
        xd = np.linspace(0, 1, len(bd))
        xs = np.linspace(0, 1, 256 + 1)
        bins1 = np.interp(xs, xd, bd)
        # Generate with FCSData.hist_bins and compare
        self.assert_list_of_arrays_equal(
            self.d[0].hist_bins(['FL1-H', 'FL2-H'],
                                nbins=256,
                                scale='linear'),
            [bins1, bins1])

    def test_nondefault_nbins_many_2(self):
        """
        Testing correct generation of hist_bins with a non-default nbins
        """
        # Generate proper bins with the method previously used in plot.hist
        bd = np.arange(1025) - 0.5
        xd = np.linspace(0, 1, len(bd))
        xs = np.linspace(0, 1, 256 + 1)
        bins1 = np.interp(xs, xd, bd)
        xs = np.linspace(0, 1, 512 + 1)
        bins2 = np.interp(xs, xd, bd)
        # Generate with FCSData.hist_bins and compare
        self.assert_list_of_arrays_equal(
            self.d[0].hist_bins(['FL1-H', 'FL2-H'],
                                nbins=[256, 512],
                                scale='linear'),
            [bins1, bins2])

    def test_nondefault_nbins_many_3(self):
        """
        Testing correct generation of hist_bins with a non-default nbins
        """
        # Generate proper bins with the method previously used in plot.hist
        bd = np.arange(1025) - 0.5
        xd = np.linspace(0, 1, len(bd))
        xs = np.linspace(0, 1, 256 + 1)
        bins1 = np.interp(xs, xd, bd)
        # Generate with FCSData.hist_bins and compare
        self.assert_list_of_arrays_equal(
            self.d[0].hist_bins(['FL1-H', 'FL2-H'],
                                nbins=[256, None],
                                scale='linear'),
            [bins1, bd])

    def test_nondefault_nbins_many_4(self):
        """
        Testing correct generation of hist_bins with a non-default nbins
        """
        # Generate proper bins with the method previously used in plot.hist
        bd = np.arange(1025) - 0.5
        xd = np.linspace(0, 1, len(bd))
        xs = np.linspace(0, 1, 256 + 1)
        bins1 = np.interp(xs, xd, bd)
        xs = np.linspace(0, 1, 512 + 1)
        bins2 = np.interp(xs, xd, bd)
        # Generate with FCSData.hist_bins and compare
        self.assert_list_of_arrays_equal(
            self.d[2].hist_bins(['FL1', 'FL2', 'FL3'],
                                nbins=[256, None, 512],
                                scale='linear'),
            [bins1, bd, bins2])

    def test_nondefault_nbins_many_5(self):
        """
        Testing correct generation of hist_bins with a non-default nbins
        """
        # Generate proper bins with the method previously used in plot.hist
        bd = np.arange(1025) - 0.5
        xd = np.linspace(0, 1, len(bd))
        xs = np.linspace(0, 1, 256 + 1)
        bins1 = np.interp(xs, xd, bd)
        xs = np.linspace(0, 1, 512 + 1)
        bins2 = np.interp(xs, xd, bd)
        # Generate with FCSData.hist_bins and compare
        self.assert_list_of_arrays_equal(
            self.d[2].hist_bins(['FL1', 'TIME', 'FL3'],
                                nbins=[256, None, 512],
                                scale='linear'),
            [bins1, np.arange(262145) - 0.5, bins2])

class TestFCSAttributes(unittest.TestCase):
    def setUp(self):
        self.d = FlowCal.io.FCSData(filenames[0])
        self.n_samples = self.d.shape[0]

    def test_str(self):
        """
        Testing string representation.

        """
        self.assertEqual(str(self.d), 'Data001.fcs')

    def test_data_type(self):
        """
        Testing of data type.

        """
        # Data001.fcs is a FCS2.0 file, with integer data.
        # We have previously looked at self.d.text['$DATATYPE']) to determine
        # the correct output for this file.
        self.assertEqual(self.d.data_type, 'I')

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
        self.d = FlowCal.io.FCSData(filenames[2])
        self.n_samples = self.d.shape[0]

    def test_str(self):
        """
        Testing string representation.

        """
        self.assertEqual(str(self.d), 'Data003.fcs')

    def test_data_type(self):
        """
        Testing of the data type.

        """
        # Data003.fcs is a FCS3.0 file, with integer data.
        # We have previously looked at self.d.text['$DATATYPE']) to determine
        # the correct output for this file.
        self.assertEqual(self.d.data_type, 'I')

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

class TestFCSAttributesFloating(unittest.TestCase):
    def setUp(self):
        self.d = FlowCal.io.FCSData(filenames[3])
        self.n_samples = self.d.shape[0]

    def test_str(self):
        """
        Testing string representation.

        """
        self.assertEqual(str(self.d), 'Data004.fcs')

    def test_data_type(self):
        """
        Testing of the data type.

        """
        # Data004.fcs is a FCS3.0 file, with floating-point data.
        # We have previously looked at self.d.text['$DATATYPE']) to determine
        # the correct output for this file.
        self.assertEqual(self.d.data_type, 'F')

    def test_time_step(self):
        """
        Testing of the time step.

        """
        # Data004.fcs is a FCS3.0 file, use the $TIMESTEP parameter.
        # We have previously looked at self.d.text['$TIMESTEP']) to determine
        # the correct output for this file.
        self.assertEqual(self.d.time_step, 0.01)

    def test_acquisition_start_time(self):
        """
        Testing of acquisition start time.

        """
        # We have previously looked at the $BTIM and #DATE attributes of
        # Data004.fcs to determine the correct value of acquisition_start_time.
        time_correct = datetime.datetime(2015, 5, 29, 17, 10, 23)
        self.assertEqual(self.d.acquisition_start_time, time_correct)

    def test_acquisition_end_time(self):
        """
        Testing of acquisition end time.

        """
        # We have previously looked at the $ETIM and #DATE attributes of
        # Data004.fcs to determine the correct value of acquisition_end_time.
        time_correct = datetime.datetime(2015, 5, 29, 17, 10, 27)
        self.assertEqual(self.d.acquisition_end_time, time_correct)

    def test_acquisition_time_event(self):
        """
        Testing acquisition time.

        """
        # Data004.fcs has the time channel, so the acquisition time should be
        # calculated using the event list.
        # We have previously looked at self.d[0, 'TIME'] and self.d[-1, 'TIME']
        # to determine the correct output for this file.
        self.assertEqual(self.d.acquisition_time, 1.7029998779296875)

    def test_acquisition_time_btim_etim(self):
        """
        Testing acquisition time using the btim/etim method.

        """
        # Data004.fcs has the time channel, so we will remove it so that the
        # BTIM and ETIM keyword arguments are used.
        # We have previously looked at d.text['$BTIM'] and d.text['$ETIM'] to
        # determine the correct output for this file.
        d = self.d[:,['FSC-A', 'FSC-H', 'FSC-W',
                      'SSC-A', 'SSC-H', 'SSC-W',
                      'FSC PMT-A', 'FSC PMT-H', 'FSC PMT-W',
                      'GFP-A', 'GFP-H',
                      'mCherry-A', 'mCherry-H']]
        self.assertEqual(d.acquisition_time, 4)

class TestFCSDataSlicing(unittest.TestCase):
    def setUp(self):
        self.d = FlowCal.io.FCSData(filenames[0])
        self.n_samples = self.d.shape[0]

    def test_1d_slicing_with_scalar(self):
        """
        Testing the 1D slicing with a scalar of a FCSData object.

        """
        ds = self.d[1]
        self.assertIsInstance(ds, FlowCal.io.FCSData)
        self.assertEqual(ds.shape, (6,))
        self.assertEqual(ds.channels,
            ('FSC-H', 'SSC-H', 'FL1-H', 'FL2-H', 'FL3-H', 'Time'))

    def test_1d_slicing_with_list(self):
        """
        Testing the 1D slicing with a list of a FCSData object.

        """
        ds = self.d[range(10)]
        self.assertIsInstance(ds, FlowCal.io.FCSData)
        self.assertEqual(ds.shape, (10,6))
        self.assertEqual(ds.channels,
            ('FSC-H', 'SSC-H', 'FL1-H', 'FL2-H', 'FL3-H', 'Time'))

    def test_slicing_channel_with_int(self):
        """
        Testing the channel slicing with an int of a FCSData object.

        """
        ds = self.d[:,2]
        self.assertIsInstance(ds, FlowCal.io.FCSData)
        self.assertEqual(ds.shape, (self.n_samples,))
        self.assertEqual(ds.channels, ('FL1-H',))

    def test_slicing_channel_with_string(self):
        """
        Testing the channel slicing with a string of a FCSData object.

        """
        ds = self.d[:,'SSC-H']
        self.assertIsInstance(ds, FlowCal.io.FCSData)
        self.assertEqual(ds.shape, (self.n_samples,))
        self.assertEqual(ds.channels, ('SSC-H', ))

    def test_slicing_channel_with_int_array(self):
        """
        Testing the channel slicing with an int array of a FCSData 
        object.
        """
        ds = self.d[:,[1,3]]
        self.assertIsInstance(ds, FlowCal.io.FCSData)
        self.assertEqual(ds.shape, (self.n_samples,2))
        self.assertEqual(ds.channels, ('SSC-H', 'FL2-H'))

    def test_slicing_channel_with_string_array(self):
        """
        Testing the channel slicing with a string array of a FCSData 
        object.

        """
        ds = self.d[:,['FSC-H', 'FL3-H']]
        self.assertIsInstance(ds, FlowCal.io.FCSData)
        self.assertEqual(ds.shape, (self.n_samples,2))
        self.assertEqual(ds.channels, ('FSC-H', 'FL3-H'))

    def test_slicing_sample(self):
        """
        Testing the sample slicing of a FCSData object.

        """
        ds = self.d[:1000]
        self.assertIsInstance(ds, FlowCal.io.FCSData)
        self.assertEqual(ds.shape, (1000,6))
        self.assertEqual(ds.channels,
            ('FSC-H', 'SSC-H', 'FL1-H', 'FL2-H', 'FL3-H', 'Time'))

    def test_2d_slicing(self):
        """
        Testing 2D slicing of a FCSData object.

        """
        ds = self.d[:1000,['SSC-H', 'FL3-H']]
        self.assertIsInstance(ds, FlowCal.io.FCSData)
        self.assertEqual(ds.shape, (1000,2))
        self.assertEqual(ds.channels, ('SSC-H', 'FL3-H'))

    def test_mask_slicing(self):
        """
        Testing mask slicing of a FCSData object.

        """
        m = self.d[:,1]>500
        ds = self.d[m,:]
        self.assertIsInstance(ds, FlowCal.io.FCSData)
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
        self.assertIsInstance(ds, FlowCal.io.FCSData)
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
        self.assertIsInstance(ds, FlowCal.io.FCSData)
        self.assertEqual(ds.channels,
            ('FSC-H', 'SSC-H', 'FL1-H', 'FL2-H', 'FL3-H', 'Time'))
        np.testing.assert_array_equal(ds[:,0], self.d[:,0])
        np.testing.assert_array_equal(ds[:,1], 5)
        np.testing.assert_array_equal(ds[:,2], 5)
        np.testing.assert_array_equal(ds[:,3], self.d[:,3])
        np.testing.assert_array_equal(ds[:,4], self.d[:,4])


class TestFCSDataOperations(unittest.TestCase):
    def setUp(self):
        self.d = FlowCal.io.FCSData(filenames[0])
        self.n_samples = self.d.shape[0]

    def test_sum_integer(self):
        """
        Testing that scalar + FCSData is consistent with scalar + ndarray

        """
        ds = self.d + 3
        ds_array = self.d.view(np.ndarray) + 3
        self.assertIsInstance(ds, FlowCal.io.FCSData)
        np.testing.assert_array_equal(ds, ds_array)
        
    def test_sqrt(self):
        """
        Testing that the sqrt(FCSData) is consistent with sqrt(ndarray)

        """
        ds = np.sqrt(self.d)
        ds_array = np.sqrt(self.d.view(np.ndarray))
        self.assertIsInstance(ds, FlowCal.io.FCSData)
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
        self.assertIsInstance(m, FlowCal.io.FCSData)
        self.assertEqual(m.shape, m_array.shape)
        np.testing.assert_array_equal(m, m_array)
        
if __name__ == '__main__':
    unittest.main()
