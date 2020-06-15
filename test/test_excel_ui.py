"""
Unit tests for the excel_ui module.

"""

import os
import collections
import unittest

import numpy as np
import pandas as pd
import pandas.util.testing as tm

import FlowCal

class TestReadTable(unittest.TestCase):
    """
    Class to test excel_ui.read_table()

    """
    def setUp(self):
        # Name of the file to read
        self.filename = 'test/test_excel_ui.xlsx'

    def test_read_table(self):
        """
        Test for proper loading of a table from an Excel sheet.

        """
        # Sheet to read
        sheetname = "Instruments"
        # Column to use as index labels
        index_col = "ID"

        # Expected output
        expected_output_list = []
        row = {}
        row[u'Description'] = u'Moake\'s Flow Cytometer'
        row[u'Forward Scatter Channel'] = u'FSC-H'
        row[u'Side Scatter Channel'] = u'SSC-H'
        row[u'Fluorescence Channels'] = u'FL1-H, FL2-H, FL3-H'
        row[u'Time Channel'] = u'Time'
        expected_output_list.append(row)
        row = {}
        row[u'Description'] = u'Moake\'s Flow Cytometer (new acquisition card)'
        row[u'Forward Scatter Channel'] = u'FSC'
        row[u'Side Scatter Channel'] = u'SSC'
        row[u'Fluorescence Channels'] = u'FL1, FL2, FL3'
        row[u'Time Channel'] = u'TIME'
        expected_output_list.append(row)
        expected_index = pd.Series([u'FC001', u'FC002'], name='ID')
        expected_columns = [u'Description',
                            u'Forward Scatter Channel',
                            u'Side Scatter Channel',
                            u'Fluorescence Channels',
                            u'Time Channel']

        expected_output = pd.DataFrame(expected_output_list,
                                       index=expected_index,
                                       columns=expected_columns)

        # Read table
        table = FlowCal.excel_ui.read_table(self.filename,
                                            sheetname=sheetname,
                                            index_col=index_col)

        # Compare
        tm.assert_frame_equal(table, expected_output)

    def test_read_table_xls(self):
        """
        Test for proper loading of a table from an old-format Excel sheet.

        """
        xls_filename = 'test/test_excel_ui.xls'

        # Sheet to read
        sheetname = "Instruments"
        # Column to use as index labels
        index_col = "ID"

        # Expected output
        expected_output_list = []
        row = {}
        row[u'Description'] = u'Moake\'s Flow Cytometer'
        row[u'Forward Scatter Channel'] = u'FSC-H'
        row[u'Side Scatter Channel'] = u'SSC-H'
        row[u'Fluorescence Channels'] = u'FL1-H, FL2-H, FL3-H'
        row[u'Time Channel'] = u'Time'
        expected_output_list.append(row)
        row = {}
        row[u'Description'] = u'Moake\'s Flow Cytometer (new acquisition card)'
        row[u'Forward Scatter Channel'] = u'FSC'
        row[u'Side Scatter Channel'] = u'SSC'
        row[u'Fluorescence Channels'] = u'FL1, FL2, FL3'
        row[u'Time Channel'] = u'TIME'
        expected_output_list.append(row)
        expected_index = pd.Series([u'FC001', u'FC002'], name='ID')
        expected_columns = [u'Description',
                            u'Forward Scatter Channel',
                            u'Side Scatter Channel',
                            u'Fluorescence Channels',
                            u'Time Channel']

        expected_output = pd.DataFrame(expected_output_list,
                                       index=expected_index,
                                       columns=expected_columns)

        # Read table
        table = FlowCal.excel_ui.read_table(xls_filename,
                                            sheetname=sheetname,
                                            index_col=index_col)

        # Compare
        tm.assert_frame_equal(table, expected_output)

    def test_read_table_no_index_col(self):
        """
        Test proper loading of a table when no index column is specified.

        """
        # Sheet to read
        sheetname = "Instruments"

        # Expected output
        expected_output_list = []
        row = {}
        row[u'ID'] = u'FC001'
        row[u'Description'] = u'Moake\'s Flow Cytometer'
        row[u'Forward Scatter Channel'] = u'FSC-H'
        row[u'Side Scatter Channel'] = u'SSC-H'
        row[u'Fluorescence Channels'] = u'FL1-H, FL2-H, FL3-H'
        row[u'Time Channel'] = u'Time'
        expected_output_list.append(row)
        row = {}
        row[u'ID'] = u'FC002'
        row[u'Description'] = u'Moake\'s Flow Cytometer (new acquisition card)'
        row[u'Forward Scatter Channel'] = u'FSC'
        row[u'Side Scatter Channel'] = u'SSC'
        row[u'Fluorescence Channels'] = u'FL1, FL2, FL3'
        row[u'Time Channel'] = u'TIME'
        expected_output_list.append(row)
        expected_columns = [u'ID',
                            u'Description',
                            u'Forward Scatter Channel',
                            u'Side Scatter Channel',
                            u'Fluorescence Channels',
                            u'Time Channel']

        expected_output = pd.DataFrame(expected_output_list,
                                       columns=expected_columns)

        # Read table
        table = FlowCal.excel_ui.read_table(self.filename,
                                            sheetname=sheetname)

        # Compare
        tm.assert_frame_equal(table, expected_output)

    def test_read_table_with_empty_row(self):
        """
        Test for proper loading of a table that includes an empty row.

        """
        # Sheet to read
        sheetname = "Instruments (empty row)"
        # Column to use as index labels
        index_col = "ID"

        # Expected output
        expected_output_list = []
        row = {}
        row[u'Description'] = u'Moake\'s Flow Cytometer'
        row[u'Forward Scatter Channel'] = u'FSC-H'
        row[u'Side Scatter Channel'] = u'SSC-H'
        row[u'Fluorescence Channels'] = u'FL1-H, FL2-H, FL3-H'
        row[u'Time Channel'] = u'Time'
        expected_output_list.append(row)
        row = {}
        row[u'Description'] = u'Moake\'s Flow Cytometer (new acquisition card)'
        row[u'Forward Scatter Channel'] = u'FSC'
        row[u'Side Scatter Channel'] = u'SSC'
        row[u'Fluorescence Channels'] = u'FL1, FL2, FL3'
        row[u'Time Channel'] = u'TIME'
        expected_output_list.append(row)
        row = {}
        row[u'Description'] = u'Some other flow cytometer'
        row[u'Forward Scatter Channel'] = u'FSC-A'
        row[u'Side Scatter Channel'] = u'SSC-A'
        row[u'Fluorescence Channels'] = u'FL1-A, FL2-A, FL3-A'
        row[u'Time Channel'] = u'Time'
        expected_output_list.append(row)
        expected_index = pd.Series([u'FC001', u'FC002', u'FC003'], name='ID')
        expected_columns = [u'Description',
                            u'Forward Scatter Channel',
                            u'Side Scatter Channel',
                            u'Fluorescence Channels',
                            u'Time Channel']

        expected_output = pd.DataFrame(expected_output_list,
                                       index=expected_index,
                                       columns=expected_columns)

        # Read table
        table = FlowCal.excel_ui.read_table(self.filename,
                                            sheetname=sheetname,
                                            index_col=index_col)

        # Compare
        tm.assert_frame_equal(table, expected_output)

    def test_read_table_duplicated_id_error(self):
        """
        Test for error when table contains duplicated index values.

        """
        # Sheet to read
        sheetname = "Instruments (duplicated)"
        # Column to use as index labels
        index_col = "ID"

        # Call function
        self.assertRaises(ValueError,
                          FlowCal.excel_ui.read_table,
                          self.filename,
                          sheetname,
                          index_col)

    def test_read_table_list_argument_error(self):
        """
        Test for error when `sheetname` is a list.

        """
        # Sheet to read
        sheetname = ["Instruments", "Instruments (duplicated)"]
        # Column to use as index labels
        index_col = "ID"

        # Call function
        self.assertRaises(TypeError,
                          FlowCal.excel_ui.read_table,
                          self.filename,
                          sheetname,
                          index_col)

    def test_read_table_none_argument_error(self):
        """
        Test for error when `sheetname` is None.

        """
        # Sheet to read
        sheetname = None
        # Column to use as index labels
        index_col = "ID"

        # Call function
        self.assertRaises(TypeError,
                          FlowCal.excel_ui.read_table,
                          self.filename,
                          sheetname,
                          index_col)

class TestRegexMEFValues(unittest.TestCase):
    """
    Class to test the regex for finding MEF Values columns.

    """
    def setUp(self):
        # Test dataset
        # Each tuple consists of a string and the corresponding match solution.
        # If the string is not expected to match, the solution is None.
        # If the string is expected to match, the solution is the expected
        # contents of group 1.
        self.test_data = [('FL1 MEF Values', 'FL1'),
                          (' FL1 MEF Values ', 'FL1'),
                          ('   FL1 MEF Values', 'FL1'),
                          ('FL1-H MEF Values', 'FL1-H'),
                          ('mCherry-A MEF Values', 'mCherry-A'),
                          ('MEF Values', None),
                          (' MEF Values', None),
                          ('  MEF Values', None),
                          (' MEF Values ', None),
                          ('test', None),
                          (' test', None),
                          ('  test', None),
                          (' test ', None),
                          ('test MEF Values', 'test'),
                          ('Parameter 14', None),
                          ('1', None),
                          ('1 MEF Values', '1'),
                          ('13', None),
                          ('13 MEF Values', '13'),
                          ('Parameter 14MEF Values', None),
                          ('Parameter 14 MEFValues', None),
                          ('Parameter 14 MEF Values', 'Parameter 14'),
                          ('Parameter 14 MEF  Values', 'Parameter 14'),
                          ('Parameter 14  MEF Values', 'Parameter 14'),
                          ('Parameter  14 MEF Values', 'Parameter  14'),
                          ('Parameter   14 MEF Values', 'Parameter   14'),
                          ('  here is another test of ', None),
                          ('  here is another test of MEF Values', 'here is another test of'),
                          ('  test test MEF Values', 'test test'),
                          ('520nm Light Intensity (umol/(m^2*s))', None),
                          ('520nm Light Intensity (umol/(m^2*s)) MEF Values', '520nm Light Intensity (umol/(m^2*s))'),
                          ]

    def test_match(self):
        # Get compiled regex
        r = FlowCal.excel_ui.re_mef_values
        # Iterate over test data
        for s, sol in self.test_data:
            # Match
            m = r.match(s)
            # Check
            if sol is None:
                self.assertIsNone(m)
            else:
                self.assertEqual(sol, m.group(1))

class TestRegexUnits(unittest.TestCase):
    """
    Class to test the regex for finding Units columns.

    """
    def setUp(self):
        # Test dataset
        # Each tuple consists of a string and the corresponding match solution.
        # If the string is not expected to match, the solution is None.
        # If the string is expected to match, the solution is the expected
        # contents of group 1.
        self.test_data = [('FL1 Units', 'FL1'),
                          (' FL1 Units ', 'FL1'),
                          ('   FL1 Units', 'FL1'),
                          ('FL1-H Units', 'FL1-H'),
                          ('mCherry-A Units', 'mCherry-A'),
                          ('Units', None),
                          (' Units', None),
                          ('  Units', None),
                          (' Units ', None),
                          ('test', None),
                          (' test', None),
                          ('  test', None),
                          (' test ', None),
                          ('test Units', 'test'),
                          ('Parameter 14', None),
                          ('1', None),
                          ('1 Units', '1'),
                          ('13', None),
                          ('13 Units', '13'),
                          ('Parameter 14Units', None),
                          ('Parameter 14 Units', 'Parameter 14'),
                          ('Parameter  14 Units', 'Parameter  14'),
                          ('Parameter   14 Units', 'Parameter   14'),
                          ('  here is another test of ', None),
                          ('  here is another test of Units', 'here is another test of'),
                          ('  test test Units', 'test test'),
                          ('520nm Light Intensity (umol/(m^2*s))', None),
                          ('520nm Light Intensity (umol/(m^2*s)) Units', '520nm Light Intensity (umol/(m^2*s))'),
                          ]

    def test_match(self):
        # Get compiled regex
        r = FlowCal.excel_ui.re_units
        # Iterate over test data
        for s, sol in self.test_data:
            # Match
            m = r.match(s)
            # Check
            if sol is None:
                self.assertIsNone(m)
            else:
                self.assertEqual(sol, m.group(1))

if __name__ == '__main__':
    unittest.main()
