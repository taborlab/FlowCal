"""
Unit tests for the excel_ui module.

"""

import os
import collections
import unittest

import numpy as np
import pandas as pd
import pandas.util.testing as tm

import fc

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
        table = fc.excel_ui.read_table(self.filename,
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
        table = fc.excel_ui.read_table(self.filename,
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
        table = fc.excel_ui.read_table(self.filename,
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
                          fc.excel_ui.read_table,
                          self.filename,
                          sheetname,
                          index_col)

if __name__ == '__main__':
    unittest.main()
