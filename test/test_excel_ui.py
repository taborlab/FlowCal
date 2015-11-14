"""Unit tests for the excel_ui module.

Author: Sebastian M. Castillo-Hair (smc9@rice.edu)

Last Modified: 10/26/2015

Requires:
    - fc.excel_ui

"""

import os
import collections
import unittest
import numpy as np

import fc

class TestReadWorkbook(unittest.TestCase):
    """Class to test excel_ui.read_workbook()

    """
    def setUp(self):
        # Name of the file to read
        self.filename = 'examples/experiment.xlsx'

        # Expected contents of workbook
        self.content_expected = collections.OrderedDict()

        # Instruments sheet
        sheet_contents = [[u'ID',
                           u'Description',
                           u'Forward Scatter Channel',
                           u'Side Scatter Channel',
                           u'Fluorescence Channels',
                           u'Time Channel',
                           ],
                          [u'FC001',
                           u'Moake\'s Flow Cytometer',
                           u'FSC-H',
                           u'SSC-H',
                           u'FL1-H, FL2-H, FL3-H',
                           u'Time',
                           ],
                          [u'FC002',
                           u'Moake\'s Flow Cytometer (new acquisition card)',
                           u'FSC',
                           u'SSC',
                           u'FL1, FL2, FL3',
                           u'TIME',
                           ],
                          ]
        self.content_expected['Instruments'] = sheet_contents

        # Beads sheet
        sheet_contents = [[u'ID',
                           u'Instrument ID',
                           u'File Path',
                           u'Lot',
                           u'FL1-H MEF Values',
                           u'FL1 MEF Values',
                           u'FL3 MEF Values',
                           u'Gate Fraction',
                           u'Clustering Channels',
                           ],
                          [u'B0001',
                           u'FC001',
                           u'FCFiles/data.006',
                           u'AF02',
                           u'0, 792, 2079, 6588, 16471, 47497, 137049, 271647',
                           '',
                           '',
                           0.3,
                           u'FL1-H',
                           ],
                          [u'B0002',
                           u'FC002',
                           u'FCFiles/data_006.fcs',
                           u'AF02',
                           '',
                           u'0, 792, 2079, 6588, 16471, 47497, 137049, 271647',
                           u'0, 1614, 4035, 12025, 31896, 95682, 353225, 1077421',
                           0.3,
                           u'FL1, FL3',
                           ],
                          ]
        self.content_expected['Beads'] = sheet_contents

        # Samples sheet
        sheet_contents = [[u'ID',
                           u'Instrument ID',
                           u'Beads ID',
                           u'File Path',
                           u'FL1-H Units',
                           u'FL1 Units',
                           u'FL2 Units',
                           u'FL3 Units',
                           u'Gate Fraction',
                           u'Strain name',
                           u'IPTG (\xb5M)',
                           ],
                          [u'S0001',
                           u'FC001',
                           '',
                           u'FCFiles/data.001',
                           '',
                           '',
                           '',
                           '',
                           0.3,
                           u'sSC0001',
                           0.0,
                           ],
                          [u'S0002',
                           u'FC001',
                           '',
                           u'FCFiles/data.002',
                           'Channel',
                           '',
                           '',
                           '',
                           0.3,
                           u'sSC0001',
                           0.0,
                           ],
                          [u'S0003',
                           u'FC001',
                           '',
                           u'FCFiles/data.003',
                           'RFI',
                           '',
                           '',
                           '',
                           0.3,
                           u'sSC0001',
                           0.0,
                           ],
                          [u'S0004',
                           u'FC001',
                           u'B0001',
                           u'FCFiles/data.004',
                           'MEF',
                           '',
                           '',
                           '',
                           0.3,
                           u'sSC0001',
                           1.0,
                           ],
                          [u'S0005',
                           u'FC001',
                           u'B0001',
                           u'FCFiles/data.005',
                           'MEF',
                           '',
                           '',
                           '',
                           0.3,
                           u'sSC0001',
                           5.0,
                           ],
                          [u'S0006',
                           u'FC002',
                           '',
                           u'FCFiles/data_001.fcs',
                           '',
                           '',
                           '',
                           '',
                           0.3,
                           u'sSC0001',
                           0.0,
                           ],
                          [u'S0007',
                           u'FC002',
                           '',
                           u'FCFiles/data_002.fcs',
                           '',
                           u'Channel',
                           '',
                           '',
                           0.2,
                           u'sSC0001',
                           0.0,
                           ],
                          [u'S0008',
                           u'FC002',
                           '',
                           u'FCFiles/data_003.fcs',
                           '',
                           'RFI',
                           '',
                           '',
                           0.2,
                           u'sSC0001',
                           0.0,
                           ],
                          [u'S0009',
                           u'FC002',
                           u'B0002',
                           u'FCFiles/data_004.fcs',
                           '',
                           'MEF',
                           'RFI',
                           '',
                           0.25,
                           u'sSC0001',
                           1.0,
                           ],
                          [u'S0010',
                           u'FC002',
                           u'B0002',
                           u'FCFiles/data_005.fcs',
                           '',
                           'MEF',
                           '',
                           'MEF',
                           0.3,
                           u'sSC0001',
                           5.0,
                           ],
                          [u'S0011',
                           u'FC002',
                           u'B0002',
                           u'FCFiles/data_004.fcs',
                           '',
                           'MEF',
                           '',
                           '',
                           0.1,
                           u'sSC0001',
                           1.0,
                           ],
                          ]
        self.content_expected['Samples'] = sheet_contents

    def test_read_workbook(self):
        """Test for proper reading of an Excel workbook.

        """
        # Load contents from the workbook
        content = fc.excel_ui.read_workbook(self.filename)
        # Compare with expected content
        self.assertEqual(self.content_expected, content)

class TestWriteWorkbook(unittest.TestCase):
    """Class to test excel_ui.write_workbook()

    """
    def setUp(self):
        # Name of the file to write to
        self.filename = 'test/test_write_workbook.xlsx'
        # Contents to write
        self.content = collections.OrderedDict()
        self.content['sheet_1'] = [[u'row1', u'row2'],
                                   [1, 2],
                                   [3, 5],
                                  ]
        self.content['sheet 2'] = [[u'abcd', u'efg', u'hijkl'],
                                   [0, 1, 2], 
                                   [1, 4, 9],
                                   [27, 8, 1],
                                  ]

    def tearDown(self):
        # Delete create excel file
        if os.path.isfile(self.filename):
            os.remove(self.filename)

    def test_write_workbook(self):
        """Test for proper writing of an Excel workbook.

        """
        # Write excel workbook
        fc.excel_ui.write_workbook(self.filename, self.content)
        # Load excel workbook and compare contents
        read_content = fc.excel_ui.read_workbook(self.filename)
        self.assertEqual(self.content, read_content)

    def test_write_workbook_content_is_not_dict_error(self):
        """Test that using a list as the content raises a TypeError.

        """
        # 
        self.assertRaises(TypeError,
                          fc.excel_ui.write_workbook,
                          self.filename,
                          ['Item 1', 'Item 2'])

    def test_write_workbook_content_is_empty_error(self):
        """Test that using an empty OrderedDict as the content raises a
        ValueError.

        """
        # 
        self.assertRaises(ValueError,
                          fc.excel_ui.write_workbook,
                          self.filename,
                          collections.OrderedDict())

    def test_write_workbook_filename_error(self):
        """Test that writing to a bad file name raises an IOError.

        """
        # 
        self.assertRaises(IOError,
                          fc.excel_ui.write_workbook,
                          '',
                          self.content)

class TestTableConversion(unittest.TestCase):
    """Class to test excel_ui.list_to_table() and excel_ui.table_to_list()

    """
    def test_list_to_table(self):
        """Test that excel_ui.list_to_table produces the correct output.

        """
        # Input data
        input_table = [[u'ID',
                        u'Description',
                        u'Forward Scatter Channel',
                        u'Side Scatter Channel',
                        u'Fluorescence Channels',
                        u'Time Channel',
                        ],
                       [u'FC001',
                        u'Moake\'s Flow Cytometer',
                        u'FSC-H',
                        u'SSC-H',
                        u'FL1-H, FL2-H, FL3-H',
                        u'Time',
                        ],
                       [u'FC002',
                        u'Moake\'s Flow Cytometer (new acquisition card)',
                        u'FSC',
                        u'SSC',
                        u'FL1, FL2, FL3',
                        u'TIME',
                        ],
                       ]
        id_header = 'ID'

        # Expected output
        expected_output = collections.OrderedDict()

        row = collections.OrderedDict()
        row[u'ID'] = u'FC001'
        row[u'Description'] = u'Moake\'s Flow Cytometer'
        row[u'Forward Scatter Channel'] = u'FSC-H'
        row[u'Side Scatter Channel'] = u'SSC-H'
        row[u'Fluorescence Channels'] = u'FL1-H, FL2-H, FL3-H'
        row[u'Time Channel'] = u'Time'
        expected_output['FC001'] = row

        row = collections.OrderedDict()
        row[u'ID'] = u'FC002'
        row[u'Description'] = u'Moake\'s Flow Cytometer (new acquisition card)'
        row[u'Forward Scatter Channel'] = u'FSC'
        row[u'Side Scatter Channel'] = u'SSC'
        row[u'Fluorescence Channels'] = u'FL1, FL2, FL3'
        row[u'Time Channel'] = u'TIME'
        expected_output['FC002'] = row

        # Run list_to_table
        table = fc.excel_ui.list_to_table(input_table, id_header)

        # Compare to expected output
        self.assertEqual(table, expected_output)


    def test_list_to_table_ignore_empty_id(self):
        """Test that excel_ui.list_to_table produces the correct output when
        the input has a row that should with an empty `header_id` field.

        """
        # Input data
        input_table = [[u'ID',
                        u'Description',
                        u'Forward Scatter Channel',
                        u'Side Scatter Channel',
                        u'Fluorescence Channels',
                        u'Time Channel',
                        ],
                       [u'FC001',
                        u'Moake\'s Flow Cytometer',
                        u'FSC-H',
                        u'SSC-H',
                        u'FL1-H, FL2-H, FL3-H',
                        u'Time',
                        ],
                       [u'FC002',
                        u'Moake\'s Flow Cytometer (new acquisition card)',
                        u'FSC',
                        u'SSC',
                        u'FL1, FL2, FL3',
                        u'TIME',
                        ],
                       [u'',
                        u'Fake flow cytometer',
                        u'FSC',
                        u'SSC',
                        u'FL1, FL2, FL3',
                        u'TIME',
                        ],
                       ]
        id_header = 'ID'

        # Expected output
        expected_output = collections.OrderedDict()

        row = collections.OrderedDict()
        row[u'ID'] = u'FC001'
        row[u'Description'] = u'Moake\'s Flow Cytometer'
        row[u'Forward Scatter Channel'] = u'FSC-H'
        row[u'Side Scatter Channel'] = u'SSC-H'
        row[u'Fluorescence Channels'] = u'FL1-H, FL2-H, FL3-H'
        row[u'Time Channel'] = u'Time'
        expected_output['FC001'] = row

        row = collections.OrderedDict()
        row[u'ID'] = u'FC002'
        row[u'Description'] = u'Moake\'s Flow Cytometer (new acquisition card)'
        row[u'Forward Scatter Channel'] = u'FSC'
        row[u'Side Scatter Channel'] = u'SSC'
        row[u'Fluorescence Channels'] = u'FL1, FL2, FL3'
        row[u'Time Channel'] = u'TIME'
        expected_output['FC002'] = row

        # Run list_to_table
        table = fc.excel_ui.list_to_table(input_table, id_header)

        # Compare to expected output
        self.assertEqual(table, expected_output)

    def test_list_to_table_no_id_error(self):
        """Test that excel_ui.list_to_table produces a ValueError when
        `id_header` it not in the table's header.

        """
        # Input data
        input_table = [[u'UID',
                        u'Description',
                        u'Forward Scatter Channel',
                        u'Side Scatter Channel',
                        u'Fluorescence Channels',
                        u'Time Channel',
                        ],
                       [u'FC001',
                        u'Moake\'s Flow Cytometer',
                        u'FSC-H',
                        u'SSC-H',
                        u'FL1-H, FL2-H, FL3-H',
                        u'Time',
                        ],
                       [u'FC002',
                        u'Moake\'s Flow Cytometer (new acquisition card)',
                        u'FSC',
                        u'SSC',
                        u'FL1, FL2, FL3',
                        u'TIME',
                        ],
                       ]
        id_header = 'ID'

        # Run function and check for error
        self.assertRaises(ValueError,
                          fc.excel_ui.list_to_table,
                          input_table,
                          id_header)

    def test_list_to_table_repeated_id_error(self):
        """Test that excel_ui.list_to_table produces a ValueError when
        the value of the `id_header` column is repeated in two rows.

        """
        # Input data
        input_table = [[u'ID',
                        u'Description',
                        u'Forward Scatter Channel',
                        u'Side Scatter Channel',
                        u'Fluorescence Channels',
                        u'Time Channel',
                        ],
                       [u'FC001',
                        u'Moake\'s Flow Cytometer',
                        u'FSC-H',
                        u'SSC-H',
                        u'FL1-H, FL2-H, FL3-H',
                        u'Time',
                        ],
                       [u'FC001',
                        u'Moake\'s Flow Cytometer (new acquisition card)',
                        u'FSC',
                        u'SSC',
                        u'FL1, FL2, FL3',
                        u'TIME',
                        ],
                       ]
        id_header = 'ID'

        # Run function and check for error
        self.assertRaises(ValueError,
                          fc.excel_ui.list_to_table,
                          input_table,
                          id_header)

    def test_list_to_table_variable_row_length_error(self):
        """Test that excel_ui.list_to_table produces a ValueError when
        the input table has different list lengths.

        """
        # Input data
        input_table = [[u'ID',
                        u'Description',
                        u'Forward Scatter Channel',
                        u'Side Scatter Channel',
                        u'Fluorescence Channels',
                        u'Time Channel',
                        ],
                       [u'FC001',
                        u'Moake\'s Flow Cytometer',
                        u'FSC-H',
                        u'SSC-H',
                        u'FL1-H, FL2-H, FL3-H',
                        u'Time',
                        ],
                       [u'FC001',
                        u'Moake\'s Flow Cytometer (new acquisition card)',
                        u'FSC',
                        u'SSC',
                        u'FL1, FL2, FL3',
                        u'TIME',
                        u'Additional field',
                        ],
                       ]
        id_header = 'ID'

        # Run function and check for error
        self.assertRaises(ValueError,
                          fc.excel_ui.list_to_table,
                          input_table,
                          id_header)

    def test_table_to_list(self):
        """Test that excel_ui.table_to_list produces the correct output.

        """
        # Input data
        input_table = collections.OrderedDict()

        row = collections.OrderedDict()
        row[u'ID'] = u'FC001'
        row[u'Description'] = u'Moake\'s Flow Cytometer'
        row[u'Forward Scatter Channel'] = u'FSC-H'
        row[u'Side Scatter Channel'] = u'SSC-H'
        row[u'Fluorescence Channels'] = u'FL1-H, FL2-H, FL3-H'
        row[u'Time Channel'] = u'Time'
        input_table['FC001'] = row

        row = collections.OrderedDict()
        row[u'ID'] = u'FC002'
        row[u'Description'] = u'Moake\'s Flow Cytometer (new acquisition card)'
        row[u'Forward Scatter Channel'] = u'FSC'
        row[u'Side Scatter Channel'] = u'SSC'
        row[u'Fluorescence Channels'] = u'FL1, FL2, FL3'
        row[u'Time Channel'] = u'TIME'
        input_table['FC002'] = row

        # Expected output
        expected_output = [[u'ID',
                        u'Description',
                        u'Forward Scatter Channel',
                        u'Side Scatter Channel',
                        u'Fluorescence Channels',
                        u'Time Channel',
                        ],
                       [u'FC001',
                        u'Moake\'s Flow Cytometer',
                        u'FSC-H',
                        u'SSC-H',
                        u'FL1-H, FL2-H, FL3-H',
                        u'Time',
                        ],
                       [u'FC002',
                        u'Moake\'s Flow Cytometer (new acquisition card)',
                        u'FSC',
                        u'SSC',
                        u'FL1, FL2, FL3',
                        u'TIME',
                        ],
                       ]

        # Run table_to_list
        table = fc.excel_ui.table_to_list(input_table)

        # Compare to expected output
        self.assertEqual(table, expected_output)

    def test_table_to_list_different_header_error(self):
        """Test that excel_ui.table_to_list produces the correct output.

        """
        # Input data
        input_table = collections.OrderedDict()

        row = collections.OrderedDict()
        row[u'ID'] = u'FC001'
        row[u'Description'] = u'Moake\'s Flow Cytometer'
        row[u'Forward Scatter Channel'] = u'FSC-H'
        row[u'Side Scatter Channel'] = u'SSC-H'
        row[u'Fluorescence Channels'] = u'FL1-H, FL2-H, FL3-H'
        row[u'Time Channel'] = u'Time'
        input_table['FC001'] = row

        row = collections.OrderedDict()
        row[u'ID'] = u'FC002'
        row[u'Description'] = u'Moake\'s Flow Cytometer (new acquisition card)'
        row[u'Forward Scatter Channel'] = u'FSC'
        row[u'Side Scatter Channel'] = u'SSC'
        row[u'Fluorescence Channels'] = u'FL1, FL2, FL3'
        row[u'Time'] = u'TIME'  # Different header than first row
        input_table['FC002'] = row

        # Run table_to_list
        self.assertRaises(ValueError,
                          fc.excel_ui.table_to_list,
                          input_table)

class TestLoadFCSFromTable(unittest.TestCase):
    """Class to test excel_ui.load_fcs_from_table()

    """

    def test_load_fcs_from_table(self):
        """Test that excel_ui.load_fcs_from_table produces the correct
        output.

        """

        # Input table
        table = collections.OrderedDict()

        row = collections.OrderedDict()
        row["ID"] = "S001"
        row["File Path"] = "test/Data001.fcs"
        row["Gate Fraction"] = 0.3
        table[row["ID"]] = row

        row = collections.OrderedDict()
        row["ID"] = "S002"
        row["File Path"] = "test/Data003.fcs"
        row["Gate Fraction"] = 0.5
        table[row["ID"]] = row

        filename_key = "File Path"

        # Expected output
        fcs_files_expected = []
        fcs_files_expected.append(fc.io.FCSData("test/Data001.fcs",
                                                metadata = table["S001"]))
        fcs_files_expected.append(fc.io.FCSData("test/Data003.fcs",
                                                metadata = table["S002"]))

        # Load files from table
        fcs_files = fc.excel_ui.load_fcs_from_table(table, filename_key)

        # Compare
        for fcs_file, fcs_file_expected in zip(fcs_files, fcs_files_expected):
            # FCSData contents
            np.testing.assert_array_equal(fcs_file, fcs_file_expected)
            # metadata
            self.assertEqual(fcs_file.metadata, fcs_file_expected.metadata)


if __name__ == '__main__':
    unittest.main()
