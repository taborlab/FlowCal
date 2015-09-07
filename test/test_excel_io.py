#!/usr/bin/python
#
# test_excel_io.py - Unit tests for excel_io module
#
# Author: Sebastian M. Castillo-Hair (smc9@rice.edu)
# Date: 7/1/2015
#
# Requires:
#   * fc.excel_io
#   * numpy
#

import os
from collections import OrderedDict

import fc.excel_io
import unittest

def is_equal(tc, r, re):
    ''' Check if two OrderedDict are equal.

    Arguments:

    tc  - TestCase class
    r   - OrderedDict 1
    re  - OrderedDict 1
    '''
    # Check length
    tc.assertEqual(len(r), len(re))

    # All keys should be the same
    keys = re[0].keys()
    for i in range(len(r)):
        tc.assertEqual(r[i].keys(), keys)
    for i in range(len(r)):
        tc.assertEqual(re[i].keys(), keys)

    # Check items
    for i in range(len(r)):
        for k in keys:
            try:
                tc.assertEqual(r[i][k], re[i][k])
            except AssertionError as e:
                print("Error in row {}, column {}.".format(i + 1, k))
                raise e

class TestImport(unittest.TestCase):
    def setUp(self):
        # File name
        self.filename = 'test/test_excel_import.xlsx'

        # Build expected data
        self.re = []
        self.re.append(OrderedDict([(u'File Path', u'cells/data.001'),
            (u'Beads File Path', u'beads/data.001'),
            (u'Beads Peaks', u'792, 2079, 6588 ,16471, 47497, 137049, 371647'),
            (u'Gate Fraction', 0.2),
            (u'Sigma', 20),
            (u'ATC', 0),
            (u'Kinase', False),
            ]))
        self.re.append(OrderedDict([(u'File Path', u'cells/data.002'),
            (u'Beads File Path', u'beads/data.001'),
            (u'Beads Peaks', u'792, 2079, 6588 ,16471, 47497, 137049, 371647'),
            (u'Gate Fraction', 0.2),
            (u'Sigma', 20),
            (u'ATC', 0.1),
            (u'Kinase', False),
            ]))
        self.re.append(OrderedDict([(u'File Path', u'cells/data.003'),
            (u'Beads File Path', ''),
            (u'Beads Peaks', ''),
            (u'Gate Fraction', 0.1),
            (u'Sigma', 20),
            (u'ATC', 0.5),
            (u'Kinase', True),
            ]))
        self.re.append(OrderedDict([(u'File Path', u'cells/data.004'),
            (u'Beads File Path', ''),
            (u'Beads Peaks', ''),
            (u'Gate Fraction', 0.1),
            (u'Sigma', 20),
            (u'ATC', 1),
            (u'Kinase', True),
            ]))
        self.re.append(OrderedDict([(u'File Path', u'cells/data.005'),
            (u'Beads File Path', ''),
            (u'Beads Peaks', ''),
            (u'Gate Fraction', ''),
            (u'Sigma', ''),
            (u'ATC', 100),
            (u'Kinase', ''),
            ]))
        self.re.append(OrderedDict([(u'File Path', u'cells/data.001'),
            (u'Beads File Path', ''),
            (u'Beads Peaks', ''),
            (u'Gate Fraction', 0.2),
            (u'Sigma', 20),
            (u'ATC', 0),
            (u'Kinase', False),
            ]))
        self.re.append(OrderedDict([(u'File Path', u'cells/data.002'),
            (u'Beads File Path', ''),
            (u'Beads Peaks', ''),
            (u'Gate Fraction', 0.2),
            (u'Sigma', 20),
            (u'ATC', ''),
            (u'Kinase', ''),
            ]))

    def test_import_rows(self):
        '''
        Testing proper loading of rows from excel file.
        '''
        # Load data
        r = fc.excel_io.import_rows(self.filename, 'cells')

        # Check equality
        is_equal(self, r, self.re)
                
    def test_import_rows_error(self):
        '''
        Testing error when loading a non-existing sheet
        '''
        self.assertRaises(IOError, fc.excel_io.import_rows, self.filename, 
            'xxx')

class TestExport(unittest.TestCase):
    def setUp(self):
        # Name of excel file to export to.
        self.filename = 'test/test_excel_export.xlsx'
        # Data to export
        self.d = {}
        self.d['sheet_1'] = [['row1', 'row2'], [1,2],[3,5]]
        self.d['sheet 2'] = [['abcd', 'efg', 'hijkl'], [0,1,2], 
                                    [1,4,9], [27, 8, 1]]
        # Expected data for sheet_1
        self.re1 = []
        self.re1.append(OrderedDict([(u'row1', 1),
            (u'row2', 2)]))
        self.re1.append(OrderedDict([(u'row1', 3),
            (u'row2', 5)]))
        # Expected data for sheet 2
        self.re2 = []
        self.re2.append(OrderedDict([(u'abcd', 0),
            (u'efg', 1),
            (u'hijkl', 2),]))
        self.re2.append(OrderedDict([(u'abcd', 1),
            (u'efg', 4),
            (u'hijkl', 9),]))
        self.re2.append(OrderedDict([(u'abcd', 27),
            (u'efg', 8),
            (u'hijkl', 1),]))

    def tearDown(self):
        # Delete excel file
        if os.path.isfile(self.filename):
            os.remove(self.filename)

    def test_export(self):
        '''
        Testing proper exporting of excel data
        '''
        # Export
        fc.excel_io.export_workbook(self.filename, self.d)

        # Import sheet 1 and test contents
        r1 = fc.excel_io.import_rows(self.filename, 'sheet_1')
        is_equal(self, r1, self.re1)

        # Import sheet 2 and test contents
        r2 = fc.excel_io.import_rows(self.filename, 'sheet 2')
        is_equal(self, r2, self.re2)
        
if __name__ == '__main__':
    unittest.main()
