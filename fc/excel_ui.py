#!/usr/bin/python
"""Module containing the Microsoft Excel User Interface.

Authors: Brian Landry (brian.landry@rice.edu)
         Sebastian M. Castillo-Hair (smc9@rice.edu)

Last Modified: 10/26/2015

"""

def read_workbook(workbook_name):
    """Open an Excel workbook and return the content of all worksheets.

    Parameters
    ----------
    workbook_name : str
        Name of the Excel workbook file to read.

    Returns
    -------
    content : OrderedDict
        The content of the specified workbook. Each item in the OrderedDict
        represents a worksheet, in which the key is the worksheet's name
        and the value is the sheet's content. A sheet's content is, in
        turn, represented as a list of lists.
    
    """
    pass


def write_workbook(workbook_name, content):
    """Write an Excel workbook with the specified content.

    Parameters
    ----------
    workbook_name : str
        Name of the Excel workbook file to write.
    content : OrderedDict
        Content to be written to the workbook. Each item in the OrderedDict
        represents a worksheet, in which the key is the worksheet's name
        and the value is the sheet's content. A sheet's content is, in
        turn, represented as a list of lists.

    Raises
    ------
    IOError
        If access to the output file is denied (e.g. file is open in
        another program).
    
    """
    pass