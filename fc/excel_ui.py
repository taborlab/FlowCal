"""Module containing the Microsoft Excel User Interface.

Authors: Brian Landry (brian.landry@rice.edu)
         Sebastian M. Castillo-Hair (smc9@rice.edu)

Last Modified: 10/26/2015

"""

import collections
import xlrd
import openpyxl

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
    # Declare output data structure
    content = collections.OrderedDict()

    # Load workbook
    wb = xlrd.open_workbook(workbook_name)
    # Iterate thorugh sheets
    for ws in wb.sheets():
        # Get worksheet contents
        ws_contents = [[cell.value for cell in ws.row(ir)] \
                            for ir in range(ws.nrows)]
        content[ws.name] = ws_contents

    return content

def write_workbook(workbook_name, content):
    """Write an Excel workbook with the specified content.

    If the specified workbook already exists, this function overwrites
    cells but not worksheets.

    Parameters
    ----------
    workbook_name : str
        Name of the Excel workbook file to write.
    content : dict or OrderedDict
        Content to be written to the workbook. Each item in the dictionary
        represents a worksheet, in which the key is the worksheet's name
        and the value is the sheet's content. A sheet's content is, in
        turn, represented as a list of lists. Use an OrderedDict to ensure
        that the order of the worksheets is as specified.

    Raises
    ------
    TypeError
        If the content is not a dictionary or OrderedDict.
    ValueError
        If the length of the content dictionary is zero.
    IOError
        If access to the output file is denied (e.g. file is open in
        another program).
    
    """
    # Check that content is a dictionary of OrderedDict
    if type(content) is not dict \
            and type(content) is not collections.OrderedDict:
        raise TypeError("Incorrect content type")
    if len(content) <= 0:
        raise ValueError("Worksheet content should have at least one sheet")

    # Create workbook
    wb = openpyxl.Workbook()
    # Eliminate the first, automatically created empty worksheet
    wb.remove_sheet(wb.get_active_sheet())

    # Iterate through content
    for sheet_name, sheet_content in content.items():
        # Create new worksheet
        ws = wb.create_sheet()
        ws.title = sheet_name
        # Write content to worksheet
        for r, row in enumerate(sheet_content):
            for c, value in enumerate(row):
                ws.cell(row=r + 1, column = c + 1).value = value

    # Try to save document
    try:
        wb.save(workbook_name)
    except IOError as e:
        e.message = "Error writing to {}".format(workbook_name)
        raise
