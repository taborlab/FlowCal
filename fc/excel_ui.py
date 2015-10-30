"""Module containing the Microsoft Excel User Interface.

Authors: Brian Landry (brian.landry@rice.edu)
         Sebastian M. Castillo-Hair (smc9@rice.edu)

Last Modified: 10/29/2015

"""

import collections
import xlrd
import openpyxl

import fc

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
    ValueError
        If the length of `content` is zero.
    IOError
        If access to the output file is denied (e.g. file is open in
        another program).
    
    """
    # Check that `content` is a dictionary or OrderedDict
    if type(content) is not dict \
            and type(content) is not collections.OrderedDict:
        raise TypeError("Incorrect content type")
    # Check that `content` is not empty
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

def read_table(table_list, id_header = 'ID'):
    """Read a table as a list of lists and return it as a OrderedDict of
    OrderedDicts.

    This function accepts a table in a list-of-lists format. All lists
    should be the same length. The first list contains the table headers.
    `id_header` should be one of the headers. The following lists contain
    the table data rows. Each of these rows is converted to an OrderedDict
    in which the keys are the table headers, and the values are the
    contents of each field. These OrderedDict rows are the values of an
    outer OrderedDict, in which the key is the value of the `id_header`
    field. No two rows should have the same `id_header`. Rows in which
    `id_header` evaluate to False are ignored.

    Parameters
    ----------
    table_list : list of lists
        Table data, as a list of lists.
    id_header : str
        The name of the field used as key in the outer OrderedDict.

    Returns
    -------
    table : OrderedDict
        The contents of the table as an OrderdDict of OrderedDicts.

    Raises
    ------
    ValueError
        If the length of all lists inside `table_list` is not the same.
    ValueError
        If `id_header` is not in the header row.
    ValueError
        If two rows have the same `id_header` value.

    """
    # Check length of internal lists
    n_headers = len(table_list[0])
    for r in range(1, len(table_list)):
        if len(table_list[r]) != n_headers:
            raise ValueError("All lists inside table_list should \
                have the same length")

    # Extract headers
    headers = table_list[0]
    # Check that id_header is in headers
    if id_header not in headers:
        raise ValueError("id_header should be in the first row of table_list")

    # Initialize table
    table = collections.OrderedDict()

    # Iterate over rows
    for r in range(1, len(table_list)):
        row = table_list[r]
        # Initialize row in OrderedDict format
        row_dict = collections.OrderedDict()
        # Populate row
        for value, header in zip(row, headers):
            row_dict[header] = value
        # Check if id is empty
        if not row_dict[id_header]:
            continue
        # Raise error if id already exists in table
        if row_dict[id_header] in table:
            raise ValueError("Duplicated values for column {} found".
                format(id_header))
        # Add row to table
        table[row_dict[id_header]] = row_dict

    return table

def load_fcs_from_table(table, filename_key):
    """Load FCS files from a table, and add table information as metadata.

    This function accepts a table formatted in the same way as the output
    of the ``read_table`` function. For each row, an FCS file with filename
    given by `filename_key` is loaded as an fc.io.FCSData object, and the
    rows's fields are used as metadata.

    Parameters
    ----------
    table : dict or OrderedDict
        Table data, as a dictionary of dictionaries.
    filename_key : str
        The field containing the name of the FCS file to load on each row.

    Returns
    -------
    list
        List of FCSData objects corresponding to the loaded FCS files.

    """
    return [fc.io.FCSData(row[filename_key], metadata = row) \
                for row_id, row in table.items()]