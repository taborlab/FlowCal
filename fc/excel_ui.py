"""
`FlowCal`'s' Microsoft Excel User Interface.

"""

import collections
import os
import os.path
import platform
import re
import subprocess
from Tkinter import Tk
from tkFileDialog import askopenfilename

from matplotlib import pyplot as plt
import numpy as np
import openpyxl
import xlrd

import fc.io
import fc.plot
import fc.gate
import fc.transform
import fc.stats
import fc.mef

def read_workbook(workbook_name):
    """
    Open an Excel workbook and return the content of all worksheets.

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
        ws_contents = [[cell.value for cell in ws.row(ir)]
                        for ir in range(ws.nrows)]
        content[ws.name] = ws_contents

    return content

def write_workbook(workbook_name, content):
    """
    Write an Excel workbook with the specified content.

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
    
    """
    # Check that `content` is a dictionary or OrderedDict
    if (type(content) is not dict
            and type(content) is not collections.OrderedDict):
        raise TypeError("incorrect content type")
    # Check that `content` is not empty
    if len(content) <= 0:
        raise ValueError("worksheet content should have at least one sheet")

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
        for row_idx, row in enumerate(sheet_content):
            for col_idx, value in enumerate(row):
                ws.cell(row=row_idx+1, column=col_idx+1).value = value

    # Try to save document
    try:
        wb.save(workbook_name)
    except IOError as e:
        e.message = "error writing to {}".format(workbook_name)
        raise

def list_to_table(table_list, id_header='ID'):
    """
    Convert a table from list-of-lists to OrderedDict-of-OrderedDicts.

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
            raise ValueError("all lists inside table_list should "
                "have the same length")

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
            raise ValueError("duplicated values for column {} found".
                format(id_header))
        # Add row to table
        table[row_dict[id_header]] = row_dict

    return table

def table_to_list(table):
    """
    Convert a table from dict-of-dicts to list-of-lists.

    This function accepts a table in a dict of dicts format. Each one of
    the inner dicts is interpreted as as row, in which the keys are the
    table headers, and the values are the contents of each field. `table`
    is a dictionary containing these rows. All of the rows should have the
    same headers. The table is then converted to a list of lists, in which
    the first list is the table header, and the subsequents lists are each
    one of the rows. The rows are ordered according to table.keys()

    Parameters
    ----------
    table : dict or OrderedDict
        Table data, as a dict of dicts.

    Returns
    -------
    table_list : list of lists
        The contents of the table as an list of lists.

    Raises
    ------
    ValueError
        If the keys of all rows are not the same.

    """
    # Extract table headers
    headers = table.values()[0].keys()
    # Check that all rows have the same keys
    for k, row in table.items():
        if row.keys() != headers:
            raise ValueError("all rows should have the same keys")

    # Initialize output
    table_list = [headers]

    # Extract rows
    for k, row in table.items():
        table_list.append(row.values())

    return table_list

def load_fcs_from_table(table, filename_key):
    """
    Load FCS files from a table and add table information as metadata.

    This function accepts a table formatted as an OrderedDict of
    OrderedDicts, the same format as the output of the ``list_to_table``
    function. For each row, an FCS file with filename given by
    `filename_key` is loaded as an fc.io.FCSData object, and the rows's
    fields are used as metadata.

    Parameters
    ----------
    table : dict or OrderedDict
        Table data, as a dictionary of dictionaries.
    filename_key : str
        The field containing the name of the FCS file to load on each row.

    Returns
    -------
    list
        FCSData objects corresponding to the loaded FCS files.

    """
    return [fc.io.FCSData(row[filename_key], metadata=row)
            for row_id, row in table.items()]

def process_beads_table(beads_table,
                      instruments_table,
                      base_dir="",
                      verbose=False,
                      plot=False,
                      plot_dir=""):
    """
    Load and process FCS files corresponding to beads.

    TODO: Describe format of table.

    This function processes the entries in `beads_table`. For each row, the
    function does the following:
    - Load the FCS file specified in the field "File Path".
    - Transform the forward scatter/side scatter channels if needed.
    - Remove the 250 first and 100 last events.
    - Remove saturated events in the forward scatter and side scatter
      channels.
    - Apply density gating on the forward scatter/side scatter channels.
    - Generate a standard curve transformation function, for each
      fluorescence channel in which the associated MEF values are
      specified.
    - Generate forward/side scatter density plots and fluorescence
      histograms, and plots of the clustering and fitting steps of
      standard curve generation, if `plot` = True.
    
    Names of forward/side scatter and fluorescence channels are taken from
    `instruments_table`.

    Parameters
    ----------
    beads_table : OrderedDict or dict
        Table specifying beads samples to be processed.
    instruments_table : OrderedDict or dict
        Table specifying instruments.
    base_dir : str, optional
        Directory from where all the other paths are specified from.
    verbose: bool, optional
        Whether to print information messages during the execution of this
        function.
    plot : bool, optional
        Whether to generate and save density/histogram plots of each
        sample, and each beads sample.
    plot_dir : str, optional
        The directory where to save the generated plots of beads samples,
        relative to `base_dir`. If `plot` is False, this parameter is
        ignored.

    Returns
    -------
    beads_samples : list of FCSData objects
        A list of processed, gated, and transformed samples, as specified
        in `beads_table`, in the order of ``beads_table.keys()``.
    to_mef : OrderedDict
        A dictionary of MEF transformation functions, indexed by
        ``beads_table.keys()``.

    """
    # Do nothing if beads table is empty
    if not beads_table:
        return

    if verbose:
        print("\nProcessing beads ({} entries)...".format(len(beads_table)))

    # Check that plotting directory exist, create otherwise
    if plot and not os.path.exists(os.path.join(base_dir, plot_dir)):
        os.makedirs(os.path.join(base_dir, plot_dir))

    # Initialize output variables
    beads_samples = []
    to_mef = collections.OrderedDict()

    # Iterate through table
    for beads_id, beads_row in beads_table.items():

        ###
        # Instrument Data
        ###
        # Get the appropriate row in the instrument table
        instruments_row = instruments_table[beads_row['Instrument ID']]
        # Scatter channels: Foward Scatter, Side Scatter
        sc_channels = [instruments_row['Forward Scatter Channel'],
                       instruments_row['Side Scatter Channel'],
                       ]
        # Fluorescence channels is a comma-separated list
        fl_channels = instruments_row['Fluorescence Channels'].split(',')
        fl_channels = [s.strip() for s in fl_channels]

        ###
        # Beads Data
        ###
        filename = os.path.join(base_dir, beads_row['File Path'])
        beads_sample = fc.io.FCSData(filename, metadata=beads_row)
        if verbose:
            print("{} loaded ({} events).".format(beads_id,
                                                  beads_sample.shape[0]))
        # Parse clustering channels data
        cluster_channels = beads_row['Clustering Channels'].split(',')
        cluster_channels = [cc.strip() for cc in cluster_channels]

        ###
        # Gating
        ###
        # Remove first and last events
        beads_sample = fc.gate.start_end(beads_sample,
                                         num_start=250,
                                         num_end=100)
        # Remove saturating events in forward/side scatter
        # The value of a saturating event is taken automatically from
        # `beads_sample.channel_info`.
        beads_sample = fc.gate.high_low(beads_sample,
                                        channels=sc_channels)
        # Density gating
        if verbose:
            print("Running density gate (fraction = {:.2f})...".format(
                beads_row['Gate Fraction']))
        beads_sample_gated, __, gate_contour = fc.gate.density2d(
            data=beads_sample,
            channels=sc_channels,
            gate_fraction=beads_row['Gate Fraction'],
            full_output=True)

        # Plot forward/side scatter density plot and fluorescence histograms
        if plot:
            figname = os.path.join(base_dir,
                                   plot_dir,
                                   "density_hist_{}.png".format(beads_id))
            plt.figure(figsize = (6,4))
            fc.plot.density_and_hist(
                beads_sample,
                beads_sample_gated, 
                density_channels=sc_channels,
                hist_channels=cluster_channels,
                gate_contour=gate_contour, 
                density_params={'mode': 'scatter'}, 
                hist_params={'div': 4},
                savefig=figname)

        # Save sample
        beads_samples.append(beads_sample_gated)

        # Process MEF values
        # For each channel specified in mef_channels, check whether a list of
        # MEF values is provided in the spreadsheet, and save them.
        # If there is no such list in the spreadsheet, throw error.
        mef_values = []
        mef_channels = []
        for fl_channel in fl_channels:
            if '{} MEF Values'.format(fl_channel) in beads_row:
                # Save channel name
                mef_channels.append(fl_channel)
                # Parse list of values
                mef = beads_row[fl_channel + ' MEF Values'].split(',')
                mef = [int(e) if e.strip().isdigit() else np.nan
                       for e in mef]
                mef_values.append(mef)
        mef_values = np.array(mef_values)

        # Obtain standard curve transformation
        if mef_channels:
            if verbose:
                print("\nCalculating standard curve...")
            to_mef[beads_id] = fc.mef.get_transform_fxn(
                beads_sample_gated,
                mef_values, 
                cluster_method='gmm', 
                cluster_channels=cluster_channels,
                mef_channels=mef_channels,
                select_peaks_method='proximity',
                select_peaks_params={'peaks_ch_max': 1015},
                verbose=verbose, 
                plot=plot,
                plot_filename=beads_id,
                plot_dir=os.path.join(base_dir, plot_dir))

    return beads_samples, to_mef

def process_samples_table(samples_table,
                        instruments_table,
                        to_mef=None,
                        base_dir="",
                        verbose=False,
                        plot=False,
                        plot_dir=""):
    """
    Load and process FCS files corresponding to samples.

    The function processes each entry in `samples_table`, and does the
    following:
    - Load the FCS file specified in the field "File Path".
    - Transform the forward scatter/side scatter channels if needed.
    - Remove the 250 first and 100 last events.
    - Remove saturated events in the forward scatter and side scatter
      channels.
    - Apply density gating on the forward scatter/side scatter channels.
    - Transform the fluorescence channels to the units specified in the
      column "<Channel name> Units".
    - Plot combined forward/side scatter density plots and fluorescence
      historgrams, if `plot` = True.
    
    Names of forward/side scatter and fluorescence channels are taken from
    `instruments_table`.

    Parameters
    ----------
    samples_table : OrderedDict or dict
        Table specifying samples to be processed.
    instruments_table : OrderedDict or dict
        Table specifying instruments.
    to_mef : dict or OrderedDict, optional
        Dictionary containing MEF transformation functions. If any entry
        in `samples_table` requires transformation to MEF, a key: value
        pair must exist in to_mef, with the key being equal to the
        contents of field "Beads ID".
    base_dir : str, optional
        Directory from where all the other paths are specified from.
    verbose: bool, optional
        Whether to print information messages during the execution of this
        function.
    plot : bool, optional
        Whether to generate and save density/histogram plots of each
        sample, and each beads sample.
    plot_dir : str, optional
        The directory where to save the generated plots of beads samples,
        relative to `base_dir`. If `plot` is False, this parameter is
        ignored.

    Returns
    -------
    samples : list of FCSData objects
        A list of processed, gated, and transformed samples, as specified
        in `samples_table`, in the order of ``samples_table.keys()``.

    """
    # Do nothing if samples table is empty
    if not samples_table:
        return

    if verbose:
        print("\nProcessing samples ({} entries)...".format(len(samples_table)))

    # Check that plotting directory exist, create otherwise
    if plot and not os.path.exists(os.path.join(base_dir, plot_dir)):
        os.makedirs(os.path.join(base_dir, plot_dir))

    # Load sample files
    samples = []
    for sample_id, sample_row in samples_table.items():

        ###
        # Instrument Data
        ###
        # Get the appropriate row in the instrument table
        instruments_row = instruments_table[sample_row['Instrument ID']]
        # Scatter channels: Foward Scatter, Side Scatter
        sc_channels = [instruments_row['Forward Scatter Channel'],
                       instruments_row['Side Scatter Channel'],
                       ]
        # Fluorescence channels is a comma-separated list
        fl_channels = instruments_row['Fluorescence Channels'].split(',')
        fl_channels = [s.strip() for s in fl_channels]

        ###
        # Sample Data
        ###
        filename = os.path.join(base_dir, sample_row['File Path'])
        sample = fc.io.FCSData(filename, metadata=sample_row)
        if verbose:
            print("{} loaded ({} events).".format(sample_id,
                                                  sample.shape[0]))

        ###
        # Transform
        ###
        if verbose:
            print("Performing data transformation...")
        # Transform FSC/SSC to relative units
        sample = fc.transform.exponentiate(sample, sc_channels)

        # Parse fluorescence channels in which to transform
        report_channels = []
        report_units = []
        for fl_channel in fl_channels:
            # Check whether there is a column with units for fl_channel
            if ("{} Units".format(fl_channel) not in sample_row
                    or sample_row['{} Units'.format(fl_channel)] == ''):
                continue

            # Decide what transformation to perform
            units = sample_row['{} Units'.format(fl_channel)].strip()
            if units == 'Channel':
                units_label = "Channel Number"
            elif units == 'RFI':
                units_label = "Relative Fluorescence Intensity, RFI"
                sample = fc.transform.exponentiate(sample, fl_channel)
            elif units == 'MEF':
                units_label = "Molecules of Equivalent Fluorophore, MEF"
                to_mef_sample = to_mef[sample_row['Beads ID']]
                sample = to_mef_sample(sample, fl_channel)
            else:
                raise ValueError("units {} not recognized for sample {}".
                    format(units, sample_id))

            # Register that reporting in this channel must be done
            report_channels.append(fl_channel)
            report_units.append(units_label)

        ###
        # Gate
        ###
        if verbose:
            print("Performing gating...")
        # Remove first and last events
        sample_gated = fc.gate.start_end(sample, num_start=250, num_end=100)
        # Remove saturating events in forward/side scatter
        # The value of a saturating event is taken automatically from
        # `sample_gated.channel_info`.
        sample_gated = fc.gate.high_low(sample_gated, sc_channels)
        # Remove saturating events in channels to report
        sample_gated = fc.gate.high_low(sample_gated, report_channels)
        # Density gating
        sample_gated, __, gate_contour = fc.gate.density2d(
            data=sample_gated,
            channels=sc_channels,
            gate_fraction=sample_row['Gate Fraction'],
            full_output=True)

        # Accumulate
        samples.append(sample_gated)

        # Plot forward/side scatter density plot and fluorescence histograms
        if plot:
            if verbose:
                print("Plotting density plot and histogram...")
            # Define density plot parameters
            density_params = {}
            density_params['mode'] = 'scatter'
            density_params['log'] = True
            # Define histogram plot parameters
            hist_params = []
            for rc, ru in zip(report_channels, report_units):
                param = {}
                param['div'] = 4
                param['xlabel'] = '{} ({})'.format(rc, ru)
                param['log'] = ru != 'Channel Number'
                hist_params.append(param)
                
            # Plot
            figname = os.path.join(base_dir,
                                   plot_dir,
                                   "{}.png".format(sample_id))
            fc.plot.density_and_hist(
                sample,
                sample_gated,
                gate_contour=gate_contour,
                density_channels=sc_channels,
                density_params=density_params,
                hist_channels=report_channels,
                hist_params=hist_params,
                savefig=figname)

    return samples

def add_stats(samples_table, samples):
    """
    Add stats fields to samples table.

    The following numbers are added to each row:
    - Number of Events
    - Acquisition Time (s)
    
    The following stats are added for each row, for each channel in which
    fluorescence units have been specified:
    - Gain
    - Mean
    - Geometric Mean
    - Media
    - Mode
    - Standard Deviation
    - Coefficient of Variation (CV)
    - Inter-Quartile Range
    - Robust Coefficient of Variation (RCV)

    Parameters
    ----------
    samples_table : dict or OrderedDict
        Table specifying samples to analyze
    samples : list
        FCSData objects from which to calculate statistics. ``samples[i]``
        should correspond to ``samples_table.values()[i]``

    """
    # Add per-row stats
    for row, sample in zip(samples_table.values(), samples):
        row['Number of Events'] = sample.shape[0]
        row['Acquisition Time (s)'] = sample.acquisition_time

    # List of channels that require stats columns
    headers = samples_table.values()[0].keys()
    r = re.compile(r'^(\S)*(\s)*Units$')
    stats_headers = [h for h in headers if r.match(h)]
    stats_channels = [s[:-5].strip() for s in stats_headers]

    # Iterate through channels
    for header, channel in zip(stats_headers, stats_channels):
        for row, sample in zip(samples_table.values(), samples):
            # If units are specified, calculate stats. If not, leave empty.
            if row[header]:
                row[channel + ' Gain'] = \
                            sample[:, channel].channel_info[0]['pmt_voltage']
                row[channel + ' Mean'] = fc.stats.mean(sample, channel)
                row[channel + ' Geom. Mean'] = fc.stats.gmean(sample, channel)
                row[channel + ' Median'] = fc.stats.median(sample, channel)
                row[channel + ' Mode'] = fc.stats.mode(sample, channel)
                row[channel + ' Std'] = fc.stats.std(sample, channel)
                row[channel + ' CV'] = fc.stats.CV(sample, channel)
                row[channel + ' IQR'] = fc.stats.iqr(sample, channel)
                row[channel + ' RCV'] = fc.stats.RCV(sample, channel)
            else:
                row[channel + ' Gain'] = ''
                row[channel + ' Mean'] = ''
                row[channel + ' Geom. Mean'] = ''
                row[channel + ' Median'] = ''
                row[channel + ' Mode'] = ''
                row[channel + ' Std'] = ''
                row[channel + ' CV'] = ''
                row[channel + ' IQR'] = ''
                row[channel + ' RCV'] = ''

def generate_histograms_lists(samples_table, samples):
    """
    Generate a list of the histograms for each specified channel.
    
    Histogram information is generated with the following specifications:
    -  The first row contains the headers 'ID' and 'Channel' in cells 1
       and 2
    -  The following rows contain the following in order:
    1. sample_id
    2. channel
    3. The type of data in the row: 'Bins' or 'Counts'
    4. A list of all of the bin or count values associated with the row
    

    Parameters
    ----------
    samples_table : dict or OrderedDict
        Table specifying samples to analyze
    samples : list
        FCSData objects from which to calculate histograms. ``samples[i]``
        should correspond to ``samples_table.values()[i]``
    
    Returns
    -------
    rows: list-of-lists
        A list of lists where the top levels represents individual rows and
        the second level represents cell values in that row

    """
    # List of channels that require stats histograms
    headers = samples_table.values()[0].keys()
    r = re.compile(r'^(\S)*(\s)*Units$')
    hist_headers = [h for h in headers if r.match(h)]
    hist_channels = [s[:-5].strip() for s in hist_headers]

    rows = []
    rows.append(['ID','Channel'])

    for sample_id, sample_row, sample \
            in zip(samples_table.keys(), samples_table.values(), samples):
        for header, channel in zip(hist_headers, hist_channels):
            if sample_row[header]:
                info = sample[:,channel].channel_info[0]

                bins_row = [sample_id, channel, 'Bins']
                bins_row.extend(info['bin_vals'])
                rows.append(bins_row)

                val_row = [sample_id, channel, 'Counts']
                counts, bins = np.histogram(sample[:,channel],
                                            bins=info['bin_edges'])
                val_row.extend(counts)
                rows.append(val_row)

    return rows

def show_open_file_dialog(filetypes):
    """
    Show an open file dialog and return the path of the file selected.

    Parameters
    ----------
    filetypes : list of tuples
        Types of file to show on the dialog. Each tuple on the list must
        have two elements associated with a filetype: the first element is
        a description, and the second is the associated extension.

    Returns
    -------
    filename: str
        The path of the filename selected, or an empty string if no file
        was chosen.

    """
    # The following line is used to Tk's main window is not shown
    Tk().withdraw()

    # OSX ONLY: Call bash script to prevent file select window from sticking
    # after use.
    if platform.system() == 'Darwin':
        subprocess.call("defaults write org.python.python " +
                "ApplePersistenceIgnoreState YES", shell=True)
        filename = askopenfilename(filetypes=filetypes)
        subprocess.call("defaults write org.python.python " +
                "ApplePersistenceIgnoreState NO", shell=True)
    else:
        filename = askopenfilename(filetypes=filetypes)

    return filename

def run(verbose=True, plot=True):
    """
    Run the MS Excel User Interface.

    This function shows a dialog to open an input Excel workbook, loads FCS
    files and processes them as specified in the spreadsheet, and
    generates plots and an output workbook with statistics for each sample.

    Parameters
    ----------
    verbose: bool, optional
        Whether to print information messages during the execution of this
        function.
    plot : bool, optional
        Whether to generate and save density/histogram plots of each
        sample, and each beads sample.

    """
    # Open input workbook
    input_path = show_open_file_dialog(filetypes=[('Excel files', '*.xlsx')])
    if not input_path:
        return
    # Extract directory, filename, and filename with no extension from path
    input_dir, input_filename = os.path.split(input_path)
    input_filename_no_ext, __ = os.path.splitext(input_filename)

    # Read workbook and extract relevant tables
    wb_content = read_workbook(input_path)
    instruments_table = list_to_table(wb_content['Instruments'])
    beads_table = list_to_table(wb_content['Beads'])
    samples_table = list_to_table(wb_content['Samples'])

    # Process beads samples
    beads_samples, to_mef = process_beads_table(
        beads_table,
        instruments_table,
        base_dir=input_dir,
        verbose=verbose,
        plot=plot,
        plot_dir='plot_beads')

    # Process samples
    samples = process_samples_table(
        samples_table,
        instruments_table,
        to_mef=to_mef,
        base_dir=input_dir,
        verbose=verbose,
        plot=plot,
        plot_dir='plot_samples')

    # Add stats to samples table
    add_stats(samples_table, samples)

    # Generate histograms
    histograms = generate_histograms_lists(samples_table, samples)

    # Generate output workbook object
    output_wb = collections.OrderedDict()
    output_wb['Instruments'] = table_to_list(instruments_table)
    output_wb['Beads'] = table_to_list(beads_table)
    output_wb['Samples'] = table_to_list(samples_table)
    output_wb['Histograms'] = histograms

    # Write output excel file
    output_filename = "{}_output.xlsx".format(input_filename_no_ext)
    output_path = os.path.join(input_dir, output_filename)
    write_workbook(output_path, output_wb)

if __name__ == '__main__':
    run(verbose=True, plot=True)