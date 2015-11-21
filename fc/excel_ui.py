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
import pandas as pd

import fc.io
import fc.plot
import fc.gate
import fc.transform
import fc.stats
import fc.mef

def read_table(filename, sheetname, index_col=None):
    """
    Return the contents of an Excel table as a pandas DataFrame.

    Parameters
    ----------
    filename : str
        Name of the Excel file to read.
    sheetname : str
        Name of the sheet inside the Excel file to read.
    index_col : str, optional
        Column name or index to be used as row labels of the DataFrame. If
        None, rows will not get custom labels.

    Returns
    -------
    table : DataFrame
        A DataFrame containing the data in the specified Excel table. If
        `index_col` is not None, rows in which their `index_col` field
        is empty will not be present in `table`.

    Raises
    ------
    ValueError
        If `index_col` is specified and two rows contain the same
        `index_col` field.

    """
    # Load excel table using pandas
    table = pd.read_excel(filename,
                          sheetname=sheetname,
                          index_col=index_col)
    # Eliminate rows whose index are null
    if index_col is not None:
        table = table[pd.notnull(table.index)]
    # Check for duplicated rows
    if table.index.has_duplicates:
        raise ValueError("sheet {} on file {} contains duplicated values "
                         "for column {}".format(sheetname, filename, index_col))

    return table

def write_workbook(filename, table_list):
    """
    Write an Excel workbook from a list of tables.

    Parameters
    ----------
    filename: str
        Name of the Excel file to write.
    table_list: list of ``(str, DataFrame)`` tuples
        Tables to be saved as individual sheets in the Excel table. Each
        tuple contains two values: the name of the sheet to be saved as a
        string, and the contents of the table as a DataFrame.

    """
    # Modify default header format
    old_header_style = pd.core.format.header_style
    pd.core.format.header_style = {"font": {"bold": True}}

    # Generate output writer object
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')

    # Write tables
    for sheet_name, df in table_list:
        # Convert index names to regular columns
        df = df.reset_index()
        # Write to an Excel sheet
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        # Set column width
        writer.sheets[sheet_name].set_column(0, len(df.columns) - 1, width=15)

    # Write excel file
    writer.save()

    # Restore previous header format
    pd.core.format.header_style = old_header_style

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
    beads_table : DataFrame
        Table specifying beads samples to be processed.
    instruments_table : DataFrame
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
    mef_transform_fxns : OrderedDict
        A dictionary of MEF transformation functions, indexed by
        ``beads_table.keys()``.

    """
    # Do nothing if beads table is empty
    if beads_table.empty:
        return

    if verbose:
        print("\nProcessing beads ({} entries)...".format(len(beads_table)))

    # Check that plotting directory exist, create otherwise
    if plot and not os.path.exists(os.path.join(base_dir, plot_dir)):
        os.makedirs(os.path.join(base_dir, plot_dir))

    # Initialize output variables
    beads_samples = []
    mef_transform_fxns = collections.OrderedDict()

    # Iterate through table
    for beads_id, beads_row in beads_table.iterrows():

        ###
        # Instrument Data
        ###
        # Get the appropriate row in the instrument table
        instruments_row = instruments_table.loc[beads_row['Instrument ID']]
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
            mef_transform_fxns[beads_id] = fc.mef.get_transform_fxn(
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

    return beads_samples, mef_transform_fxns

def process_samples_table(samples_table,
                          instruments_table,
                          mef_transform_fxns=None,
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
    samples_table : DataFrame
        Table specifying samples to be processed.
    instruments_table : DataFrame
        Table specifying instruments.
    mef_transform_fxns : dict or OrderedDict, optional
        Dictionary containing MEF transformation functions. If any entry
        in `samples_table` requires transformation to MEF, a key: value
        pair must exist in mef_transform_fxns, with the key being equal to the
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
    if samples_table.empty:
        return

    if verbose:
        print("\nProcessing samples ({} entries)...".format(len(samples_table)))

    # Check that plotting directory exist, create otherwise
    if plot and not os.path.exists(os.path.join(base_dir, plot_dir)):
        os.makedirs(os.path.join(base_dir, plot_dir))

    # Load sample files
    samples = []
    for sample_id, sample_row in samples_table.iterrows():

        ###
        # Instrument Data
        ###
        # Get the appropriate row in the instrument table
        instruments_row = instruments_table.loc[sample_row['Instrument ID']]
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
                    or sample_row['{} Units'.format(fl_channel)] == ''
                    or pd.isnull(sample_row['{} Units'.format(fl_channel)])):
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
                sample = mef_transform_fxns[sample_row['Beads ID']](sample,
                                                                    fl_channel)
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
    samples_table : DataFrame
        Table specifying samples to analyze
    samples : list
        FCSData objects from which to calculate statistics. ``samples[i]``
        should correspond to ``samples_table.values()[i]``

    """
    # Add per-row stats
    samples_table['Number of Events'] = [sample.shape[0] for sample in samples]
    samples_table['Acquisition Time (s)'] = [sample.acquisition_time
                                                for sample in samples]

    # List of channels that require stats columns
    headers = list(samples_table.columns)
    r = re.compile(r'^(\S)*(\s)*Units$')
    stats_headers = [h for h in headers if r.match(h)]
    stats_channels = [s[:-5].strip() for s in stats_headers]

    # Iterate through channels
    for header, channel in zip(stats_headers, stats_channels):
        # Add empty columns to table
        samples_table[channel + ' Gain'] = np.nan
        samples_table[channel + ' Mean'] = np.nan
        samples_table[channel + ' Geom. Mean'] = np.nan
        samples_table[channel + ' Median'] = np.nan
        samples_table[channel + ' Mode'] = np.nan
        samples_table[channel + ' Std'] = np.nan
        samples_table[channel + ' CV'] = np.nan
        samples_table[channel + ' IQR'] = np.nan
        samples_table[channel + ' RCV'] = np.nan
        for row_id, sample in zip(samples_table.index, samples):
            # If units are specified, calculate stats. If not, leave empty.
            if pd.notnull(samples_table[header][row_id]):
                samples_table.set_value(row_id,
                                        channel + ' Gain',
                                        sample[:, channel].
                                        channel_info[0]['pmt_voltage'])
                samples_table.set_value(row_id,
                                        channel + ' Mean',
                                        fc.stats.mean(sample, channel))
                samples_table.set_value(row_id,
                                        channel + ' Geom. Mean',
                                        fc.stats.gmean(sample, channel))
                samples_table.set_value(row_id,
                                        channel + ' Median',
                                        fc.stats.median(sample, channel))
                samples_table.set_value(row_id,
                                        channel + ' Mode',
                                        fc.stats.mode(sample, channel))
                samples_table.set_value(row_id,
                                        channel + ' Std',
                                        fc.stats.std(sample, channel))
                samples_table.set_value(row_id,
                                        channel + ' CV',
                                        fc.stats.CV(sample, channel))
                samples_table.set_value(row_id,
                                        channel + ' IQR',
                                        fc.stats.iqr(sample, channel))
                samples_table.set_value(row_id,
                                        channel + ' RCV',
                                        fc.stats.RCV(sample, channel))

def generate_histograms_table(samples_table, samples):
    """
    Generate a table of histograms as a DataFrame.

    Parameters
    ----------
    samples_table : DataFrame
        Table specifying samples to analyze
    samples : list
        FCSData objects from which to calculate histograms. ``samples[i]``
        should correspond to ``samples_table.iloc[i]``
    
    Returns
    -------
    hist_table: DataFrame
        A multi-indexed DataFrame. Rows cotain the histogram bins and
        counts for every sample and channel specified in samples_table.
        `hist_table` is indexed by the sample's ID, the channel name,
        and whether the row corresponds to bins or counts.

    """
    # Extract channels that require stats histograms
    headers = samples_table.columns.tolist()
    r = re.compile(r'^(\S)*(\s)*Units$')
    hist_headers = [h for h in headers if r.match(h)]
    hist_channels = [s[:-5].strip() for s in hist_headers]

    # The number of columns in the DataFrame has to be set to the maximum
    # number of bins of any of the histograms about to be generated.
    # The following iterates through these histograms and finds the
    # largest.
    n_columns = 0
    for sample_id, sample in zip(samples_table.index, samples):
        for header, channel in zip(hist_headers, hist_channels):
            if pd.notnull(samples_table[header][sample_id]):
                bins = sample[:,channel].channel_info[0]['bin_vals']
                if n_columns < len(bins):
                    n_columns = len(bins)

    # Declare multi-indexed DataFrame
    index = pd.MultiIndex.from_arrays([[],[],[]],
                                      names = ['Sample ID', 'Channel', ''])
    columns = ['Bin {}'.format(i + 1) for i in range(n_columns)]
    hist_table = pd.DataFrame([], index=index, columns=columns)

    # Generate histograms
    for sample_id, sample in zip(samples_table.index, samples):
        for header, channel in zip(hist_headers, hist_channels):
            if pd.notnull(samples_table[header][sample_id]):
                # Store bins
                bins = sample[:,channel].channel_info[0]['bin_vals']
                hist_table.loc[(sample_id, channel, 'Bins'),
                               columns[0:len(bins)]] = bins
                # Calculate and store histogram counts
                bin_edges = sample[:,channel].channel_info[0]['bin_edges']
                hist, __ = np.histogram(sample[:,channel], bins=bin_edges)
                hist_table.loc[(sample_id, channel, 'Counts'),
                               columns[0:len(bins)]] = hist

    return hist_table

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

    # Read relevant tables from workbook
    instruments_table = read_table(input_path,
                                   sheetname='Instruments',
                                   index_col='ID')
    beads_table = read_table(input_path,
                             sheetname='Beads',
                             index_col='ID')
    samples_table = read_table(input_path,
                               sheetname='Samples',
                               index_col='ID')

    # Process beads samples
    beads_samples, mef_transform_fxns = process_beads_table(
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
        mef_transform_fxns=mef_transform_fxns,
        base_dir=input_dir,
        verbose=verbose,
        plot=plot,
        plot_dir='plot_samples')

    # Add stats to samples table
    add_stats(samples_table, samples)

    # # Generate histograms
    histograms_table = generate_histograms_table(samples_table, samples)

    # Generate list of tables to save
    table_list = []
    table_list.append(('Instruments', instruments_table))
    table_list.append(('Beads', beads_table))
    table_list.append(('Samples', samples_table))
    table_list.append(('Histograms', histograms_table))

    # Write output excel file
    output_filename = "{}_output.xlsx".format(input_filename_no_ext)
    output_path = os.path.join(input_dir, output_filename)
    write_workbook(output_path, table_list)

if __name__ == '__main__':
    run(verbose=True, plot=True)
