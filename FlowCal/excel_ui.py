"""
`FlowCal`'s Microsoft Excel User Interface.

This module contains functions to read, gate, and transform data from a set
of FCS files, according to an input Microsoft Excel file. This file should
contain the following tables:
    - "Instruments" table (TODO: describe fields of the table)
    - "Beads" table (TODO: describe fields of the table)
    - "Samples" table (TODO: describe fields of the table)

"""

import collections
import os
import os.path
import platform
import re
import subprocess
import time
from Tkinter import Tk
from tkFileDialog import askopenfilename

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import FlowCal.io
import FlowCal.plot
import FlowCal.gate
import FlowCal.transform
import FlowCal.stats
import FlowCal.mef

# Regular expressions for headers that specify some fluorescence channel
re_mef_values = re.compile(r'^\s*(\S*)\s*MEF\s*Values\s*$')
re_units = re.compile(r'^\s*(\S*)\s*Units\s*$')

class ExcelUIException(Exception):
    """
    FlowCal Excel UI Error.

    """
    pass

def read_table(filename, sheetname, index_col=None):
    """
    Return the contents of an Excel table as a pandas DataFrame.

    Parameters
    ----------
    filename : str
        Name of the Excel file to read.
    sheetname : str or int
        Name or index of the sheet inside the Excel file to read.
    index_col : str, optional
        Column name or index to be used as row labels of the DataFrame. If
        None, default index will be used.

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
    # Catch sheetname as list or None
    if sheetname is None or hasattr(sheetname, '__iter__'):
        raise TypeError("sheetname should specify a single sheet")

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
    # Pandas' default header format is bold text with thin borders. Here we
    # use bold text only, without borders.
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
                        plot_dir="",
                        full_output=False):
    """
    Load and process FCS files corresponding to beads.

    This function processes the entries in `beads_table`. For each row, the
    function does the following:
        - Load the FCS file specified in the field "File Path".
        - Transform the forward scatter/side scatter channels if needed.
        - Remove the 250 first and 100 last events.
        - Remove saturated events in the forward scatter and side scatter
          channels.
        - Apply density gating on the forward scatter/side scatter
          channels.
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
        Table specifying beads samples to be processed. For more
        information about the fields required in this table, please consult
        the module's documentation.
    instruments_table : DataFrame
        Table specifying instruments. For more information about the fields
        required in this table, please consult the module's documentation.
    base_dir : str, optional
        Directory from where all the other paths are specified.
    verbose: bool, optional
        Whether to print information messages during the execution of this
        function.
    plot : bool, optional
        Whether to generate and save density/histogram plots of each
        sample, and each beads sample.
    plot_dir : str, optional
        Directory relative to `base_dir` into which plots are saved. If
        `plot` is False, this parameter is ignored.
    full_output : bool, optional
        Flag indicating whether to include an additional output, containing
        intermediate results from the generation of the MEF transformation
        functions.

    Returns
    -------
    beads_samples : list of FCSData objects
        A list of processed, gated, and transformed samples, as specified
        in `beads_table`, in the order of ``beads_table.index``.
    mef_transform_fxns : OrderedDict
        A dictionary of MEF transformation functions, indexed by
        ``beads_table.index``.
    mef_outputs : list
        A list with intermediate results of the generation of the MEF
        transformation functions, indexed by ``beads_table.index``. Only
        included if `full_output` is True.

    """
    # Initialize output variables
    beads_samples = []
    mef_transform_fxns = collections.OrderedDict()
    mef_outputs = []

    # Return empty structures if beads table is empty
    if beads_table.empty:
        if full_output:
            return beads_samples, mef_transform_fxns, mef_outputs
        else:
            return beads_samples, mef_transform_fxns

    if verbose:
        msg = "Processing Beads table ({} entries)".format(len(beads_table))
        print("")
        print(msg)
        print("="*len(msg))

    # Check that plotting directory exist, create otherwise
    if plot and not os.path.exists(os.path.join(base_dir, plot_dir)):
        os.makedirs(os.path.join(base_dir, plot_dir))

    # Extract header and channel names for which MEF values are specified.
    headers = list(beads_table.columns)
    mef_headers_all = [h for h in headers if re_mef_values.match(h)]
    mef_channels_all = [re_mef_values.search(h).group(1)
                        for h in mef_headers_all]

    # Iterate through table
    # We will look for a ExcelUIException on each iteration. If an exception
    # is caught, it will be stored in beads_samples.
    for beads_id, beads_row in beads_table.iterrows():
        try:
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
            if verbose:
                print("\nBeads ID {}...".format(beads_id))
                print("Loading file \"{}\"...".format(beads_row['File Path']))

            # Attempt to open file
            filename = os.path.join(base_dir, beads_row['File Path'])
            try:
                beads_sample = FlowCal.io.FCSData(filename)
            except IOError:
                raise ExcelUIException("file \"{}\" not found".format(
                    beads_row['File Path']))
            # Check that the number of events is greater than 400
            if beads_sample.shape[0] < 400:
                raise ExcelUIException("number of events is lower than 400")

            ###
            # Transform
            ###
            if verbose:
                print("Performing data transformation...")
            # Transform FSC/SSC to linear scale
            beads_sample = FlowCal.transform.to_rfi(beads_sample, sc_channels)

            # Parse clustering channels data
            cluster_channels = beads_row['Clustering Channels'].split(',')
            cluster_channels = [cc.strip() for cc in cluster_channels]

            ###
            # Gate
            ###
            if verbose:
                print("Performing gating...")
            # Remove first and last events. Transients in fluidics can make the
            # first few and last events slightly different from the rest.
            beads_sample_gated = FlowCal.gate.start_end(beads_sample,
                                                        num_start=250,
                                                        num_end=100)
            # Remove saturating events in forward/side scatter. The value of a
            # saturating event is taken automatically from
            # `beads_sample_gated.domain`.
            beads_sample_gated = FlowCal.gate.high_low(beads_sample_gated,
                                                       channels=sc_channels)
            # Density gating
            beads_sample_gated, __, gate_contour = FlowCal.gate.density2d(
                data=beads_sample_gated,
                channels=sc_channels,
                gate_fraction=beads_row['Gate Fraction'],
                full_output=True)

            # Plot forward/side scatter density plot and fluorescence histograms
            if plot:
                if verbose:
                    print("Plotting density plot and histogram...")
                # Define density plot parameters
                density_params = {}
                density_params['mode'] = 'scatter'
                density_params['xlog'] = bool(
                    beads_sample_gated.amplification_type(sc_channels[0])[0])
                density_params['ylog'] = bool(
                    beads_sample_gated.amplification_type(sc_channels[1])[0])
                density_params["title"] = "{} ({:.1f}% retained)".format(
                    beads_id,
                    beads_sample_gated.shape[0] * 100. / beads_sample.shape[0])
                # Plot
                figname = os.path.join(base_dir,
                                       plot_dir,
                                       "density_hist_{}.png".format(beads_id))
                plt.figure(figsize = (6,4))
                FlowCal.plot.density_and_hist(
                    beads_sample,
                    beads_sample_gated,
                    density_channels=sc_channels,
                    hist_channels=cluster_channels,
                    gate_contour=gate_contour,
                    density_params=density_params,
                    hist_params={'div': 4},
                    savefig=figname)

            ###
            # Process MEF values
            ###
            # For each fluorescence channel, check whether a list of known MEF
            # values of the bead subpopulations is provided in `beads_row`. This
            # involves checking that a column named "[channel] MEF Values"
            # exists and is not empty. If so, store the name of the channel in
            # `mef_channels`, and the specified MEF values in `mef_values`.
            ###
            mef_values = []
            mef_channels = []
            for fl_channel in fl_channels:
                if fl_channel in mef_channels_all:
                    # Get header from channel name
                    mef_header = \
                        mef_headers_all[mef_channels_all.index(fl_channel)]
                    # Extract text. If empty, ignore.
                    mef_str = beads_row[mef_header]
                    if pd.isnull(mef_str):
                        continue
                    # Save channel name
                    mef_channels.append(fl_channel)
                    # Parse list of values
                    mef = mef_str.split(',')
                    mef = [int(e) if e.strip().isdigit() else np.nan
                           for e in mef]
                    mef_values.append(mef)
            mef_values = np.array(mef_values)

            # Obtain standard curve transformation
            if mef_channels:
                if verbose:
                    if len(mef_channels) == 1:
                        print("Calculating standard curve for channel {}..." \
                            .format(mef_channels[0]))
                    else:
                        print("Calculating standard curve for channels {}..." \
                            .format(", ".join(mef_channels)))

                mef_output = FlowCal.mef.get_transform_fxn(
                    beads_sample_gated,
                    mef_values,
                    mef_channels=mef_channels,
                    clustering_channels=cluster_channels,
                    verbose=False,
                    plot=plot,
                    plot_filename=beads_id,
                    plot_dir=os.path.join(base_dir, plot_dir),
                    full_output=full_output)

                if full_output:
                    mef_transform_fxn = mef_output.transform_fxn
                else:
                    mef_transform_fxn = mef_output

            else:
                mef_transform_fxn = None
                mef_output = None

        except ExcelUIException as e:
            # Print Exception message
            if verbose:
                print("ERROR: {}".format(str(e)))
            # Append exception to beads_samples array, and None to everything
            # else
            beads_samples.append(e)
            mef_transform_fxns[beads_id] = None
            if full_output:
                mef_outputs.append(None)

        else:
            # If no errors were found, store results
            beads_samples.append(beads_sample_gated)
            mef_transform_fxns[beads_id] = mef_transform_fxn
            if full_output:
                mef_outputs.append(mef_output)


    if full_output:
        return beads_samples, mef_transform_fxns, mef_outputs
    else:
        return beads_samples, mef_transform_fxns

def process_samples_table(samples_table,
                          instruments_table,
                          mef_transform_fxns=None,
                          beads_table=None,
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
        - Apply density gating on the forward scatter/side scatter
          channels.
        - Transform the fluorescence channels to the units specified in the
          column "<Channel name> Units".
        - Plot combined forward/side scatter density plots and fluorescence
          historgrams, if `plot` = True.
    
    Names of forward/side scatter and fluorescence channels are taken from
    `instruments_table`.

    Parameters
    ----------
    samples_table : DataFrame
        Table specifying samples to be processed. For more information
        about the fields required in this table, please consult the
        module's documentation.
    instruments_table : DataFrame
        Table specifying instruments. For more information about the fields
        required in this table, please consult the module's documentation.
    mef_transform_fxns : dict or OrderedDict, optional
        Dictionary containing MEF transformation functions. If any entry
        in `samples_table` requires transformation to MEF, a key: value
        pair must exist in mef_transform_fxns, with the key being equal to
        the contents of field "Beads ID".
    beads_table : DataFrame, optional
        Table specifying beads samples used to generate
        `mef_transform_fxns`. This is used to check if a beads sample was
        taken at the same acquisition settings as a sample to be
        transformed to MEF. For any beads sample and channel for which a
        MEF transformation function has been generated, the following
        fields should be populated: ``<channel> Amp. Type`` and
        ``<channel> Detector Volt``. If `beads_table` is not specified, no
        checking will be performed.
    base_dir : str, optional
        Directory from where all the other paths are specified.
    verbose: bool, optional
        Whether to print information messages during the execution of this
        function.
    plot : bool, optional
        Whether to generate and save density/histogram plots of each
        sample, and each beads sample.
    plot_dir : str, optional
        Directory relative to `base_dir` into which plots are saved. If
        `plot` is False, this parameter is ignored.

    Returns
    -------
    samples : list of FCSData objects
        A list of processed, gated, and transformed samples, as specified
        in `samples_table`, in the order of ``samples_table.index``.

    """
    # Initialize output variable
    samples = []

    # Return empty list if samples table is empty
    if samples_table.empty:
        return samples

    if verbose:
        msg = "Processing Samples table ({} entries)".format(len(samples_table))
        print("")
        print(msg)
        print("="*len(msg))

    # Check that plotting directory exist, create otherwise
    if plot and not os.path.exists(os.path.join(base_dir, plot_dir)):
        os.makedirs(os.path.join(base_dir, plot_dir))

    # Extract header and channel names for which units are specified.
    headers = list(samples_table.columns)
    report_headers_all = [h for h in headers if re_units.match(h)]
    report_channels_all = [re_units.search(h).group(1)
                           for h in report_headers_all]

    # Iterate through table
    # We will look for a ExcelUIException on each iteration. If an exception
    # is caught, it will be stored in beads_samples.
    for sample_id, sample_row in samples_table.iterrows():
        try:
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
            if verbose:
                print("\nSample ID {}...".format(sample_id))
                print("Loading file \"{}\"...".format(sample_row['File Path']))

            # Attempt to open file
            filename = os.path.join(base_dir, sample_row['File Path'])
            try:
                sample = FlowCal.io.FCSData(filename)
            except IOError:
                raise ExcelUIException("file \"{}\" not found".format(
                    sample_row['File Path']))
            # Check that the number of events is greater than 400
            if sample.shape[0] < 400:
                raise ExcelUIException("number of events is lower than 400")

            ###
            # Transform
            ###
            if verbose:
                print("Performing data transformation...")
            # Transform FSC/SSC to linear scale
            sample = FlowCal.transform.to_rfi(sample, sc_channels)

            # Parse fluorescence channels in which to transform
            report_channels = []
            report_units = []
            for fl_channel in fl_channels:
                if fl_channel in report_channels_all:
                    # Get header from channel name
                    report_header = report_headers_all[
                        report_channels_all.index(fl_channel)]
                    # Extract text. If empty, ignore.
                    units_str = sample_row[report_header]
                    if pd.isnull(units_str):
                        continue
                    # Decide what transformation to perform
                    units = units_str.strip()
                    if units.lower() == 'channel':
                        units_label = "Channel Number"

                    elif units.lower() == 'rfi':
                        units_label = "Relative Fluorescence Intensity, RFI"
                        sample = FlowCal.transform.to_rfi(sample, fl_channel)
                    elif units.lower() == 'a.u.' or units.lower() == 'au':
                        units_label = "Arbitrary Units, a.u."
                        sample = FlowCal.transform.to_rfi(sample, fl_channel)

                    elif units.lower() == 'mef':
                        units_label = "Molecules of Equivalent Fluorophore, MEF"
                        # Check if transformation function is available
                        if mef_transform_fxns[sample_row['Beads ID']] is None:
                            raise ExcelUIException("MEF transformation "
                                "function not available")

                        # If beads_table is available, check if the same
                        # settings have been used to acquire the corresponding
                        # beads sample
                        if beads_table is not None:
                            beads_row = beads_table.loc[sample_row['Beads ID']]
                            # Instrument
                            beads_iid = beads_row['Instrument ID']
                            if beads_iid != sample_row['Instrument ID']:
                                raise ExcelUIException("Instruments for "
                                    "acquisition of beads and samples are not "
                                    "the same (beads {}'s instrument: {}, "
                                    "sample's instrument: {})".format(
                                        sample_row['Beads ID'],
                                        beads_iid,
                                        sample_row['Instrument ID']))
                            # Amplification type
                            beads_at = beads_row['{} Amp. Type'. \
                                format(fl_channel)]
                            if sample.amplification_type(fl_channel)[0]:
                                sample_at = "Log"
                            else:
                                sample_at = "Linear"
                            if beads_at != sample_at:
                                raise ExcelUIException("Amplification type for "
                                    "acquisition of beads and samples in "
                                    "channel {} are not the same (beads {}'s "
                                    "amplification: {}, sample's "
                                    "amplification: {})".format(
                                        fl_channel,
                                        sample_row['Beads ID'],
                                        beads_at,
                                        sample_at))
                            # Detector voltage
                            beads_dv = beads_row['{} Detector Volt.'. \
                                format(fl_channel)]
                            if beads_dv != sample.detector_voltage(fl_channel):
                                raise ExcelUIException("Detector voltage for "
                                    "acquisition of beads and samples in "
                                    "channel {} are not the same (beads {}'s "
                                    "detector voltage: {}, sample's "
                                    "detector voltage: {})".format(
                                        fl_channel,
                                        sample_row['Beads ID'],
                                        beads_dv,
                                        sample.detector_voltage(fl_channel)))

                        # Attempt to transform
                        # Transformation function raises a ValueError if a
                        # standard curve does not exist for a channel
                        try:
                            sample = mef_transform_fxns[sample_row['Beads ID']](
                                sample,
                                fl_channel)
                        except ValueError:
                            raise ExcelUIException("no standard curve for "
                                "channel {}".format(fl_channel))
                    else:
                        raise ExcelUIException("units \"{}\" not recognized". \
                            format(units, sample_id))

                    # Register that reporting in this channel must be done
                    report_channels.append(fl_channel)
                    report_units.append(units_label)

            ###
            # Gate
            ###
            if verbose:
                print("Performing gating...")
            # Remove first and last events. Transients in fluidics can make the
            # first few and last events slightly different from the rest.
            sample_gated = FlowCal.gate.start_end(sample,
                                                  num_start=250,
                                                  num_end=100)
            # Remove saturating events in forward/side scatter, and fluorescent
            # channels to report. The value of a saturating event is taken
            # automatically from `sample_gated.domain`.
            sample_gated = FlowCal.gate.high_low(sample_gated,
                                                 sc_channels + report_channels)
            # Density gating
            sample_gated, __, gate_contour = FlowCal.gate.density2d(
                data=sample_gated,
                channels=sc_channels,
                gate_fraction=sample_row['Gate Fraction'],
                full_output=True)

            # Plot forward/side scatter density plot and fluorescence histograms
            if plot:
                if verbose:
                    print("Plotting density plot and histogram...")
                # Define density plot parameters
                density_params = {}
                density_params['mode'] = 'scatter'
                density_params['xlog'] = bool(
                    sample_gated.amplification_type(sc_channels[0])[0])
                density_params['ylog'] = bool(
                    sample_gated.amplification_type(sc_channels[1])[0])
                density_params["title"] = "{} ({:.1f}% retained)".format(
                    sample_id,
                    sample_gated.shape[0] * 100. / sample.shape[0])
                # Define histogram plot parameters
                hist_params = []
                for rc, ru in zip(report_channels, report_units):
                    param = {}
                    param['div'] = 4
                    param['xlabel'] = '{} ({})'.format(rc, ru)
                    param['log'] = (ru != 'Channel Number') and \
                        bool(sample_gated.amplification_type(rc)[0])
                    hist_params.append(param)
                    
                # Plot
                figname = os.path.join(base_dir,
                                       plot_dir,
                                       "{}.png".format(sample_id))
                FlowCal.plot.density_and_hist(
                    sample,
                    sample_gated,
                    gate_contour=gate_contour,
                    density_channels=sc_channels,
                    density_params=density_params,
                    hist_channels=report_channels,
                    hist_params=hist_params,
                    savefig=figname)

        except ExcelUIException as e:
            # Print Exception message
            if verbose:
                print("ERROR: {}".format(str(e)))
            # Append exception to samples array
            samples.append(e)

        else:
            # If no errors were found, store results
            samples.append(sample_gated)

    return samples

def add_beads_stats(beads_table, beads_samples, mef_outputs=None):
    """
    Add stats fields to beads table.

    The following information is added to each row:
        - Notes (warnings, errors) resulting from the analysis
        - Number of Events
        - Acquisition Time (s)

    The following information is added for each row, for each channel in
    which MEF values have been specified:
        - Detector voltage (gain)
        - Amplification type
        - Bead model fitted parameters

    Parameters
    ----------
    beads_table : DataFrame
        Table specifying bead samples to analyze. For more information
        about the fields required in this table, please consult the
        module's documentation.
    beads_samples : list
        FCSData objects from which to calculate statistics.
        ``beads_samples[i]`` should correspond to
        ``beads_table.values()[i]``.
    mef_outputs : list, optional
        A list with the intermediate results of the generation of the MEF
        transformation functions, as given by ``mef.get_transform_fxn()``.
        This is used to populate the field ``<channel> Bead Model Params``.
        If specified, ``mef_outputs[i]`` should correspond to
        ``beads_table.values()[i]``.

    """
    # Add per-row info
    notes = []
    n_events = []
    acq_time = []
    for beads_sample in beads_samples:
        # Check if sample is an exception, otherwise assume it's an FCSData
        if isinstance(beads_sample, ExcelUIException):
            # Print error message
            notes.append("ERROR: {}".format(str(beads_sample)))
            n_events.append(np.nan)
            acq_time.append(np.nan)
        else:
            notes.append('')
            n_events.append(beads_sample.shape[0])
            acq_time.append(beads_sample.acquisition_time)

    beads_table['Analysis Notes'] = notes
    beads_table['Number of Events'] = n_events
    beads_table['Acquisition Time (s)'] = acq_time

    # List of channels that require stats columns
    headers = list(beads_table.columns)
    stats_headers = [h for h in headers if re_mef_values.match(h)]
    stats_channels = [re_mef_values.search(h).group(1) for h in stats_headers]

    # Iterate through channels
    for header, channel in zip(stats_headers, stats_channels):
        # Add empty columns to table
        beads_table[channel + ' Detector Volt.'] = np.nan
        beads_table[channel + ' Amp. Type'] = ""
        if mef_outputs:
            beads_table[channel + ' Beads Model'] = ""
            beads_table[channel + ' Beads Params. Names'] = ""
            beads_table[channel + ' Beads Params. Values'] = ""

        # Iterate
        for i, row_id in enumerate(beads_table.index):
            # If error, skip
            if isinstance(beads_samples[i], ExcelUIException):
                continue
            # If MEF values are specified, calculate stats. If not, leave empty.
            if pd.notnull(beads_table[header][row_id]):

                # Detector voltage
                beads_table.set_value(
                    row_id,
                    channel + ' Detector Volt.',
                    beads_samples[i].detector_voltage(channel))

                # Amplification type
                if beads_samples[i].amplification_type(channel)[0]:
                    amplification_type = "Log"
                else:
                    amplification_type = "Linear"
                beads_table.set_value(row_id,
                                      channel + ' Amp. Type',
                                      amplification_type)

                # Bead model and parameters
                # Only populate if mef_outputs has been provided
                if mef_outputs:
                    # Try to find the current channel among the mef'd channels.
                    # If successful, extract bead fitted parameters.
                    try:
                        mef_channel_index = mef_outputs[i]. \
                            mef_channels.index(channel)
                    except ValueError:
                        pass
                    else:
                        # Bead model
                        beads_model_str = mef_outputs[i]. \
                            fitting['beads_model_str'][mef_channel_index]
                        beads_table.set_value(row_id,
                                              channel + ' Beads Model',
                                              beads_model_str)
                        # Bead parameter names
                        params_names = mef_outputs[i]. \
                            fitting['beads_params_names'][mef_channel_index]
                        params_names_str = ", ".join([str(p)
                                                      for p in params_names])
                        beads_table.set_value(row_id,
                                              channel + ' Beads Params. Names',
                                              params_names_str)
                        # Bead parameter values
                        params = mef_outputs[i]. \
                            fitting['beads_params'][mef_channel_index]
                        params_str = ", ".join([str(p) for p in params])
                        beads_table.set_value(row_id,
                                              channel + ' Beads Params. Values',
                                              params_str)


def add_samples_stats(samples_table, samples):
    """
    Add stats fields to samples table.

    The following information is added to each row:
        - Notes (warnings, errors) resulting from the analysis
        - Number of Events
        - Acquisition Time (s)
    
    The following information is added for each row, for each channel in
    which fluorescence units have been specified:
        - Detector voltage (gain)
        - Amplification type
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
        Table specifying samples to analyze. For more information about the
        fields required in this table, please consult the module's
        documentation.
    samples : list
        FCSData objects from which to calculate statistics. ``samples[i]``
        should correspond to ``samples_table.values()[i]``

    """
    # Add per-row info
    notes = []
    n_events = []
    acq_time = []
    for sample in samples:
        # Check if sample is an exception, otherwise assume it's an FCSData
        if isinstance(sample, ExcelUIException):
            # Print error message
            notes.append("ERROR: {}".format(str(sample)))
            n_events.append(np.nan)
            acq_time.append(np.nan)
        else:
            notes.append('')
            n_events.append(sample.shape[0])
            acq_time.append(sample.acquisition_time)

    samples_table['Analysis Notes'] = notes
    samples_table['Number of Events'] = n_events
    samples_table['Acquisition Time (s)'] = acq_time

    # List of channels that require stats columns
    headers = list(samples_table.columns)
    stats_headers = [h for h in headers if re_units.match(h)]
    stats_channels = [re_units.search(h).group(1) for h in stats_headers]

    # Iterate through channels
    for header, channel in zip(stats_headers, stats_channels):
        # Add empty columns to table
        samples_table[channel + ' Detector Volt.'] = np.nan
        samples_table[channel + ' Amp. Type'] = ""
        samples_table[channel + ' Mean'] = np.nan
        samples_table[channel + ' Geom. Mean'] = np.nan
        samples_table[channel + ' Median'] = np.nan
        samples_table[channel + ' Mode'] = np.nan
        samples_table[channel + ' Std'] = np.nan
        samples_table[channel + ' CV'] = np.nan
        samples_table[channel + ' Geom. Std'] = np.nan
        samples_table[channel + ' Geom. CV'] = np.nan
        samples_table[channel + ' IQR'] = np.nan
        samples_table[channel + ' RCV'] = np.nan
        for row_id, sample in zip(samples_table.index, samples):
            # If error, skip
            if isinstance(sample, ExcelUIException):
                continue
            # If units are specified, calculate stats. If not, leave empty.
            if pd.notnull(samples_table[header][row_id]):
                # Acquisition settings
                # Detector voltage
                samples_table.set_value(row_id,
                                        channel + ' Detector Volt.',
                                        sample.detector_voltage(channel))
                # Amplification type
                if sample.amplification_type(channel)[0]:
                    amplification_type = "Log"
                else:
                    amplification_type = "Linear"
                samples_table.set_value(row_id,
                                        channel + ' Amp. Type',
                                        amplification_type)

                # Statistics from event list
                samples_table.set_value(row_id,
                                        channel + ' Mean',
                                        FlowCal.stats.mean(sample, channel))
                samples_table.set_value(row_id,
                                        channel + ' Geom. Mean',
                                        FlowCal.stats.gmean(sample, channel))
                samples_table.set_value(row_id,
                                        channel + ' Median',
                                        FlowCal.stats.median(sample, channel))
                samples_table.set_value(row_id,
                                        channel + ' Mode',
                                        FlowCal.stats.mode(sample, channel))
                samples_table.set_value(row_id,
                                        channel + ' Std',
                                        FlowCal.stats.std(sample, channel))
                samples_table.set_value(row_id,
                                        channel + ' CV',
                                        FlowCal.stats.cv(sample, channel))
                samples_table.set_value(row_id,
                                        channel + ' Geom. Std',
                                        FlowCal.stats.gstd(sample, channel))
                samples_table.set_value(row_id,
                                        channel + ' Geom. CV',
                                        FlowCal.stats.gcv(sample, channel))
                samples_table.set_value(row_id,
                                        channel + ' IQR',
                                        FlowCal.stats.iqr(sample, channel))
                samples_table.set_value(row_id,
                                        channel + ' RCV',
                                        FlowCal.stats.rcv(sample, channel))

def generate_histograms_table(samples_table, samples):
    """
    Generate a table of histograms as a DataFrame.

    Parameters
    ----------
    samples_table : DataFrame
        Table specifying samples to analyze. For more information about the
        fields required in this table, please consult the module's
        documentation.
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
    headers = list(samples_table.columns)
    hist_headers = [h for h in headers if re_units.match(h)]
    hist_channels = [re_units.search(h).group(1) for h in hist_headers]

    # The number of columns in the DataFrame has to be set to the maximum
    # number of bins of any of the histograms about to be generated.
    # The following iterates through these histograms and finds the
    # largest.
    n_columns = 0
    for sample_id, sample in zip(samples_table.index, samples):
        if isinstance(sample, ExcelUIException):
            continue
        for header, channel in zip(hist_headers, hist_channels):
            if pd.notnull(samples_table[header][sample_id]):
                bins = sample.domain(channel)
                if n_columns < len(bins):
                    n_columns = len(bins)

    # Declare multi-indexed DataFrame
    index = pd.MultiIndex.from_arrays([[],[],[]],
                                      names = ['Sample ID', 'Channel', ''])
    columns = ['Bin {}'.format(i + 1) for i in range(n_columns)]
    hist_table = pd.DataFrame([], index=index, columns=columns)

    # Generate histograms
    for sample_id, sample in zip(samples_table.index, samples):
        if isinstance(sample, ExcelUIException):
            continue
        for header, channel in zip(hist_headers, hist_channels):
            if pd.notnull(samples_table[header][sample_id]):
                # Get units in which bins are being reported
                unit = samples_table[header][sample_id]
                # Store bins
                bins = sample.domain(channel)
                hist_table.loc[(sample_id,
                                channel,
                                'Bin Values ({})'.format(unit)),
                               columns[0:len(bins)]] = bins
                # Calculate and store histogram counts
                bin_edges = sample.hist_bin_edges(channel)
                hist, __ = np.histogram(sample[:,channel], bins=bin_edges)
                hist_table.loc[(sample_id, channel, 'Counts'),
                               columns[0:len(bins)]] = hist

    return hist_table

def generate_about_table(extra_info={}):
    """
    Make a table with information about FlowCal and the current analysis.

    Parameters
    ----------
    extra_info : dict, optional
        Additional keyword:value pairs to include in the table.

    Returns
    -------
    about_table: DataFrame
        Table with information about FlowCal and the current analysis, as
        keyword:value pairs. The following keywords are included: FlowCal
        version, and date and time of analysis. Keywords and values from
        `extra_info` are also included.

    """
    # Make keyword and value arrays
    keywords = []
    values = []
    # FlowCal version
    keywords.append('FlowCal version')
    values.append(FlowCal.__version__)
    # Analysis date and time
    keywords.append('Date of analysis')
    values.append(time.strftime("%Y/%m/%d"))
    keywords.append('Time of analysis')
    values.append(time.strftime("%I:%M:%S%p"))
    # Add additional keyword:value pairs
    for k, v in extra_info.items():
        keywords.append(k)
        values.append(v)

    # Make table as data frame
    about_table = pd.DataFrame(values, index=keywords)

    # Set column names
    about_table.columns = ['Value']
    about_table.index.name = 'Keyword'

    return about_table

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

def run(input_path=None, output_path=None, verbose=True, plot=True):
    """
    Run the MS Excel User Interface.

    This function shows a dialog to open an input Excel workbook, loads FCS
    files and processes them as specified in the spreadsheet, and
    generates plots and an output workbook with statistics for each sample.

    Parameters
    ----------
    input_path: str
        Path to the Excel file to use as input. If None, show a dialog to
        select an input file.
    output_path: str
        Path to which to save the output Excel file. If None, use
        "`input_path`_output".
    verbose: bool, optional
        Whether to print information messages during the execution of this
        function.
    plot : bool, optional
        Whether to generate and save density/histogram plots of each
        sample, and each beads sample.

    """

    # If input file has not been specified, show open file dialog
    if input_path is None:
        input_path = show_open_file_dialog(filetypes=[('Excel files',
                                                       '*.xlsx')])
        if not input_path:
            if verbose:
                print("No input file selected.")
            return
    # Extract directory, filename, and filename with no extension from path
    input_dir, input_filename = os.path.split(input_path)
    input_filename_no_ext, __ = os.path.splitext(input_filename)

    # Read relevant tables from workbook
    if verbose:
        print("Reading {}...".format(input_filename))
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
    beads_samples, mef_transform_fxns, mef_outputs = process_beads_table(
        beads_table,
        instruments_table,
        base_dir=input_dir,
        verbose=verbose,
        plot=plot,
        plot_dir='plot_beads',
        full_output=True)

    # Add stats to beads table
    if verbose:
        print("")
        print("Calculating statistics for beads...")
    add_beads_stats(beads_table, beads_samples, mef_outputs)

    # Process samples
    samples = process_samples_table(
        samples_table,
        instruments_table,
        mef_transform_fxns=mef_transform_fxns,
        beads_table=beads_table,
        base_dir=input_dir,
        verbose=verbose,
        plot=plot,
        plot_dir='plot_samples')

    # Add stats to samples table
    if verbose:
        print("")
        print("Calculating statistics for all samples...")
    add_samples_stats(samples_table, samples)

    # Generate histograms
    if verbose:
        print("Generating histograms table...")
    histograms_table = generate_histograms_table(samples_table, samples)

    # Generate about table
    about_table = generate_about_table({'Input file path': input_path})

    # Generate list of tables to save
    table_list = []
    table_list.append(('Instruments', instruments_table))
    table_list.append(('Beads', beads_table))
    table_list.append(('Samples', samples_table))
    table_list.append(('Histograms', histograms_table))
    table_list.append(('About Analysis', about_table))

    # Write output excel file
    if verbose:
        print("Saving output Excel file...")
    if output_path is None:
        output_filename = "{}_output.xlsx".format(input_filename_no_ext)
        output_path = os.path.join(input_dir, output_filename)
    write_workbook(output_path, table_list)

    if verbose:
        print("\nDone.")

if __name__ == '__main__':
    # Read command line arguments
    import argparse
    parser = argparse.ArgumentParser(
        description="process flow cytometry files with FlowCal's Excel UI.")
    parser.add_argument(
        "-i",
        "--inputpath",
        type=str,
        nargs='?',
        help="input Excel file name. If not specified, show open file window")
    parser.add_argument(
        "-o",
        "--outputpath",
        type=str,
        nargs='?',
        help="output Excel file name. If not specified, use [INPUTPATH]_output")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="print information about individual processing steps")
    parser.add_argument(
        "-p",
        "--plot",
        action="store_true",
        help="generate and save density plots/histograms of beads and samples")
    args = parser.parse_args()

    # Run Excel UI
    run(input_path=args.inputpath,
        output_path=args.outputpath,
        verbose=args.verbose,
        plot=args.plot)
