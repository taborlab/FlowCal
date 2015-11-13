"""
Classes and utiliy functions for interpreting FCS files.

"""

import os
import copy
import collections
import warnings
import numpy as np

class FCSData(np.ndarray):
    """Class describing an FCS Data file.

    An `FCSData` object is an NxD numpy array representing N cytometry
    events with D dimensions extracted from the DATA segment of an FCS
    file. The TEXT segment information is included as a dictionary in one
    of the class attributes, `text`. ANALYSIS information is parsed
    similarly.

    Two additional attributes are implemented: `channel_info` stores
    information related to each channels, including name, gain,
    precalculated bins, etc. `metadata` keeps user-defined,
    channel-independent, sample-specific information, separate from `text`
    and `analysis`.

    The following versions of the FCS standard are supported:
        - FCS 2.0
        - FCS 3.0
        - FCS 3.1
        - FCS 2.0 from CellQuestPro 5.1.1/BD FACScan Flow Cytometer

    We assume that the TEXT segment is such that:
        - There is only one data set in the file.
        - $MODE = 'L' (list mode, histogram mode not supported).
        - $DATATYPE = 'I' (unsigned integer), 'F' (32-bit floating point),
            or 'D' (64-bit floating point). 'A' (ASCII) is not supported.
        - If $DATATYPE = 'I', $PnB % 8 = 0 for all channels.
        - $BYTEORD = '4,3,2,1' (big endian) or '1,2,3,4' (little endian)
        - $GATE = 0
    
    Attributes
    ----------
    infile : str or file-like object
        The path of the associated FCS file.
    text : dict
        Keyword-value pairs from the TEXT section of the FCS file.
    analysis : dict
        Keyword-value pairs from the ANALYSIS section of the FCS file.
    channel_info : list
        List of dictionaries, each one containing information about each
        channel. The keys of each one are:
        - label :  Name of the channel.
        - number : Channel index.
        - pmt_voltage : Voltage of the PMT detector.
        - amplifier : amplifier type, 'lin' or 'log'.
        - bin_vals : numpy array with bin values.
        - bin_vals : numpy array with bin edges.
    metadata : dict
        Additional channel-independent, sample-specific information.
    channels
    time_step
    acquisition_time

    References
    ----------
    .. [1] P.N. Dean, C.B. Bagwell, T. Lindmo, R.F. Murphy, G.C. Salzman,
       "Data file standard for flow cytometry. Data File Standards
       Committee of the Society for Analytical Cytology," Cytometry vol
       11, pp 323-332, 1990, PMID 2340769.

    .. [2] L.C. Seamer, C.B. Bagwell, L. Barden, D. Redelman, G.C. Salzman,
       J.C. Wood, R.F. Murphy, "Proposed new data file standard for flow
       cytometry, version FCS 3.0," Cytometry vol 28, pp 118-122, 1997,
       PMID 9181300.

    .. [3] J. Spidlen, et al, "Data File Standard for Flow Cytometry,
       version FCS 3.1," Cytometry A vol 77A, pp 97-100, 2009, PMID
       19937951.
    
    .. [4] R. Hicks, "BD$WORD file header fields,"
       https://lists.purdue.edu/pipermail/cytometry/2001-October/020624.html

    """

    @staticmethod
    def _read_fcs_text_segment(f, begin, end, delim = None):
        """Parse region of specified file and interpret as TEXT segment.

        Since the ANALYSIS and supplemental TEXT segments are encoded in
        the same way, this function can also be used to parse the ANALYSIS
        section.

        Parameters
        ----------
        f : file-like object
            FCS file.
        begin : int
            Offset (in bytes) to first byte of TEXT segment.
        end : int
            Offset (in bytes) to last byte of TEXT segment.
        delim : str, optional
            Delimiter character, placed at the start and the end of a
            keyword value. If None, will extract delimiter as first byte
            of TEXT segment.

        Returns
        -------
        text : dict
            Keyword-value pairs contained in the specified TEXT section.
        delim : str
            Delimiter character.

        """
        # Read delimiter if necessary
        if delim is None:
            f.seek(begin)
            delim = str(f.read(1))
        
        # Offsets point to the byte BEFORE the indicated boundary. This way,
        # you can seek to the offset and then read 1 byte to read the indicated
        # boundary. This means the length of the TEXT section is
        # ((end+1) - begin).
        f.seek(begin)
        text_raw = f.read((end + 1) - begin)

        text_list = text_raw.split(delim)

        # The first and last list items should be empty because the TEXT
        # section starts and ends with the delimiter
        if text_list[0] != '' or text_list[-1] != '':
            raise ImportError('Segment should start and end with delimiter.')
        else:
            del text_list[0]
            del text_list[-1]

        # Detect if delimiter was used in keyword or value. This is technically
        # legal, but I'm too lazy to try and fix it because I don't think it's
        # relevant to us. According to the FCS2.0 standard, "If the separator
        # appears in a keyword or in a keyword value, it must be 'quoted' by
        # being repeated" and "null (zero length) keywords or keyword values
        # are not permitted", so this issue should manifest itself as an empty
        # element in the list.
        if any(x=='' for x in text_list):
            raise ImportError('Use of delimiter in keywords or keyword '
                + 'values is not supported')

        # List length should be even since all key-value entries should be pairs
        if len(text_list) % 2 != 0:
            raise ImportError('Odd number of elements in segment; '
                + 'unpaired key or value.')

        text = dict(zip(text_list[0::2], text_list[1::2]))

        return text, delim

    @staticmethod
    def load_from_file(infile):
        """Load data, text, and channel_info from a specified FCS file.

        Parameters
        ----------
        infile : str or file-like object
            If string, it should contain the path of the FCS file to load
            data from. If file-like object, it should refer to the FCS file
            itself.

        Returns
        -------
        data : array
            NxD array containing information from the DATA section of the
            FCS file.
        text : dict
            Keyword-value pairs contained in the TEXT section of the FCS
            file.
        channel_info : list
           - List of dictionaries, each one containing information about
           each channel.

        """

        # Open file if necessary
        if isinstance(infile, basestring):
            f = open(infile, 'rb')
        else:
            f = infile

        ######################
        # Read HEADER segment
        ######################

        # Process version, throw error if not supported
        version = f.read(10).rstrip()
        if version not in ['FCS2.0', 'FCS3.0', 'FCS3.1']:
            raise TypeError('FCS version {} not supported.'.format(version))

        # Get segment offsets
        text_begin = int(f.read(8))
        text_end = int(f.read(8))
        
        data_begin = int(f.read(8))
        data_end = int(f.read(8))
        
        ab = f.read(8)
        ae = f.read(8)
        analysis_begin = (0 if ab == ' '*8 else int(ab))
        analysis_end = (0 if ab == ' '*8 else int(ae))

        ####################
        # Read TEXT segment
        ####################

        text, delim = FCSData._read_fcs_text_segment(
            f = f,
            begin = text_begin,
            end = text_end)

        # For FCS3.0 and above, supplemental TEXT segment offsets are always
        # specified via required key-value pairs in the primary TEXT segment.
        if version in ['FCS3.0','FCS3.1']:
            stext_begin = int(text['$BEGINSTEXT'])
            stext_end = int(text['$ENDSTEXT'])
            if stext_begin and stext_end:
                stext = FCSData._read_fcs_text_segment(
                    buf = f,
                    begin = stext_begin,
                    end = stext_end,
                    delim = delim)[0]
                text.update(stext)

        # Check Mode
        if text['$MODE'] != 'L':
            raise NotImplementedError("Only $MODE = 'L' is supported"
                + " (detected $MODE = '{}')".format(text['$MODE']))

        # Check datatype
        if text['$DATATYPE'] not in ('I','F','D'):
            raise NotImplementedError("Only $DATATYPE = 'I', 'F', and"
                + " 'D' are supported (detected $DATATYPE ="
                + " '{}')".format(text['$DATATYPE']))

        # Extract number channels and bits per channel
        num_channels = int(text['$PAR'])
        bits_per_channel = [int(text['$P{0}B'.format(p)])
                            for p in xrange(1, num_channels + 1)]

        # Check that number of bits is multiple of 8 if $DATATYPE == 'I'
        if text['$DATATYPE'] == 'I':
            if not all(bc % 8 == 0 for bc in bits_per_channel):
                raise NotImplementedError("If $DATATYPE = 'I', only byte"
                    + ' aligned channel bit widths are'
                    + ' supported (detected {0})'.format(
                        ', '.join('$P{0}B={1}'.format(
                                    p, text['$P{0}B'.format(p)])
                                for p in xrange(1, num_channels + 1)
                                if bits_per_channel[p - 1] % 8 != 0)))

        # Check byte ordering
        if text['$BYTEORD'] not in ('4,3,2,1', '2,1', '1,2,3,4', '1,2'):
            raise NotImplementedError("Only big endian ($BYTEORD = '4,3,2,1'"
                + " or '2,1') and little endian ($BYTEORD = '1,2,3,4' or"
                + " '1,2') are supported (detected $BYTEORD ="
                + " '{0}')".format(text['$BYTEORD']))

        # Check number of gate parameter
        if '$GATE' in text and int(text['$GATE']) > 0:
            raise NotImplementedError("Gate parameter parsing not supported.")

        # Check that there is no additional data set
        if int(text['$NEXTDATA']):
            warnings.warn('Additional data sets detected. Will ignore.'
                + ' ($NEXTDATA = {0})'.format(text['$NEXTDATA']))

        # Populate channel_info, to facilitate access to channel information.
        channel_info = []
        for i in range(1, num_channels + 1):
            chi = {}
            # Get label
            chi['label'] = text.get('$P{}N'.format(i))

            if chi['label'].lower() == 'time':
                pass
            else:
                # Gain
                if 'CellQuest Pro' in text.get('CREATOR'):
                    chi['pmt_voltage'] = text.get('BD$WORD{}'.format(12 + i))
                else:
                    chi['pmt_voltage'] = text.get('$P{}V'.format(i))
                # Amplification type
                if '$P{}E'.format(i) in text:
                    if text['$P{}E'.format(i)] == '0,0':
                        chi['amplifier'] = 'lin'
                    else:
                        chi['amplifier'] = 'log'
                else:
                    chi['amplifier'] = None
                # Range and bins
                PnR = '$P{}R'.format(i)
                chi['range'] = [0, int(text.get(PnR))-1, int(text.get(PnR))]
                chi['bin_vals'] = np.arange(int(text.get(PnR)))
                chi['bin_edges'] = np.arange(int(text.get(PnR)) + 1) - 0.5

            channel_info.append(chi)
        
        ####################
        # Read DATA segment
        ####################

        # Update data_begin and data_end if necessary
        if version in ('FCS3.0', 'FCS3.1'):
            data_begin = int(text['$BEGINDATA'])
            data_end = int(text['$ENDDATA'])

        # Get relevant TEXT keyword values
        n_events = int(text['$TOT'])
        n_channels = int(text['$PAR'])
        data_shape = (n_events, n_channels)
        big_endian = text['$BYTEORD'] in ('4,3,2,1', '2,1')

        # Check that the total number of bytes that we're about to read is
        # exactly the number of bytes in the DATA segment.
        # Assume that $DATATYPE is one of ('I', 'F', 'D')
        total_bits = np.sum(bits_per_channel)
        if (n_events * total_bits / 8) !=\
                ((data_end + 1) - data_begin):
            raise ImportError('DATA size does not match expected array'
                + ' size (array size ='
                + ' {0} bytes,'.format(n_events * total_bits / 8)
                + ' DATA segment size = {0} bytes)'.format(
                    data_end + 1 - data_begin))

        # Integer data type
        if text['$DATATYPE'] == 'I':

            # Check if all parameters fit into preexisting data types
            if (all(bc == 8  for bc in bits_per_channel) or
                all(bc == 16 for bc in bits_per_channel) or
                all(bc == 32 for bc in bits_per_channel) or
                all(bc == 64 for bc in bits_per_channel)):

                num_bits = bits_per_channel[0]

                dtype = np.dtype('{0}u{1}'.format('>' if big_endian else '<',
                                                  num_bits/8))
                data = np.memmap(
                    f,
                    dtype=dtype,
                    mode='r',
                    offset=data_begin,
                    shape=data_shape,
                    order='C')

                # Cast memmap object to regular numpy array stored in memory
                data = np.array(data)

            else:
                # The FCS standards technically allows for parameters to NOT be
                # byte aligned, but parsing a DATA segment which is not byte
                # aligned requires significantly more computation (and probably
                # an external library which exposes bit level resolution to a
                # block of memory). I don't think this is a common use case, so
                # I'm just going to detect it and raise an error.
                if (not all(bc % 8 == 0 for bc in bits_per_channel) or
                    any(bc > 64 for bc in bits_per_channel)):
                    raise NotImplementedError('Only byte aligned channel bit'
                        + ' widths (bw % 8 = 0), <= 64 are supported'
                        + ' (bits per channel: {0})'.format(bits_per_channel))

                # Read data in as a byte array
                byte_shape = (n_events, np.sum(bits_per_channel)/8)

                byte_data = np.memmap(
                    f,
                    dtype='uint8',  # endianness doesn't matter for 1 byte
                    mode='r',
                    offset=data_begin,
                    shape=byte_shape,
                    order='C')

                # Upcast all data to fit nearest supported data type of largest
                # bit width
                upcast_bw = int(2**np.max(np.ceil(np.log2(bits_per_channel))))

                # Create new array of upcast data type and use byte data to
                # populate it. The new array will have endianness native to
                # user's machine; does not preserve endianness.
                upcast_dtype = 'u{0}'.format(upcast_bw/8)
                data = np.zeros(data_shape, dtype=upcast_dtype)

                # Array mapping each column of data to first corresponding
                # column in byte_data
                byte_boundaries = np.roll(np.cumsum(bits_per_channel)/8, 1)
                byte_boundaries[0] = 0

                # Reconstitute columns of data by bit shifting appropriate
                # columns in byte_data and accumulating them
                for col in xrange(data.shape[1]):
                    num_bytes = bits_per_channel[col]/8
                    for b in xrange(num_bytes):
                        byte_data_col = byte_boundaries[col] + b
                        byteshift = (num_bytes - b - 1) if big_endian else b

                        if byteshift > 0:
                            # byte_data must be upcast or else bit shift fails
                            data[:,col] += \
                                byte_data[:,byte_data_col].astype(upcast_dtype)\
                                << (byteshift*8)
                        else:
                            data[:,col] += byte_data[:,byte_data_col]

            # To strictly follow the FCS standards, mask off the unused high
            # bits as specified by the parameter range.
            for col in xrange(data.shape[1]):
                # Obtain bits used from range parameter
                col_range = int(text['$P{}R'.format(col + 1)])
                bits_used = int(np.ceil(np.log2(col_range)))

                # Create a bit mask to mask off all but the lowest bits_used
                # bits. bitmask is a native python int type which does not have
                # an underlying size. The int type is effectively left-padded
                # with 0s (infinitely), and the '&' operation preserves the
                # dataype of the array, so this shouldn't be an issue.
                bitmask = ~((~0) << bits_used)
                data[:,col] &= bitmask


        elif text['$DATATYPE'] in ('F', 'D'):
            # Get number of bits
            num_bits = 32 if text['$DATATYPE'] == 'F' else 64

            # Confirm that bit widths are consistent with data type
            if not all(bc == num_bits for bc in bits_per_channel):
                raise ValueError("All bits per channel should be"
                    + " {0} if datatype is".format(num_bits)
                    + " '{0}' (bits per channel: ".format(text['$DATATYPE'])
                    + "{0})".format(bits_per_channel))

            dtype = np.dtype('{0}f{1}'.format('>' if big_endian else '<',
                                              num_bits/8))
            data = np.memmap(
                f,
                dtype=dtype,
                mode='r',
                offset=data_begin,
                shape=data_shape,
                order='C')

            # Cast memmap object to regular numpy array stored in memory
            data = np.array(data)

        elif text['$DATATYPE'] == 'A':
            raise NotImplementedError("Only 'I' (unsigned binary integer),"
                + " 'F' (single precision floating point), and 'D' (double"
                + " precision floating point) data types are supported"
                + " (detected datatype: '{0}')".format(text['$DATATYPE']))
        else:
            raise ValueError("Unrecognized datatype (detected datatype: "
                + "'{0}')".format(text['$DATATYPE']))

        ########################
        # Read ANALYSIS segment
        ########################

        # Update analysis_begin and analysis_end if necessary
        if version in ('FCS3.0', 'FCS3.1'):
            analysis_begin = int(text['$BEGINANALYSIS'])
            analysis_end = int(text['$ENDANALYSIS'])

        # Read analysis segment
        if analysis_begin and analysis_end:
            analysis = FCSData._read_fcs_text_segment(
                f = f,
                begin = analysis_begin,
                end = analysis_end,
                delim = delim)[0]
        else:
            analysis = {}

        # Close file if necessary
        if isinstance(infile, basestring):
            f.close()

        return (data, text, analysis, channel_info)

    def __new__(cls, infile, metadata = {}):
        """Class constructor.

        Parameters
        ----------
        infile : str or file-like object
            If string, it should contain the path of the FCS file to load
            data from. If file-like object, it should refer to the FCS file
            itself.
        metadata : str
            Additional channel-independent, sample-specific information, to
            be copied without modification to the `metadata` attribute.

        Notes
        -----
        Since this class inherits from a numpy array, we use the function
        `__new__` and not `__init__`. `cls.view` needs to be called iniside
        the `__new__` function. For more details, consult
        http://docs.scipy.org/doc/numpy/user/basics.subclassing.html

        """

        # Load all data from fcs file
        data, text, analysis, channel_info = cls.load_from_file(infile)

        # Call constructor of numpy array
        obj = data.view(cls)

        # Add attributes
        obj.infile = infile
        obj.text = text
        obj.analysis = analysis
        obj.channel_info = channel_info
        obj.metadata = metadata

        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        """Method called after all methods of construction of the class."""

        # If called from explicit constructor, do nothing.
        if obj is None: return

        # Otherwise, copy attributes from "parent"
        self.infile = getattr(obj, 'infile', None)
        if hasattr(obj, 'text'):
            self.text = copy.deepcopy(obj.text)
        if hasattr(obj, 'analysis'):
            self.analysis = copy.deepcopy(obj.analysis)
        if hasattr(obj, 'channel_info'):
            self.channel_info = copy.deepcopy(obj.channel_info)
        if hasattr(obj, 'metadata'):
            self.metadata = copy.deepcopy(obj.metadata)

    def __array_wrap__(self, out_arr, context = None):
        """Method called after numpy ufuncs."""
        if out_arr.ndim == 0:
            return None
        else:
            return np.ndarray.__array_wrap__(self, out_arr, context)

    def __str__(self):
        """Return name of FCS file."""
        return os.path.basename(str(self.infile)) 

    @property
    def channels(self):
        """List of channel names."""
        return [i['label'] for i in self.channel_info]

    @property
    def time_step(self):
        """Time step of the time channel.

        The time step is such that ``self[:,'Time']*time_step`` is in
        seconds.

        FCS Standard files store the time step in the $TIMESTEP keyword.
        In CellQuest Pro's FCS2.0, the TIMETICKS keyword parameter contains
        the time step in milliseconds.

        Raises
        ------
        IOError
            If the $TIMESTEP and the $TIMETICKS keywords are both not
            available.

        """

        if 'TIMETICKS' in self.text:
            return float(self.text['TIMETICKS'])/1000.
        elif '$TIMESTEP' in self.text:
            return float(self.text['$TIMESTEP'])
        else:
            raise IOError("Time information not available.")

    @property
    def acquisition_time(self):
        """Acquisition time, in seconds.

        The acquisition time is calculated using the 'time' channel by
        default (case independent). If the 'time' channel is not available,
        the ETIM and BTIM keyword parameters will be used, if available.

        Raises
        ------
        IOError
            If the 'time' channel and the ETIM and BTIM keywords are not
            available.

        """
        # Get time channels indices
        channel_i = [i for i, chi in enumerate(self.channels)\
                                                    if chi.lower() == 'time']
        if len(channel_i) > 1:
            raise KeyError("More than one time channel in data.")
        # Check if the time channel is available
        elif len(channel_i) == 1:
            # Use the event list
            ch = self.channels[channel_i[0]]
            return (self[-1, ch] - self[0, ch]) * self.time_step
        elif '$BTIM' and '$ETIM' in self.text:
            # Use BTIM and ETIM keywords
            # In FCS2.0, times are specified as HH:MM:SS
            # In FCS3.0, times are specified as HH:MM:SS[.cc] (cc optional)
            # First, separate string into HH:MM:SS and .cc parts
            t0s = self.text['$BTIM'].split('.')
            tfs = self.text['$ETIM'].split('.')
            # Read HH:MM:SS portion and subtract
            import datetime
            t0 = datetime.datetime.strptime(t0s[0], '%H:%M:%S')
            tf = datetime.datetime.strptime(tfs[0], '%H:%M:%S')
            dt = (tf - t0).total_seconds()
            # Add .cc portion if available
            if len(t0s) > 1:
                dt = dt - float(t0s[1])/100
            if len(tfs) > 1:
                dt = dt + float(tfs[1])/100
            return dt
        else:
            raise IOError("Time information not available.")

    def name_to_index(self, channels):
        """Return the channel indices for the specified channel names.

        Parameters
        ----------
        channels : str or list of str
            Name(s) of the channel(s) of interest.

        Returns
        -------
        int or list of int
            Numerical index(ces) of the specified channels.

        """

        if isinstance(channels, basestring):
            # channels is a string containing a channel name
            if channels in self.channels:
                return self.channels.index(channels)
            else:
                raise ValueError("{} is not a valid channel name."
                    .format(channels))

        elif isinstance(channels, list):
            # channels is a list of strings
            lst = []
            for ci in channels:
                if ci in self.channels:
                    lst.append(self.channels.index(ci))
                else:
                    raise ValueError("{} is not a valid channel name."
                        .format(ci))
            return lst

        else:
            raise ValueError("Input argument should be a string or list \
                of strings.")

    def __getitem__(self, key):
        """Overriden __getitem__ function.

        If the second value of the provided `key` is a string corresponding
        to a valid channel name, this function converts it to a number and
        passes it to ndarray's `__getitem__`. This allows for indexing by
        channel name. In addition, this function takes care of properly
        slicing the `channel_info` attribute.

        """

        # If key is a tuple with no None, decompose and interpret key[1] as 
        # the channel. If it contains Nones, pass directly to 
        # ndarray.__getitem__() and convert to np.ndarray. Otherwise, pass
        # directly to ndarray.__getitem__().
        if isinstance(key, tuple) and len(key) == 2 \
            and key[0] is not None and key[1] is not None:
            # Separate key components
            key_sample = key[0]
            key_channel = key[1]

            # Check if key_channel is a string, list/tuple, or other
            if isinstance(key_channel, basestring):
                key_channel = self.name_to_index(key_channel)
                key_all = (key_sample, key_channel)

            elif hasattr(key_channel, '__iter__'):
                # Make mutable
                key_channel = list(key_channel)  
                # Change any strings into channel indices
                for i, j in enumerate(key_channel):
                    if isinstance(j, basestring):
                        key_channel[i] = self.name_to_index(j)
                key_all = (key_sample, key_channel)

            else:
                key_all = (key_sample, key_channel)

            # Get sliced array
            new_arr = np.ndarray.__getitem__(self, key_all)
            # Return if not an array
            if not hasattr(new_arr, '__iter__'):
                return new_arr

            # Finally, slice the channel_info attribute
            if hasattr(key_channel, '__iter__'):
                new_arr.channel_info = [new_arr.channel_info[kc] \
                    for kc in key_channel]
            elif isinstance(key_channel, slice):
                new_arr.channel_info = new_arr.channel_info[key_channel]
            else:
                new_arr.channel_info = [new_arr.channel_info[key_channel]]

        elif isinstance(key, tuple) and len(key) == 2 \
            and (key[0] is None or key[1] is None):
            # Get sliced array and convert to np.ndarray
            new_arr = np.ndarray.__getitem__(self, key)
            new_arr = new_arr.view(np.ndarray)

        else:
            # Get sliced array using native getitem function.
            new_arr = np.ndarray.__getitem__(self, key)

        return new_arr

    def __setitem__(self, key, item):
        """Overriden __setitem__ function.

        If the second value of the provided `key` is a string corresponding
        to a valid channel name, this function converts it to a number and
        passes it to ndarray's `__setitem__`. This allows for indexing by
        channel name when writing to a FCSData object.

        """

        # If key is a tuple with no Nones, decompose and interpret key[1] as 
        # the channel. If it contains Nones, pass directly to 
        # ndarray.__setitem__().
        if isinstance(key, tuple) and len(key) == 2 \
            and key[0] is not None and key[1] is not None:
            # Separate key components
            key_sample = key[0]
            key_channel = key[1]

            # Check if key_channel is a string, list/tuple, or other
            if isinstance(key_channel, basestring):
                key_channel = self.name_to_index(key_channel)
                key_all = (key_sample, key_channel)

            elif hasattr(key_channel, '__iter__'):
                # Make mutable
                key_channel = list(key_channel)  
                # Change any strings into channel indices
                for i, j in enumerate(key_channel):
                    if isinstance(j, basestring):
                        key_channel[i] = self.name_to_index(j)
                key_all = (key_sample, key_channel)

            else:
                key_all = (key_sample, key_channel)

            # Write into array
            np.ndarray.__setitem__(self, key_all, item)

        else:
            # Get sliced array using native getitem function.
            np.ndarray.__setitem__(self, key, item)
