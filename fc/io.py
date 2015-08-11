#!/usr/bin/python
#
# io.py - Module containing wrapper classes for flow cytometry data files.
#
# Authors: John T. Sexton (john.t.sexton@rice.edu)
#          Sebastian M. Castillo-Hair (smc9@rice.edu)
# Date: 6/30/2015
#
# Requires:
#   * numpy

import os
import copy
import collections
import warnings

import numpy as np

class FCSData(np.ndarray):
    '''Class describing an FCS Data file.

    The following versions are supported:
        - FCS 2.0
        - FCS 3.0
        - FCS 3.1
        - FCS 2.0 from CellQuestPro 5.1.1/BD FACScan Flow Cytometer

    We assume that the TEXT segment is such that:
        - There is only one data set in the file.
        - $MODE = 'L' (list mode, histogram mode not supported).
        - $DATATYPE = 'I' (unsigned integer), 'F' (32-bit floating point), or
            'D' (64-bit floating point). 'A' (ASCII) is not supported.
        - If $DATATYPE = 'I', $PnB % 8 = 0 for all channels.
        - $BYTEORD = '4,3,2,1' (big endian) or '1,2,3,4' (little endian)
        - $GATE = 0

    The object is an NxD numpy array representing N cytometry events with D
    dimensions extracted from the FCS DATA segment of an FCS file. The TEXT
    segment information is included in attributes. ANALYSIS information is not
    processed.

    Two additional attributes are implemented: channel_info stores information
    related to each channels, including name, gain, precalculated bins, etc.
    metadata keeps channel-independent, sample-specific information.
    
    Class Attributes:
        * infile - string or file-like object
        * text   - dictionary of KEY-VALUE pairs extracted from FCS TEXT
                   section
        * channel_info - list of dictionaries describing each channels. Keys:
            * 'label'
            * 'number'
            * 'pmt_voltage' (i.e. gain)
            * '100x_lin_gain'
            * 'amplifier' (values = 'lin' or 'log')
            * 'threshold'
            * 'range': [min, max, steps]
            * 'bin_vals': numpy array with bin values
            * 'bin_edges': numpy array with bin edges
        * metadata  - dictionary with additional channel-independent, 
                      sample-specific information.

    References:
    
    FCS2.0 Standard:
    [Dean, PN; Bagwell, CB; Lindmo, T; Murphy, RF; Salzman, GC (1990).
    "Data file standard for flow cytometry. Data File Standards Committee
    of the Society for Analytical Cytology.". Cytometry 11: 323-332.
    PMID 2340769]

    FCS3.0 Standard:
    [Seamer LC, Bagwell CB, Barden L, Redelman D, Salzman GC, Wood JC,
    Murphy RF. "Proposed new data file standard for flow cytometry, version
    FCS 3.0.". Cytometry. 1997 Jun 1;28(2):118-22. PMID 9181300]

    FCS3.1 Standard:
    [Spidlen J et al. "Data File Standard for Flow Cytometry, version
    FCS 3.1.". Cytometry A. 2010 Jan;77(1):97-100. PMID 19937951]
    
    Description of special BS$WORD TEXT fields:
    [https://lists.purdue.edu/pipermail/cytometry/2001-October/020624.html]
    '''

    @staticmethod
    def _read_fcs_text_segment(f, begin, end, delim = None):
        '''
        Parse region of specified file and interpret as TEXT segment.

        Since the ANALYSIS and supplemental TEXT segments are encoded in the
        same way, this function can also be used to parse them.

        f       - file-like object
        begin   - offset (in bytes) to first byte of TEXT segment
        end     - offset (in bytes) to last byte of TEXT segment
        delim   - 1-byte delimiter character (optional). If None, will extract
                  delimiter as first byte of TEXT segment. See FCS standards.

        returns - dictionary of KEY-VALUE pairs, string containing delimiter
        '''
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
            raise ImportError('TEXT segment should start and '
                + 'end with delimiter.')
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
            raise ImportError('Odd number of elements in TEXT segment; '
                + 'unpaired key or value.')

        text = dict(zip(text_list[0::2], text_list[1::2]))

        return text, delim

    @staticmethod
    def load_from_file(infile):
        '''
        Load data, text, and channel_info from FCS file.

        infile - String or file-like object. If string, it contains the 
                    name of the file to load data from. If file-like 
                    object, it refers to the file itself.

        returns:

        data            - numpy array with data from the FCS file
        text            - dictionary of KEY-VALUE pairs extracted from FCS TEXT
                            section.
        channel_info    - list of dictionaries describing each channels.
        '''

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

        # Extract number of bits per parameter
        num_channels = int(text['$PAR'])
        bits_per_channel = [int(text['$P{0}B'.format(p)])
                            for p in xrange(1, num_channels + 1)]

        # Check that number of bits is multiple of 8 if $DATATYPE == 'I'
        if text['$DATATYPE'] == 'I':
            if not all(bw % 8 == 0 for bw in bits_per_channel):
                raise NotImplementedError("If $DATATYPE = 'I', only byte"
                    + ' aligned parameter bit widths are'
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
            if (all(bw == 8  for bw in bits_per_channel) or
                all(bw == 16 for bw in bits_per_channel) or
                all(bw == 32 for bw in bits_per_channel) or
                all(bw == 64 for bw in bits_per_channel)):

                num_bits = bits_per_channel[0]

                dtype = np.dtype('{0}u{1}'.format('>' if big_endian else '<',
                                                  num_bits/8))
                data = np.memmap(
                    f,
                    dtype=dtype,
                    mode='r',
                    offset=data_begin,
                    shape=(n_events, n_channels),
                    order='C')

                # Cast memmap object to regular numpy array stored in memory
                data = np.array(data)

            # Special case: 24 bits, big endian, use the lowest 2 bytes
            elif all(bw == 24 for bw in bits_per_channel):
                # Timing: 0.35s for data003.fcs
                # Most of the load is in the last line

                dtype = np.dtype('>u1, >u2')
                data = np.memmap(
                    f,
                    dtype=dtype,
                    mode='r',
                    offset=data_begin,
                    shape=(n_events, n_channels),
                    order='C')

                # Keep only second word
                data = np.array([[chi[1] for chi in event] for event in data])

            else:
                raise NotImplementedError("ERROR")

        elif text['$DATATYPE'] in ('F', 'D'):
            # Get number of bits
            num_bits = 32 if text['$DATATYPE'] == 'F' else 64

            # Confirm that bit widths are consistent with data type
            if not all(bw == num_bits for bw in bits_per_channel):
                raise ValueError("All channel bit widths should be"
                    + " {0} if datatype is".format(num_bits)
                    + " '{0}' (bit widths: ".format(text['$DATATYPE'])
                    + "{0})".format(bits_per_channel))

            dtype = np.dtype('{0}f{1}'.format('>' if big_endian else '<',
                                              num_bits/8))
            data = np.memmap(
                f,
                dtype=dtype,
                mode='r',
                offset=data_begin,
                shape=(n_events, n_channels),
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
        '''
        Class constructor. 

        Special care needs to be taken when inheriting from a numpy array. 
        Details can be found here: 
        http://docs.scipy.org/doc/numpy/user/basics.subclassing.html
        '''

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
        '''Method called after all methods of construction of the class.'''
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
        '''Method called after numpy ufuncs.'''
        if out_arr.ndim == 0:
            return None
        else:
            return np.ndarray.__array_wrap__(self, out_arr, context)

    def __str__(self):
        '''Return name of fcs file.'''
        return os.path.basename(str(self.infile)) 

    @property
    def channels(self):
        ''' Return a list of the channel names
        '''
        return [i['label'] for i in self.channel_info]

    @property
    def time_step(self):
        ''' Return the time step of the time channel.

            The time step is such that self[:,'Time']*self.time_step is in
            seconds.

            In FCS2.0, the TIMETICKS keyword parameter contains the time step
            in milliseconds.
        '''
        return float(self.text['TIMETICKS'])/1000.

    @property
    def acquisition_time(self):
        ''' Return the acquisition time for this sample, in seconds.

        The acquisition time is calculated using the 'Time' channel by default.
        If the 'Time' channel is not available, the ETIM and BTIM keyword
        parameters will be used.
        '''
        # Check if the time channel is available
        if 'Time' in self.channels:
            # Use the event list
            return (self[-1, 'Time'] - self[0, 'Time']) * self.time_step
        else:
            # Use BTIM and ETIM keywords
            # In FCS2.0, times are specified as HH:MM:SS
            import time
            t0 = time.mktime(time.strptime(self.text['$BTIM'], '%H:%M:%S'))
            tf = time.mktime(time.strptime(self.text['$ETIM'], '%H:%M:%S'))
            return tf - t0

    def name_to_index(self, channels):
        '''Return the channel indexes for the named channels

        channels - String or list of strings indicating the name(s) of the 
                    channel(s) of interest.

        returns:
                - a number with the index(ces) of the channels of interest.
        '''

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
        '''Overriden __getitem__ function.

        This function achieves two things: It allows for channel indexing by
        channel name, and it takes care of properly slicing the channel_info 
        array.
        '''
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
        '''Overriden __setitem__ function.

        This function allows for channel indexing by channel name.
        '''
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
