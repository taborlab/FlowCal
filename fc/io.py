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

import numpy as np

class TaborLabFCSData(np.ndarray):
    '''Class describing FCS data which come off of the flow cytometer used
    in Jeff Tabor's lab at Rice University [http://www.taborlab.rice.edu/].

    The class itself is an NxD numpy array describing N cytometry events 
    observing D data dimensions extracted from FCS DATA section in the FCS file.
    
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

    Instrument: BD FACScan flow cytometer

    FCS file assumptions:
        * version = FCS2.0
        * $DATATYPE = I (unsigned binary integers)
        * $MODE = L (list mode)
        * $BYTEORD = 4,3,2,1 (big endian)
        * $PnB = 16 for all $PAR (meaning all channels use 2 bytes to describe
            data. This allows for straightforward use of numpy.memmap which
            results in a significant speedup)
        * only 1 data set per file
    
    For more details, see the FCS2.0 standard:
    
    [Dean, PN; Bagwell, CB; Lindmo, T; Murphy, RF; Salzman, GC (1990).
    "Data file standard for flow cytometry. Data File Standards Committee
    of the Society for Analytical Cytology.". Cytometry 11: 323-332.
    PMID 2340769]
    
    also, see mailing list post:
    
    [https://lists.purdue.edu/pipermail/cytometry/2001-October/020624.html]

    for BD$WORD key-value interpretations for FCS files coming off of BD
    instruments.
    
    Based in part on the fcm python library [https://github.com/jfrelinger/fcm].
    '''

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

        ###
        # Import relevant fields from HEADER section
        ###
        _version = f.read(10)

        if _version != 'FCS2.0    ':
            raise TypeError('incorrection FCS file version')
        else:
            _version = _version.strip()

        _text_begin = int(f.read(8))
        _text_end = int(f.read(8))
        
        _data_begin = int(f.read(8))
        _data_end = int(f.read(8))

        ###
        # Import key-value pairs from TEXT section
        ###
        f.seek(_text_begin)
        _separator = f.read(1)
        
        # Offsets point to the byte BEFORE the indicated boundary. This way,
        # you can seek to the offset and then read 1 byte to read the indicated
        # boundary. This means the length of the TEXT section is
        # ((end+1) - begin).
        f.seek(_text_begin)
        _text = f.read((_text_end+1) - _text_begin)

        l = _text.split(_separator)

        # The first and last list items should be empty because the TEXT
        # section starts and ends with the delimiter
        if l[0] != '' or l[-1] != '':
            raise ImportError('error parsing TEXT section')
        else:
            del l[0]
            del l[-1]

        # Detect if delimiter was used in keyword or value. This is technically
        # legal, but I'm too lazy to try and fix it because I don't think it's
        # relevant to us. According to the FCS2.0 standard, "If the separator
        # appears in a keyword or in a keyword value, it must be 'quoted' by
        # being repeated" and "null (zero length) keywords or keyword values
        # are not permitted", so this issue should manifest itself as an empty
        # element in the list.
        if any(x=='' for x in l):
            raise ImportError('error parsing TEXT section: delimiter used in'
                              + ' keyword or value.')

        # List length should be even since all key-value entries should be pairs
        if len(l) % 2 != 0:
            raise ImportError('error parsing TEXT section: odd # of'
                              + ' key-value entries')

        text = dict(zip(l[0::2], l[1::2]))

        # Confirm FCS file assumptions
        if text['$DATATYPE'] != 'I':
            raise TypeError('FCS file $DATATYPE is not I')

        if text['$MODE'] != 'L':
            raise TypeError('FCS file $MODE is not L')

        if text['$BYTEORD'] != '4,3,2,1':
            raise TypeError('FCS file $BYTEORD is not 4,3,2,1')

        num_channels = int(text['$PAR'])
        bits_per_channel = [int(text['$P%dB'%c])
                            for c in xrange(1,num_channels+1)]
        if not all(b==16 for b in bits_per_channel):
            raise TypeError('channel bit width error: $PnB != 16 for all'
                            + ' parameters (channels)')
        
        if text['$NEXTDATA'] != '0':
            raise TypeError('FCS file contains more than one data set')

        # From the following mailing list post:
        #
        # https://lists.purdue.edu/pipermail/cytometry/2001-October/020624.html
        #
        # the BD$WORD keys are interpreted for BD instruments. Populate a list
        # of dictionaries based on this interpretation to make it easier to
        # extract parameters like the channel gain.

        def amp(a):
            'Mapping of amplifier VALUE to human-readable string'
            if a is None:
                return None
            if a == '1':
                return 'lin'
            if a == '0':
                return 'log'
            raise ImportError('unrecognized amplifier setting')

        ch1 = {
            'label':text.get('$P1N'),
            'number':1,
            'pmt_voltage':text.get('BD$WORD13'),
            '100x_lin_gain':text.get('BD$WORD18'),
            'amplifier':amp(text.get('BD$WORD23')),
            'threshold':text.get('BD$WORD29')
            }
        ch2 = {
            'label':text.get('$P2N'),
            'number':2,
            'pmt_voltage':text.get('BD$WORD14'),
            '100x_lin_gain':text.get('BD$WORD19'),
            'amplifier':amp(text.get('BD$WORD24')),
            'threshold':text.get('BD$WORD30')
            }
        ch3 = {
            'label':text.get('$P3N'),
            'number':3,
            'pmt_voltage':text.get('BD$WORD15'),
            '100x_lin_gain':text.get('BD$WORD20'),
            'amplifier':amp(text.get('BD$WORD25')),
            'threshold':text.get('BD$WORD31')
            }
        ch4 = {
            'label':text.get('$P4N'),
            'number':4,
            'pmt_voltage':text.get('BD$WORD16'),
            '100x_lin_gain':text.get('BD$WORD21'),
            'amplifier':amp(text.get('BD$WORD26')),
            'threshold':text.get('BD$WORD32')
            }
        ch5 = {
            'label':text.get('$P5N'),
            'number':5,
            'pmt_voltage':text.get('BD$WORD17'),
            '100x_lin_gain':text.get('BD$WORD22'),
            'amplifier':amp(text.get('BD$WORD27')),
            'threshold':text.get('BD$WORD33')
            }
        ch6 = {
            'label':text.get('$P6N'),
            'number':6,
            }

        channel_info = [ch1, ch2, ch3, ch4, ch5, ch6]
        
        ###
        # Import DATA section
        ###
        shape = (int(text['$TOT']), int(text['$PAR']))

        # Sanity check that the total # of bytes that we're about to interpret
        # is exactly the # of bytes in the DATA section.
        if (shape[0]*shape[1]*2) != ((_data_end+1)-_data_begin):
            raise ImportError('DATA size does not match expected array size')

        # Use a numpy memmap object to interpret the binary data straight from
        # the file as a linear numpy array.
        data = np.memmap(
            f,
            dtype=np.dtype('>u2'),      # big endian, unsigned 2-byte integer
            mode='r',                   # read-only
            offset=_data_begin,
            shape=shape,
            order='C'                   # memory layout is row-major
            )

        # Cast memmap object to regular numpy array stored in memory (as
        # opposed to being backed by disk)
        data = np.array(data)

        # Close file if necessary
        if isinstance(infile, basestring):
            f.close()

        return (data, text, channel_info)

    def __new__(cls, infile):
        '''
        Class constructor. 

        Special care needs to be taken when inheriting from a numpy array. 
        Details can be found here: 
        http://docs.scipy.org/doc/numpy/user/basics.subclassing.html
        '''

        # First, load all data from fcs file
        data, text, channel_info = cls.load_from_file(infile)

        # Call constructor of numpy array
        obj = data.view(cls)

        # Add attributes
        obj.infile = infile
        obj.text = text
        obj.channel_info = channel_info

        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        '''Method called after all methods of construction of the class.'''
        # If called from explicit constructor, do nothing.
        if obj is None: return

        # Otherwise, copy attributes from "parent"
        self.infile = getattr(obj, 'infile', None)
        self.text = getattr(obj, 'text', None)
        self.channel_info = getattr(obj, 'channel_info', None)

    def __str__(self):
        return str(self.infile)

    @property
    def channels(self):
        return [i['label'] for i in self.channel_info]

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
        # Separate key components
        key_sample = key[0]
        key_channel = key[1]

        # Check if key_channel is a string, list/tuple, or other
        if isinstance(key_channel, basestring):
            key_channel = self.name_to_index(key_channel)
            key_all = (key_sample, key_channel)

        elif isinstance(key_channel, tuple) or isinstance(key_channel, list):
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
        else:
            new_arr.channel_info = [new_arr.channel_info[key_channel]]

        return new_arr
