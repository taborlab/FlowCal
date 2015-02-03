#!/usr/bin/python
#
# io.py - Module containing wrapper classes for flow cytometry data files.
#
# Author: John T. Sexton (john.t.sexton@rice.edu)
# Date: 2/2/2015
#
# Requires:
#   * numpy

import numpy as np

class TaborLabFCSFile:
    '''Class describing FCS files which come off of the flow cytometer used
    in Jeff Tabor's lab at Rice University [http://www.taborlab.rice.edu/].
    
    Class Attributes:
        * infile - string or file-like object
        * text - dictionary of KEY-VALUE pairs extracted from FCS TEXT section
        * data - NxD numpy array describing N cytometry events observing D
                     data dimensions extracted from FCS DATA section
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
    
    def __init__(self, infile):
        'infile - string or file-like object'
        
        self.infile = infile

        if isinstance(infile, basestring):
            f = open(infile, 'rb')
        else:
            f = infile

        ###
        # Import relevant fields from HEADER section
        ###
        self._version = f.read(10)

        if self._version != 'FCS2.0    ':
            raise TypeError('incorrection FCS file version')
        else:
            self._version = self._version.strip()

        self._text_begin = int(f.read(8))
        self._text_end = int(f.read(8))
        
        self._data_begin = int(f.read(8))
        self._data_end = int(f.read(8))

        ###
        # Import key-value pairs from TEXT section
        ###
        f.seek(self._text_begin)
        self._separator = f.read(1)
        
        # Offsets point to the byte BEFORE the indicated boundary. This way,
        # you can seek to the offset and then read 1 byte to read the indicated
        # boundary. This means the length of the TEXT section is
        # ((end+1) - begin).
        f.seek(self._text_begin)
        self._text = f.read((self._text_end+1) - self._text_begin)

        l = self._text.split(self._separator)

        # The first and last list items should be empty because the TEXT
        # section starts and ends with the delimiter
        if l[0] != '' or l[-1] != '':
            raise ImportError('error parsing TEXT section')
        else:
            del l[0]
            del l[-1]

        # Detect if delimiter was used in keyword or value. This is technically
        # legal, but I'm too lazy to try and fix it because I don't think it's
        # relevant to us. This issue should manifest itself as an empty element
        # in the list since, according to the standard, any instance of the
        # delimiter in a keyword or a value must be "quoted" by repeating it.
        if any(x=='' for x in l):
            raise ImportError('error parsing TEXT section: delimiter used in'
                              + ' keyword or value.')

        # List length should be even since all key-value entries should be pairs
        if len(l) % 2 != 0:
            raise ImportError('error parsing TEXT section: odd # of'
                              + ' key-value entries')

        self.text = dict(zip(l[0::2], l[1::2]))

        # Confirm FCS file assumptions
        if self.text['$DATATYPE'] != 'I':
            raise TypeError('FCS file $DATATYPE is not I')

        if self.text['$MODE'] != 'L':
            raise TypeError('FCS file $MODE is not L')

        if self.text['$BYTEORD'] != '4,3,2,1':
            raise TypeError('FCS file $BYTEORD is not 4,3,2,1')

        num_channels = int(self.text['$PAR'])
        bits_per_channel = [int(self.text['$P%dB'%c])
                            for c in xrange(1,num_channels+1)]
        if not all(b==16 for b in bits_per_channel):
            raise TypeError('channel bit width error: $PnB != 16 for all'
                            + ' parameters (channels)')
        
        if self.text['$NEXTDATA'] != '0':
            raise TypeError('FCS file contains more than one data set')

        # From the following mailing list post:
        #
        # https://lists.purdue.edu/pipermail/cytometry/2001-October/020624.html
        #
        # the BD$WORD keys are interpreted for BD instruments. Populate a list
        # of dictionaries based on this interpretation to make it easier to
        # extract parameters like the channel gain.
        if num_channels != 6:
            raise ImportError('expecting 6 channels (FSC, SSC, FL1, FL2, FL3,'
                              + ' Time), detected %d'%num_channels)

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
            'label':self.text.get('$P1N'),
            'number':1,
            'pmt_voltage':self.text.get('BD$WORD13'),
            '100x_lin_gain':self.text.get('BD$WORD18'),
            'amplifier':amp(self.text.get('BD$WORD23')),
            'threshold':self.text.get('BD$WORD29')
            }
        ch2 = {
            'label':self.text.get('$P2N'),
            'number':2,
            'pmt_voltage':self.text.get('BD$WORD14'),
            '100x_lin_gain':self.text.get('BD$WORD19'),
            'amplifier':amp(self.text.get('BD$WORD24')),
            'threshold':self.text.get('BD$WORD30')
            }
        ch3 = {
            'label':self.text.get('$P3N'),
            'number':3,
            'pmt_voltage':self.text.get('BD$WORD15'),
            '100x_lin_gain':self.text.get('BD$WORD20'),
            'amplifier':amp(self.text.get('BD$WORD25')),
            'threshold':self.text.get('BD$WORD31')
            }
        ch4 = {
            'label':self.text.get('$P4N'),
            'number':4,
            'pmt_voltage':self.text.get('BD$WORD16'),
            '100x_lin_gain':self.text.get('BD$WORD21'),
            'amplifier':amp(self.text.get('BD$WORD26')),
            'threshold':self.text.get('BD$WORD32')
            }
        ch5 = {
            'label':self.text.get('$P5N'),
            'number':5,
            'pmt_voltage':self.text.get('BD$WORD17'),
            '100x_lin_gain':self.text.get('BD$WORD22'),
            'amplifier':amp(self.text.get('BD$WORD27')),
            'threshold':self.text.get('BD$WORD33')
            }
        ch6 = {
            'label':self.text.get('$P6N'),
            'number':6,
            }

        self.channel_info = [ch1, ch2, ch3, ch4, ch5, ch6]
        
        ###
        # Import DATA section
        ###
        shape = (int(self.text['$TOT']), int(self.text['$PAR']))

        # Sanity check that the total # of bytes that we're about to interpret
        # is exactly the # of bytes in the DATA section.
        if (shape[0]*shape[1]*2) != ((self._data_end+1)-self._data_begin):
            raise ImportError('DATA size does not match expected array size')

        # Use a numpy memmap object to interpret the binary data straight from
        # the file as a linear numpy array.
        data = np.memmap(
            f,
            dtype=np.dtype('>u2'),      # big endian, unsigned 2-byte integer
            mode='r',                   # read-only
            offset=self._data_begin,
            shape=shape,
            order='C'                   # memory layout is row-major
            )

        # Cast memmap object to regular numpy array stored in memory (as
        # opposed to being backed by disk)
        self.data = np.array(data)

        if isinstance(infile, basestring):
            f.close()

    def __repr__(self):
        return str(self.infile)
