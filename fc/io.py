#!/usr/bin/python
#
# io.py - Module containing wrapper classes for flow cytometry data files.
#
# Author: John T. Sexton (john.t.sexton@rice.edu)
# Date: 1/29/2015
#
# Requires:
#   * numpy

import numpy as np

class TaborLabFCSFile:
    '''Class describing FCS files which come off of the flow cytometer used
    in Jeff Tabor's lab at Rice University [http://www.taborlab.rice.edu/].

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

    Based in part on the fcm python library [https://github.com/jfrelinger/fcm].

    Class Attributes:
        * infile - string or file-like object
        * text - dictionary of KEY-VALUE pairs extracted from FCS TEXT part
        * data - numpy array of data extracted from FCS DATA part
        * channel_labels - list of strings describing channels
        * gains - dictionary mapping channel label to gain
    '''
    
    def __init__(self, infile):
        
        self.infile = infile

        if isinstance(infile, basestring):
            f = open(infile, 'rb')
        else:
            f = infile

        ###
        # Import relevant fields from HEADER part
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
        # Import key-value pairs from TEXT part
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

        num_channels = int(self.text['$PAR'])
        self.channel_labels = [self.text['$P%dN'%c]
                               for c in xrange(1,num_channels+1)]

        #TODO add gains

        # Confirm FCS file assumptions
        if self.text['$DATATYPE'] != 'I':
            raise TypeError('FCS file $DATATYPE is not I')

        if self.text['$MODE'] != 'L':
            raise TypeError('FCS file $MODE is not L')

        if self.text['$BYTEORD'] != '4,3,2,1':
            raise TypeError('FCS file $BYTEORD is not 4,3,2,1')

        bits_per_channel = [int(self.text['$P%dB'%c])
                            for c in xrange(1,num_channels+1)]
        if not all(b==16 for b in bits_per_channel):
            raise TypeError('channel bit width error: $PnB != 16 for all'
                            + ' parameters (channels)')
        
        if self.text['$NEXTDATA'] != '0':
            raise TypeError('FCS file contains more than one data set')
        
        ###
        # Import DATA part
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
