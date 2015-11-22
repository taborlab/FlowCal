"""
Classes and utiliy functions for interpreting FCS files.

"""

import os
import copy
import collections
import warnings
import numpy as np

###
# Utility functions for importing segments of FCS files
###

def read_fcs_header_segment(buf, begin=0):
    """
    Parse HEADER segment of FCS file.

    Parameters
    ----------
    buf : file-like object
        Buffer containing data to interpret as HEADER segment.
    begin : int
        Offset (in bytes) to first byte of HEADER segment in `buf`.

    Returns
    -------
    header : namedtuple
        ``namedtuple`` containing version information and byte offset
        values of other FCS segments (see FCS standards for more
        information) in the following order:
        version : str
        text_begin : int
        text_end : int
        data_begin : int
        data_end : int
        analysis_begin : int
        analysis_end : int

    Notes
    -----
    Blank ANALYSIS segment offsets are converted to zeros.

    OTHER segment offsets are ignored (see FCS standards for more
    information about OTHER segments).

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

    """
    fields = [
        'version',
        'text_begin',
        'text_end',
        'data_begin',
        'data_end',
        'analysis_begin',
        'analysis_end']

    FCSHeader = collections.namedtuple('FCSHeader', fields)

    field_values = []

    buf.seek(begin)
    field_values.append(str(buf.read(10)).rstrip())     # version

    field_values.append(int(buf.read(8)))               # text_begin
    field_values.append(int(buf.read(8)))               # text_end
    field_values.append(int(buf.read(8)))               # data_begin
    field_values.append(int(buf.read(8)))               # data_end

    fv = buf.read(8)                                    # analysis_begin
    field_values.append(0 if fv == ' '*8 else int(fv))
    fv = buf.read(8)                                    # analysis_end
    field_values.append(0 if fv == ' '*8 else int(fv))

    header = FCSHeader._make(field_values)
    return header

def read_fcs_text_segment(buf, begin, end, delim=None):
    """
    Parse TEXT segment of FCS file.

    Parameters
    ----------
    buf : file-like object
        Buffer containing data to interpret as TEXT segment.
    begin : int
        Offset (in bytes) to first byte of TEXT segment in `buf`.
    end : int
        Offset (in bytes) to last byte of TEXT segment in `buf`.
    delim : str, optional
        1-byte delimiter character which delimits key-value entries of
        TEXT segment. If None, will extract delimter as first byte
        of TEXT segment.

    Returns
    -------
    text : dict
        Dictionary of key-value entries extracted from TEXT segment.
    delim : str
        String containing delimiter character.

    Raises
    ------
    ValueError
        If TEXT segment does not start and end with delimiter.
    ValueError
        If function detects odd number of total extracted keys and
        values (indicating an unpaired key or value).
    NotImplementedError
        If delimiter is used in a keyword or value.

    Notes
    -----
    ANALYSIS segments and TEXT segments are parsed the same way, so
    this function can also be used to parse ANALYSIS segments.

    This function does not automatically parse supplemental TEXT
    segments (see FCS3.0 [2]_). Supplemental TEXT segments and regular
    TEXT segments are parsed the same way, though, so this function
    can be manually directed to parse a supplemental TEXT segment by
    providing the appropriate `begin` and `end` values.

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

    """
    if delim is None:
        buf.seek(begin)
        delim = str(buf.read(1))

    # The offsets are inclusive (meaning they specify first and last byte
    # WITHIN segment) and seeking is inclusive (read() after seek() reads the
    # byte which was seeked to). This means the length of the segment is
    # ((end+1) - begin).
    buf.seek(begin)
    raw = buf.read((end+1)-begin)

    pairs_list = raw.split(delim)

    # The first and last list items should be empty because the TEXT
    # segment starts and ends with the delimiter
    if pairs_list[0] != '' or pairs_list[-1] != '':
        raise ValueError("segment should start and end with delimiter")
    else:
        del pairs_list[0]
        del pairs_list[-1]

    # Detect if delimiter was used in keyword or value (which, according to
    # the standards, is technically legal). According to the FCS2.0 standard,
    # "If the separator appears in a keyword or in a keyword value, it must be
    # 'quoted' by being repeated" and "null (zero length) keywords or keyword
    # values are not permitted", so this issue should manifest itself as an
    # empty element in the list.
    if any(x=='' for x in pairs_list):
        raise NotImplementedError("use of delimiter in keywords or keyword"
            + " values is not supported")

    # List length should be even since all key-value entries should be pairs
    if len(pairs_list) % 2 != 0:
        raise ValueError("odd # of (keys + values); unpaired key or value")

    text = dict(zip(pairs_list[0::2], pairs_list[1::2]))

    return text, delim

def read_fcs_data_segment(buf,
                          begin,
                          end,
                          datatype,
                          num_events,
                          param_bit_widths,
                          big_endian,
                          param_ranges=None):
    """
    Parse DATA segment of FCS file.

    Parameters
    ----------
    buf : file-like object
        Buffer containing data to interpret as DATA segment.
    begin : int
        Offset (in bytes) to first byte of DATA segment in `buf`.
    end : int
        Offset (in bytes) to last byte of DATA segment in `buf`.
    datatype : {'I', 'F', 'D', 'A'}
        String specifying FCS file datatype (see $DATATYPE keyword from
        FCS standards). Supported datatypes include 'I' (unsigned
        binary integer), 'F' (single precision floating point), and 'D'
        (double precision floating point). 'A' (ASCII) is recognized
        but not supported.
    num_events : int
        Total number of events (see $TOT keyword from FCS standards).
    param_bit_widths : array-like
        Array specifying parameter (aka channel) bit width for each
        parameter (see $PnB keywords from FCS standards). The length of
        `param_bit_widths` should match the $PAR keyword value from the
        FCS standards (which indicates the total number of parameters).
        If `datatype` is 'I', data must be byte aligned (i.e. all
        parameter bit widths should be divisible by 8), and data are
        upcast to the nearest uint8, uint16, uint32, or uint64 data
        type. Bit widths larger than 64 bits are not supported.
    big_endian : bool
        Endianness of computer used to acquire data (see $BYTEORD
        keyword from FCS standards). True implies big endian; False
        implies little endian.
    param_ranges : array-like, optional
        Array specifying parameter (aka channel) range for each
        parameter (see $PnR keywords from FCS standards). Used to
        ensure erroneous values are not read from DATA segment by
        applying a bit mask to remove unused bits. The length of
        `param_ranges` should match the $PAR keyword value from the FCS
        standards (which indicates the total number of parameters). If
        None, no masking is performed.

    Returns
    -------
    data : numpy array
        NxD numpy array describing N cytometry events observing D data
        dimensions.

    Raises
    ------
    ValueError
        If lengths of `param_bit_widths` and `param_ranges` don't match.
    ValueError
        If calculated DATA segment size (as determined from the number
        of events, the number of parameters, and the number of bytes per
        data point) does not match size specified by `begin` and `end`.
    ValueError
        If `param_bit_widths` doesn't agree with `datatype` for single
        precision or double precision floating point (i.e. they should
        all be 32 or 64, respectively).
    ValueError
        If `datatype` is unrecognized.
    NotImplementedError
        If `datatype` is 'A'.
    NotImplementedError
        If `datatype` is 'I' but data is not byte aligned.

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

    """
    num_params = len(param_bit_widths)
    if (param_ranges is not None and len(param_ranges) != num_params):
        raise ValueError("param_bit_widths and param_ranges must have same"
            + " length")

    shape = (int(num_events), num_params)

    if datatype == 'I':
        # Check if all parameters fit into preexisting data type
        if (all(bw == 8  for bw in param_bit_widths) or
            all(bw == 16 for bw in param_bit_widths) or
            all(bw == 32 for bw in param_bit_widths) or
            all(bw == 64 for bw in param_bit_widths)):

            num_bits = param_bit_widths[0]

            # Sanity check that the total # of bytes that we're about to
            # interpret is exactly the # of bytes in the DATA segment.
            if (shape[0]*shape[1]*(num_bits/8)) != ((end+1)-begin):
                raise ValueError("DATA size does not match expected array"
                    + " size (array size ="
                    + " {0} bytes,".format(shape[0]*shape[1]*(num_bits/8))
                    + " DATA segment size = {0} bytes)".format((end+1)-begin))

            dtype = np.dtype('{0}u{1}'.format('>' if big_endian else '<',
                                              num_bits/8))
            data = np.memmap(
                buf,
                dtype=dtype,
                mode='r',
                offset=begin,
                shape=shape,
                order='C')

            # Cast memmap object to regular numpy array stored in memory (as
            # opposed to being backed by disk)
            data = np.array(data)
        else:
            # The FCS standards technically allows for parameters to NOT be
            # byte aligned, but parsing a DATA segment which is not byte
            # aligned requires significantly more computation (and probably an
            # external library which exposes bit level resolution to a block
            # of memory). I don't think this is a common use case, so I'm just
            # going to detect it and raise an error.
            if (not all(bw % 8 == 0 for bw in param_bit_widths) or
                any(bw > 64 for bw in param_bit_widths)):
                raise NotImplementedError("only byte aligned parameter bit"
                    + " widths (bw % 8 = 0) <= 64 are supported"
                    + " (param_bit_widths={0})".format(param_bit_widths))

            # Read data in as a byte array
            byte_shape = (int(num_events),
                          np.sum(np.array(param_bit_widths)/8))

            # Sanity check that the total # of bytes that we're about to
            # interpret is exactly the # of bytes in the DATA segment.
            if (byte_shape[0]*byte_shape[1]) != ((end+1)-begin):
                raise ValueError("DATA size does not match expected array"
                    + " size (array size ="
                    + " {0} bytes,".format(byte_shape[0]*byte_shape[1])
                    + " DATA segment size = {0} bytes)".format((end+1)-begin))

            byte_data = np.memmap(
                buf,
                dtype='uint8',  # endianness doesn't matter for 1 byte
                mode='r',
                offset=begin,
                shape=byte_shape,
                order='C')

            # Upcast all data to fit nearest supported data type of largest
            # bit width
            upcast_bw = int(2**np.max(np.ceil(np.log2(param_bit_widths))))

            # Create new array of upcast data type and use byte data to
            # populate it. The new array will have endianness native to user's
            # machine; does not preserve endianness of stored FCS data.
            upcast_dtype = 'u{0}'.format(upcast_bw/8)
            data = np.zeros(shape,dtype=upcast_dtype)

            # Array mapping each column of data to first corresponding column
            # in byte_data
            byte_boundaries = np.roll(np.cumsum(param_bit_widths)/8,1)
            byte_boundaries[0] = 0

            # Reconstitute columns of data by bit shifting appropriate columns
            # in byte_data and accumulating them
            for col in xrange(data.shape[1]):
                num_bytes = param_bit_widths[col]/8
                for b in xrange(num_bytes):
                    byte_data_col = byte_boundaries[col] + b
                    byteshift = (num_bytes-b-1) if big_endian else b

                    if byteshift > 0:
                        # byte_data must be upcast or else bit shift fails
                        data[:,col] += \
                            byte_data[:,byte_data_col].astype(upcast_dtype) \
                            << (byteshift*8)
                    else:
                        data[:,col] += byte_data[:,byte_data_col]

        if param_ranges is not None:
            # To strictly follow the FCS standards, mask off the unused high bits
            # as specified by param_ranges.
            for col in xrange(data.shape[1]):
                # bits_used should be related to resolution of cytometer ADC
                bits_used = int(np.ceil(np.log2(param_ranges[col])))

                # Create a bit mask to mask off all but the lowest bits_used bits.
                # bitmask is a native python int type which does not have an
                # underlying size. The int type is effectively left-padded with
                # 0s (infinitely), and the '&' operation preserves the dataype of
                # the array, so this shouldn't be an issue.
                bitmask = ~((~0) << bits_used)
                data[:,col] &= bitmask

    elif datatype in ('F','D'):
        num_bits = 32 if datatype == 'F' else 64

        # Confirm that bit widths are consistent with data type
        if not all(bw == num_bits for bw in param_bit_widths):
            raise ValueError("all param_bit_widths should be"
                + " {0} if datatype =".format(num_bits)
                + " \'{0}\' (param_bit_widths=".format(datatype)
                + "{0})".format(param_bit_widths))

        # Sanity check that the total # of bytes that we're about to interpret
        # is exactly the # of bytes in the DATA segment.
        if (shape[0]*shape[1]*(num_bits/8)) != ((end+1)-begin):
            raise ValueError("DATA size does not match expected array size"
                + " (array size = {0}".format(shape[0]*shape[1]*(num_bits/8))
                + " bytes, DATA segment size ="
                + " {0} bytes)".format((end+1)-begin))

        dtype = np.dtype('{0}f{1}'.format('>' if big_endian else '<',
                                          num_bits/8))
        data = np.memmap(
            buf,
            dtype=dtype,
            mode='r',
            offset=begin,
            shape=shape,
            order='C')

        # Cast memmap object to regular numpy array stored in memory (as
        # opposed to being backed by disk)
        data = np.array(data)
    elif datatype == 'A':
        raise NotImplementedError("only \'I\' (unsigned binary integer),"
            + " \'F\' (single precision floating point), and \'D\' (double"
            + " precision floating point) data types are supported (detected"
            + " datatype=\'{0}\')".format(datatype))
    else:
        raise ValueError("unrecognized datatype (detected datatype="
            + "\'{0}\')".format(datatype))

    return data

###
# Classes
###

class FCSFile(object):
    """
    Class representing an FCS flow cytometry data file.

    This class parses a binary FCS file and exposes a read-only view
    of the HEADER, TEXT, DATA, and ANALYSIS segments via Python-friendly
    data structures.

    Parameters
    ----------
    infile : str or file-like
        Reference to the associated FCS file.

    Attributes
    ----------
    infile : str or file-like
        Reference to associated FCS file.
    header : namedtuple
        ``namedtuple`` containing version information and byte offset
        values of other FCS segments in the following order:
        version : str
        text_begin : int
        text_end : int
        data_begin : int
        data_end : int
        analysis_begin : int
        analysis_end : int
    text : dict
        Dictionary of keyword-value entries from TEXT segment and
        optional supplemental TEXT segment.
    data : numpy array
        Unwriteable NxD numpy array describing N cytometry events
        observing D data dimensions.
    analysis : dict
        Dictionary of keyword-value entries from ANALYSIS segment.

    Raises (see FCS standards for more information)
    -----------------------------------------------
    NotImplementedError
        If $MODE is not 'L'.
    NotImplementedError
        If $DATATYPE is not 'I', 'F', or 'D'.
    NotImplementedError
        If $DATATYPE is 'I' but data is not byte aligned.
    NotImplementedError
        If $BYTEORD is not big endian ('4,3,2,1' or '2,1') or little
        endian ('1,2,3,4', '1,2').
    ValueError
        If TEXT-like segment does not start and end with delimiter.
    ValueError
        If TEXT-like segment has odd number of total extracted keys and
        values (indicating an unpaired key or value).
    NotImplementedError
        If the TEXT segment delimiter is used in a TEXT-like segment
        keyword or value.
    ValueError
        If calculated DATA segment size (as determined from the number
        of events, the number of parameters, and the number of bytes per
        data point) does not match size specified in HEADER segment
        offsets.
    Warning
        If more than one data set is detected in the same file.
    
    Notes
    -----
    The Flow Cytometry Standard (FCS) describes the de facto standard
    file format used by flow cytometry acquisition and analysis software
    to record flow cytometry data to and load flow cytometry data from a
    file. The standard dictates that each file must have the following
    segments: HEADER, TEXT, and DATA. The HEADER segment contains
    version information and byte offset values of other segments, the
    TEXT segment contains delimited key-value pairs containing
    acquisition information, and the DATA segment contains the recorded
    flow cytometry data. The file may optionally have an ANALYSIS
    segment (structurally identicaly to the TEXT segment), a
    supplemental TEXT segment (according to more recent versions of the
    standard), and user-defined OTHER segments.

    This class supports a subset of the FCS3.1 standard which should be
    backwards compatible with FCS3.0 and FCS2.0. The FCS file must be
    of the following form:
        - $MODE = 'L' (list mode; histogram mode is not supported).
        - $DATATYPE = 'I' (unsigned binary integers), 'F' (single
          precision floating point), or 'D' (double precision floating
          point). 'A' (ASCII) is not supported.
        - If $DATATYPE = 'I', $PnB % 8 = 0 (byte aligned) for all
          parameters (aka channels).
        - $BYTEORD = '4,3,2,1' (big endian) or '1,2,3,4' (little
          endian).
        - One data set per file.

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

    """
    def __init__(self, infile):
        
        self._infile = infile

        if isinstance(infile, basestring):
            f = open(infile, 'rb')
        else:
            f = infile

        self._header = read_fcs_header_segment(buf=f)

        # Import primary TEXT segment and optional supplemental TEXT segment.
        # Primary TEXT segment offsets are always specified in the HEADER
        # segment. For FCS3.0 and above, supplemental TEXT segment offsets
        # are always specified via required key-value pairs in the primary
        # TEXT segment.
        self._text, delim = read_fcs_text_segment(
            buf=f,
            begin=self._header.text_begin,
            end=self._header.text_end)

        if self._header.version in ('FCS3.0','FCS3.1'):
            stext_begin = int(self._text['$BEGINSTEXT'])   # required keyword
            stext_end = int(self._text['$ENDSTEXT'])       # required keyword
            if stext_begin and stext_end:
                stext = read_fcs_text_segment(
                    buf=f,
                    begin=stext_begin,
                    end=stext_end,
                    delim=delim)[0]
                self._text.update(stext)

        # Confirm FCS file assumptions. All queried keywords are required
        # keywords.
        if self._text['$MODE'] != 'L':
            raise NotImplementedError("only $MODE = \'L\' is supported"
                + " (detected $MODE = \'{0}\')".format(self._text['$MODE']))

        if self._text['$DATATYPE'] not in ('I','F','D'):
            raise NotImplementedError("only $DATATYPE = \'I\', \'F\', and"
                + " \'D\' are supported (detected $DATATYPE ="
                + " \'{0}\')".format(self._text['$DATATYPE']))

        D = int(self._text['$PAR']) # total number of parameters (aka channels)
        param_bit_widths = [int(self._text['$P{0}B'.format(p)])
                            for p in xrange(1,D+1)]
        if self._text['$DATATYPE'] == 'I':
            if not all(bw % 8 == 0 for bw in param_bit_widths):
                raise NotImplementedError("if $DATATYPE = \'I\', only byte"
                    + " aligned parameter bit widths (bw % 8 = 0) are"
                    + " supported (detected {0})".format(
                        ", ".join('$P{0}B={1}'.format(
                            p,self._text['$P{0}B'.format(p)])
                        for p in xrange(1,D+1)
                        if param_bit_widths[p-1] % 8 != 0)))

        if self._text['$BYTEORD'] not in ('4,3,2,1', '2,1', '1,2,3,4', '1,2'):
            raise NotImplementedError("only big endian ($BYTEORD = \'4,3,2,1\'"
                + " or \'2,1\') and little endian ($BYTEORD = \'1,2,3,4\' or"
                + " \'1,2\') are supported (detected $BYTEORD ="
                + " \'{0}\')".format(self._text['$BYTEORD']))
        big_endian = self._text['$BYTEORD'] in ('4,3,2,1', '2,1')

        if int(self._text['$NEXTDATA']):
            warnings.warn("detected (and ignoring) additional data set"
                + " ($NEXTDATA = {0})".format(self._text['$NEXTDATA']))

        # Import optional ANALYSIS segment
        if self._header.analysis_begin and self._header.analysis_end:
            # Prioritize ANALYSIS segment offsets specified in HEADER over
            # offsets specified in TEXT segment.
            self._analysis = read_fcs_text_segment(
                buf=f,
                begin=self._header.analysis_begin,
                end=self._header.analysis_end,
                delim=delim)[0]
        elif self._header.version in ('FCS3.0', 'FCS3.1'):
            analysis_begin = int(self._text['$BEGINANALYSIS'])
            analysis_end = int(self._text['$ENDANALYSIS'])
            if analysis_begin and analysis_end:
                self._analysis = read_fcs_text_segment(
                    buf=f,
                    begin=analysis_begin,
                    end=analysis_end,
                    delim=delim)[0]
            else:
                self._analysis = {}
        else:
            self._analysis = {}
        
        # Import DATA segment
        param_ranges = [int(self._text['$P{0}R'.format(p)])
                        for p in xrange(1,D+1)]
        if self._header.data_begin and self._header.data_end:
            # Prioritize DATA segment offsets specified in HEADER over
            # offsets specified in TEXT segment.
            self._data = read_fcs_data_segment(
                buf=f,
                begin=self._header.data_begin,
                end=self._header.data_end,
                datatype=self._text['$DATATYPE'],
                num_events=int(self._text['$TOT']),
                param_bit_widths=param_bit_widths,
                param_ranges=param_ranges,
                big_endian=big_endian)
        elif self._header.version in ('FCS3.0', 'FCS3.1'):
            data_begin = int(self._text['$BEGINDATA'])
            data_end = int(self._text['$ENDDATA'])
            if data_begin and data_end:
                self._data = read_fcs_data_segment(
                    buf=f,
                    begin=data_begin,
                    end=data_end,
                    datatype=self._text['$DATATYPE'],
                    num_events=int(self._text['$TOT']),
                    param_bit_widths=param_bit_widths,
                    param_ranges=param_ranges,
                    big_endian=big_endian)
            else:
                raise ValueError("DATA segment incorrectly specified")
        else:
            raise ValueError("DATA segment incorrectly specified")
        self._data.flags.writeable = False

        if isinstance(infile, basestring):
            f.close()

    # Expose attributes as read-only properties
    @property
    def infile(self):
        """
        Reference to the associated FCS file.

        """
        return self._infile

    @property
    def header(self):
        """
        ``namedtuple`` containing version information and byte offset
        values of other FCS segments in the following order:
        version : str
        text_begin : int
        text_end : int
        data_begin : int
        data_end : int
        analysis_begin : int
        analysis_end : int

        """
        return self._header

    @property
    def text(self):
        """
        Dictionary of key-value entries from TEXT segment and optional
        supplemental TEXT segment.

        """
        return self._text

    @property
    def data(self):
        """
        Unwriteable NxD numpy array describing N cytometry events
        observing D data dimensions.

        """
        return self._data

    @property
    def analysis(self):
        """
        Dictionary of key-value entries from ANALYSIS segment.

        """
        return self._analysis

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (self.infile == other.infile
                and self.header == other.header
                and self.text == other.text
                and np.array_equal(self.data, other.data)
                and self.analysis == other.analysis)
        else:
            return NotImplemented

    def __ne__(self, other):
        if isinstance(other, self.__class__):
            return not self == other
        else:
            return NotImplemented

    def __hash__(self):
        return hash((self.infile,
                     self.header,
                     frozenset(self.text.items()),
                     self.data.tobytes(),
                     frozenset(self.analysis.items())))

    def __repr__(self):
        return str(self.infile)

class FCSData(np.ndarray):
    """
    Object containing events from a flow cytometry sample.

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

    `FCSData` can read standard FCS files with versions 2.0, 3.0, and 3.1.
    Some non-standard FCS files, which store information used by `FCSData`
    in vendor-specific keywords, are also supported. Currently,
    `FCSData` can read non-standard FCS files from the following
    acquisition software/instrument combinations:
        - FCS 2.0 from CellQuestPro 5.1.1/BD FACScan Flow Cytometer

    The FCS file must be of the following form:
        - Only one data set present.
        - $MODE = 'L' (list mode; histogram mode not supported).
        - $DATATYPE = 'I' (unsigned integer), 'F' (32-bit floating point),
            or 'D' (64-bit floating point). 'A' (ASCII) is not supported.
        - If $DATATYPE = 'I', $PnB % 8 = 0 (byte-aligned) for all channels.
        - $BYTEORD = '4,3,2,1' (big endian) or '1,2,3,4' (little endian).
        - $GATE not present in TEXT, or $GATE = 0.
    
    Parameters
    ----------
    infile : str or file-like
        Reference to the associated FCS file.
    metadata : dict
        Additional channel-independent, sample-specific information.

    Attributes
    ----------
    infile : str or file-like
        Reference to associated FCS file.
    text : dict
        Dictionary of keyword-value entries from TEXT segment and optional
        supplemental TEXT segment of FCS file.
    analysis : dict
        Dictionary of keyword-value entries from ANALYSIS segment of FCS
        file.
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

    ###
    # Properties
    ###

    @property
    def channels(self):
        """
        List of channel names.

        """
        return [i['label'] for i in self.channel_info]

    def name_to_index(self, channels):
        """
        Return the channel indices for the specified channel names.

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

    @property
    def time_step(self):
        """
        Time step of the time channel.

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
        """
        Acquisition time, in seconds.

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

    ###
    # Functions overriding inherited np.ndarray functions
    ###
    # For more details, see
    # http://docs.scipy.org/doc/numpy/user/basics.subclassing.html.
    ###

    # Functions involved in the creation of new arrays

    def __new__(cls, infile, metadata={}):

        # Load FCS file
        fcs_file = FCSFile(infile)

        # Populate channel_info
        num_channels = int(fcs_file.text['$PAR'])
        channel_info = []
        for i in range(1, num_channels + 1):
            chi = {}

            # Get label
            chi['label'] = fcs_file.text.get('$P{}N'.format(i))

            if chi['label'].lower() == 'time':
                pass
            else:
                # Gain
                if 'CellQuest Pro' in fcs_file.text.get('CREATOR'):
                    chi['pmt_voltage'] = \
                        fcs_file.text.get('BD$WORD{}'.format(12 + i))
                else:
                    chi['pmt_voltage'] = fcs_file.text.get('$P{}V'.format(i))

                # Amplification type
                if '$P{}E'.format(i) in fcs_file.text:
                    if fcs_file.text['$P{}E'.format(i)] == '0,0':
                        chi['amplifier'] = 'lin'
                    else:
                        chi['amplifier'] = 'log'
                else:
                    chi['amplifier'] = None

                # Range and bins
                PnR = '$P{}R'.format(i)
                chi['range'] = [0,
                                int(fcs_file.text.get(PnR))-1,
                                int(fcs_file.text.get(PnR))]
                chi['bin_vals'] = np.arange(int(fcs_file.text.get(PnR)))
                chi['bin_edges'] = \
                    np.arange(int(fcs_file.text.get(PnR)) + 1) - 0.5

            channel_info.append(chi)

        # Call constructor of numpy array
        obj = fcs_file.data.view(cls)

        # Add attributes
        obj.infile = infile
        obj.text = fcs_file.text
        obj.analysis = fcs_file.analysis
        obj.channel_info = channel_info
        obj.metadata = metadata

        return obj

    def __array_finalize__(self, obj):
        """
        Method called after all methods of construction of the class.

        """
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

    # Functions overridden to allow string-based indexing.

    def __array_wrap__(self, out_arr, context = None):
        """
        Method called after numpy ufuncs.

        """
        if out_arr.ndim == 0:
            return None
        else:
            return np.ndarray.__array_wrap__(self, out_arr, context)

    def __getitem__(self, key):
        """
        Get an element or elements of the array.

        This function extends ``ndarray.__getitem__``.

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
        """
        Set an element or elements of the array.

        This function extends ``ndarray.__setitem__``.

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

    # Functions overridden to improve printed representation.

    def __str__(self):
        """
        Return name of FCS file.

        """
        return os.path.basename(str(self.infile))
