Reading Flow Cytometry Data
===========================

This tutorial focuses on how to open FCS files and manipulate the data therein using ``FlowCal``.

To start, navigate to the ``examples/FCFiles`` directory included with FlowCal, and open a ``python`` session therein. Then, import ``FlowCal`` as with any other python module.

>>> import FlowCal

FCS files are standard files in which the events resulting from measuring a flow cytometry sample are stored. Normally, one FCS file corresponds to one flow cytometry sample.

The object :class:`FlowCal.io.FCSData` allows a user to open an FCS file. The following instruction opens the file ``data_001.fcs``, loads the information into an ``FCSData`` object, and assigns it to a variable ``s``.

>>> s = FlowCal.io.FCSData('data_001.fcs')

An ``FCSData`` object is a 2D ``numpy`` array with a few additional features. The first dimension indexes the event number, and the second dimension indexes the flow cytometry channel (or "parameter", as called by the FCS standard). We can see the number of events and channels using the standard ``numpy``'s ``shape`` property:

>>> print s.shape
(15319, 8)

As with any ``numpy`` array, we can slice an ``FCSData`` object. For example, let's obtain the first 100 events.

>>> s_sub = s[:100]
>>> print s_sub.shape
(100, 8)

Note that the product of slicing an FCSData object is also an FCSData object. We can also get all the events in a subset of channels by slicing in the second dimension.

>>> s_sub_ch = s[:, [3, 4, 5]]
>>> print s_sub_ch.shape
(15319, 3)

However, it is not immediately obvious what channels we are getting. Fortunately, the ``FCSData`` object contains some additional information about the acquisition settings. In particular, we can check the name of the channels with the ``channels`` property.

>>> print s.channels
('TIME', 'FSC', 'SSC', 'FL1', 'FL2', 'FL3', 'SSCW', 'SSCA')
>>> print s_sub_ch.channels
('FL1', 'FL2', 'FL3')

It turns out that ``s_sub_ch`` contains the fluorescence channels ``FL1``, ``FL2``, and ``FL3``.

One of the most practical features of an ``FCSData`` object is the ability to slice channels using their name. For example, if we want the fluorescence channels we can use the following.

>>> s_sub_ch_2 = s[:, ['FL1', 'FL2', 'FL3']]
>>> print s_sub_ch_2.channels
('FL1', 'FL2', 'FL3')

This is completely equivalent to indexing with integers.

>>> import numpy as np
>>> np.all(s_sub_ch == s_sub_ch_2)
True

``FCSData`` contains more acquisition information, such as the acquisition time, amplifier type, and the detector voltage of each channel. For more information, consult the documentation of :class:`FlowCal.io.FCSData`.
