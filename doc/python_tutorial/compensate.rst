Compensating Flow Cytometry Data
================================

This tutorial focuses on how to perform multi-color compensation on flow cytometry data using ``FlowCal``, particularly by using the module :mod:`FlowCal.compensate`.

The dataset in ``examples`` includes nine cell samples from a strain containing both sfGFP and mCherry genes, exposed to nine different inducer (aTc) levels that result in changes of sfGFP and mCherry expression. Ideally, we would be able to observe sfGFP in fluorescence channel FL1 and mCherry in FL2 without any crosstalk or interference. In reality, the signal observed in each channel is the sum of the following three components:

* The channel's "main" fluorophore (in this case sfGFP for FL1 and mCherry for FL2).
* "Bleedthrough" from all other fluorophores (sfGFP for FL2 and mCherry for FL1, although the latter, in this particular case, is close to zero).
* Cell autofluorescence.

Here, we show how to perform multi-color compensation to extract the true fluorophore signal from each channel. To allow this, our dataset also includes the following controls (folder ``controls``):

* One no-fluorophore control (NFC) taken from a strain contaning no fluorophores.
* Two single-fluorophore controls (SFCs) taken from strains containing a single fluorophore each - in this case, sfGFP and mCherry.

Multi-color compensation requires all fluorescence data to be in units proportional to the fluorophore's emission intensity. In most cytometers, fluorescence in a.u. satisfies this requirement. However, older instruments can give a signal that is slightly nonlinear with respect to intensity. MEF calibration can compensate for this, and thus it is recommended to run compensation after calibration. In addition, calibration allows samples and controls taken at different cytometer settings to be used for compensation. Please refer to the :doc:`calibration tutorial </python_tutorial/mef>` for more information. In order to keep this tutorial focused, compensation here will be completely performed in a.u.

Loading data and performing gating and calibration
--------------------------------------------------

To start, navigate to the ``examples`` directory included with FlowCal, and open a ``python`` session therein. Then, import ``numpy``, ``matplotlib``, and ``FlowCal``.

>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> import FlowCal

First, we load the experimental sample files and perform density gating:

>>> samples_filenames = ['FCFiles/sample029.fcs',
...                      'FCFiles/sample030.fcs',
...                      'FCFiles/sample031.fcs',
...                      'FCFiles/sample032.fcs',
...                      'FCFiles/sample033.fcs',
...                      'FCFiles/sample034.fcs',
...                      'FCFiles/sample035.fcs',
...                      'FCFiles/sample036.fcs',
...                      'FCFiles/sample037.fcs']
>>> # The list ``samples`` will store processed, transformed data of cell samples
>>> samples = []
>>> # Iterate over cell sample filenames
>>> for sample_id, sample_filename in enumerate(samples_filenames):
...     # Load file
...     sample = FlowCal.io.FCSData(sample_filename)
...     # Transform data to RFI
...     sample = FlowCal.transform.to_rfi(sample)
...     # Apply density gating
...     sample = FlowCal.gate.density2d(
...         data=sample,
...         channels=['FSC','SSC'],
...         gate_fraction=0.85)
...     # Save
...     samples.append(sample)

Next, we do the same for the control samples:

>>> # File names
>>> nfc_sample_filename = 'FCFiles/controls/nfc/sample004.fcs'
>>> sfc1_sample_filename = 'FCFiles/controls/sfc1/sample007.fcs'
>>> sfc2_sample_filename = 'FCFiles/controls/sfc2/sample019.fcs'
>>> # Load files
>>> nfc_sample = FlowCal.io.FCSData(nfc_sample_filename)
>>> sfc1_sample = FlowCal.io.FCSData(sfc1_sample_filename)
>>> sfc2_sample = FlowCal.io.FCSData(sfc2_sample_filename)
>>> # Transform to RFI
>>> nfc_sample = FlowCal.transform.to_rfi(nfc_sample)
>>> sfc1_sample = FlowCal.transform.to_rfi(sfc1_sample)
>>> sfc2_sample = FlowCal.transform.to_rfi(sfc2_sample)
>>> # Perform density gating
>>> nfc_sample = FlowCal.gate.density2d(
...     data=nfc_sample,
...     channels=['FSC','SSC'],
...     gate_fraction=0.85)
>>> sfc1_sample = FlowCal.gate.density2d(
...     data=sfc1_sample,
...     channels=['FSC','SSC'],
...     gate_fraction=0.85)
>>> sfc2_sample = FlowCal.gate.density2d(
...     data=sfc2_sample,
...     channels=['FSC','SSC'],
...     gate_fraction=0.85)

Observing bleedthrough
----------------------

Let's now look at the FL1 and FL2 fluorescence of the SFCs. As a reminder, each control should contain only one fluorophore (sfGFP or mCherry). Ideally, the sfGFP SFC should only produce signal in FL1, and the mCherry SFC should only produce FL2 signal.

>>> # Obtain mean autofluorescence in FL1 and FL2 to plot in all panels
>>> autofl = FlowCal.stats.mean(nfc_sample, channels=['FL1', 'FL2'])
>>> # Plot controls
>>> samples_to_plot = [
...     nfc_sample,
...     sfc1_sample,
...     sfc2_sample,
... ]
>>> samples_titles = [
...     "No-fluroescence control",
...     "SFC, sfGFP",
...     "SFC, mCherry",
... ]
>>> plt.figure(figsize=(9, 3))
>>> for plot_id, (sample_to_plot, sample_title) in \
...         enumerate(zip(samples_to_plot, samples_titles)):
...     plt.subplot(1, 3, 1 + plot_id)
...     # Density plot of sample
...     FlowCal.plot.density2d(
...         sample_to_plot,
...         channels=['FL1', 'FL2'],
...         mode='scatter')
...     # Plot autofluorescence lines
...     plt.axvline(autofl[0], color='gray')
...     plt.axhline(autofl[1], color='gray')
...     # Set the axes identically accross all samples
...     plt.gca().set_xscale('logicle', T=2e4)
...     plt.gca().set_yscale('logicle', T=2e4)
...     plt.title(sample_title)
>>> plt.tight_layout()
>>> plt.show()

.. image:: /_static/img/python_tutorial/python_tutorial_compensate_1.png

While the mCherry SFC produces signal above autofluorescence in FL2 only (right plot), the sfGFP SFC results in signals in both channels (middle plot). While the sfGFP-induced FL2 signal is small in this case, it can make it difficult to resolve mCherry signals that are small to begin with. To see this, let's analyze the experimental samples:

>>> # aTc concentration of each cell sample, in ng/mL.
>>> atc = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 7.5, 20])
>>> # Plot violins of experimental samples as a function of aTc
>>> # Plot the NFC to indicate the minimum possible fluorescence
>>> plt.figure(figsize=(8, 3.5))
>>> plt.subplot(1, 2, 1)
>>> FlowCal.plot.violin_dose_response(
...     data=samples,
...     channel='FL1',
...     positions=atc,
...     min_data=nfc_sample,
...     xlabel='aTc Concentration (ng/mL)',
...     xscale='log',
...     yscale='log',
...     ylim=(1e0,1e3),
...     violin_width=0.12,
...     violin_kwargs={'facecolor': 'tab:green',
...                    'edgecolor':'black'},
... )
>>> plt.ylabel('FL1 Fluorescence (MEFL)')
>>> plt.subplot(1, 2, 2)
>>> FlowCal.plot.violin_dose_response(
...     data=samples,
...     channel='FL2',
...     positions=atc,
...     min_data=nfc_sample,
...     xlabel='aTc Concentration (ng/mL)',
...     xscale='log',
...     yscale='log',
...     ylim=(1e0,2e3),
...     violin_width=0.12,
...     violin_kwargs={'facecolor': 'tab:orange',
...                    'edgecolor':'black'},
... )
>>> plt.ylabel('FL2 Fluorescence (MEPE)')
>>> plt.tight_layout()
>>> plt.show()

.. image:: /_static/img/python_tutorial/python_tutorial_compensate_2.png

As we can see here, at low inducer (aTc) levels, FL2 fluorescence (right plot) is small but non-zero. This may be the result of a phenomenon called "leakiness", where the output of a genetic system is not completely off in a situation where it should be. However, given that sfGFP fluorescence is high at the same inducer levels (FL1, left), it is hard to know whether the observed FL2 signal is due to leaky mCherry expression or bleedthrough from sfGFP.

Eliminating bleedthrough via compensation
-----------------------------------------

In ``FlowCal``, compensation is performed by creating a transformation function using ``compensate.get_transform_fxn()``, which in turn requires data from the control samples. The resulting transformation function can be used afterwards to compensate data from other samples.

>>> # Create compensation function
>>> compensation_fxn = FlowCal.compensate.get_transform_fxn(
...     nfc_sample,
...     [sfc1_sample, sfc2_sample],
...     ['FL1', 'FL2'],
... )
>>> # Apply compensation to the samples and the NFC
>>> samples_compensated = [compensation_fxn(s, ['FL1', 'FL2']) for s in samples]
>>> nfc_sample_compensated = compensation_fxn(nfc_sample, ['FL1', 'FL2'])
<BLANKLINE>
>>> # Plot violins with compensated data
>>> plt.figure(figsize=(8, 3.5))
>>> plt.subplot(1, 2, 1)
>>> FlowCal.plot.violin_dose_response(
...     data=samples_compensated,
...     channel='FL1',
...     positions=atc,
...     min_data=nfc_sample_compensated,
...     xlabel='aTc Concentration (ng/mL)',
...     xscale='log',
...     yscale='logicle',
...     ylim=(-3e1, 1e3),
...     violin_width=0.12,
...     violin_kwargs={'facecolor': 'tab:green',
...                    'edgecolor':'black'},
...                    )
>>> plt.ylabel('FL1 Fluorescence (MEFL)')
>>> plt.subplot(1, 2, 2)
>>> FlowCal.plot.violin_dose_response(
...     data=samples_compensated,
...     channel='FL2',
...     positions=atc,
...     min_data=nfc_sample_compensated,
...     xlabel='aTc Concentration (ng/mL)',
...     xscale='log',
...     yscale='logicle',
...     ylim=(-3e1, 2e3),
...     violin_width=0.12,
...     violin_kwargs={'facecolor': 'tab:orange',
...                    'edgecolor':'black'},
...                    )
>>> plt.ylabel('FL2 Fluorescence (MEPE)')
>>> plt.tight_layout()
>>> plt.show()

.. image:: /_static/img/python_tutorial/python_tutorial_compensate_3.png

Here we can observe two changes. First, the NFC (black violin) is now centered around zero. This is an effect of removing the autofluorescence component, measured from the NFC itself, from the FL2 signal during compensation. Second, FL2 violins at low inducer levels are now centered around zero as well. Because both autofluorescence and bleedthrough from sfGFP were removed by the compensation process, the fact that the remaining FL2 signal is zero shows that the output of the genetic system driving mCherry is not leaky as we hypothesized above.

A final note about compensation: most flow-cytometry software packages perform compensation without taking into account autofluorescence subtraction. In fact, one can mimic this procedure in ``FlowCal`` by calling ``compensate.get_transform_fxn()`` without an NFC:

>>> compensation_fxn = FlowCal.compensate.get_transform_fxn(
...     None,
...     [sfc1_sample, sfc2_sample],
...     ['FL1', 'FL2'],
... )

Differences resulting from the usage of an NFC are negligible when sample fluorescence is much greater than autofluorescence. This may happen when the fluorescence signal is actually really large, or with modern instruments where an NFC histogram would be centered around zero (although in our experience this does not always happen perfectly). However, in cases where sample fluorescence is close to autofluorescence, ignoring the NFC can lead to non-sensical results where low fluorescence levels are brought down below autofluorescence. In fact, if we run this compensation method with our samples we obtain the following violins:

.. image:: /_static/img/python_tutorial/python_tutorial_compensate_4.png

We recommend using both NFCs and SFCs when possible, ideally acquired simultaneously with the experimental samples.