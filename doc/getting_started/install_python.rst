Installing FlowCal in an Existing Python Evironment
=======================================================

Python (2.7, 3.6, or 3.7) is required, along with ``pip`` and ``setuptools``. The easiest way is to install ``FlowCal`` is to use ``pip``::

	pip install FlowCal

This should take care of all the requirements automatically. Linux and Mac OSX users may need to request administrative permissions by preceding this command with ``sudo``.

Alternatively, download ``FlowCal`` from `here <https://github.com/taborlab/FlowCal/archive/master.zip>`_. Next, make sure that the following Python packages are present:

* ``packaging`` (>=16.8)
* ``six`` (>=1.10.0)
* ``numpy`` (>=1.8.2)
* ``scipy`` (>=0.14.0)
* ``matplotlib`` (>=2.0.0)
* ``palettable`` (>=2.1.1)
* ``scikit-image`` (>=0.10.0)
* ``scikit-learn`` (>=0.16.0)
* ``pandas`` (>=0.16.1)
* ``xlrd`` (>=0.9.2)
* ``XlsxWriter`` (>=0.5.2)

If you have ``pip``, a ``requirements.txt`` file is provided, such that the required packages can be installed by running::

	pip install -r requirements.txt

To install ``FlowCal``, run the following in ``FlowCal``'s root directory::

	python setup.py install

Again, some users may need to precede the previous commands with ``sudo``.

.. note::
	**Ubuntu/Linux Mint**: ``FlowCal`` might need more recent versions of some python packages than the ones provided via ``apt``. To upgrade these, some non-python packages need to be installed in your system. On freshly installed systems, the following packages may need to be manually installed:

	* ``gcc``
	* ``g++``
	* ``gfortran``
	* ``libblas-dev``
	* ``liblapack-dev``
	* ``libfreetype6-dev``
	* ``python-dev``
	* ``python-tk``
	* ``python-pip``

	All of these can be installed using::

		sudo apt install <package-name>

	Next, ``pip`` should be upgraded by using::

		sudo pip install --upgrade pip

	After this, you may install ``FlowCal`` by following the steps above.