#!/usr/bin/python

# setuptools setup module.
#
# Based on setup.py on https://github.com/pypa/sampleproject.

from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path
# To scrape version information
import re

def find_version(file_path):
    """
    Scrape version information from specified file path.

    """
    with open(file_path, 'r') as f:
        file_contents = f.read()
    version_match = re.search(r"^__version__\s*=\s*['\"]([^'\"]*)['\"]",
                              file_contents, re.M)
    if version_match:
        return version_match.group(1)
    else:
        raise RuntimeError("unable to find version string")

here = path.abspath(path.dirname(__file__))

# Get long description
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='FlowCal',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version=find_version(path.join(here, 'FlowCal', '__init__.py')),

    description='Flow Cytometry Calibration Library',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/taborlab/FlowCal',

    # Author details
    author='Sebastian Castillo-Hair',
    author_email='castillohair@gmail.com',

    # Choose your license
    license='MIT',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 5 - Production/Stable',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],

    # What does your project relate to?
    keywords='flow cytometry',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=['FlowCal'],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=['packaging>=16.8',
                      'six>=1.10.0',
                      'numpy>=1.9.0',
                      'scipy>=0.19.0',
                      'matplotlib>=2.0.0',
                      'scikit-image>=0.10.0',
                      'scikit-learn>=0.16.0',
                      'pandas>=0.23.0',
                      'xlrd>=0.9.2,<2.0.0',
                      'openpyxl>=2.2.0'],

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    # extras_require={
    #     'dev': ['check-manifest'],
    #     'test': ['coverage'],
    # },

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    package_data={
        '.': ['CONTRIBUTE.rst'],
        'FlowCal': [],
    },

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files # noqa
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    # data_files=[('my_data', ['data/data_file'])],

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points = {
        'console_scripts': [
            'flowcal=FlowCal.excel_ui:run_command_line',
        ]
    },
)
