import os, sys
import numpy
from setuptools import setup, Extension, find_packages

try:
    from Cython.Build import cythonize
    extensions = cythonize([
        Extension("idr.inv_cdf", 
                  ["idr/inv_cdf.pyx", ],
                  include_dirs=[numpy.get_include()]),
    ])
except ImportError:
    extensions = [
        Extension("idr.inv_cdf", 
                  ["idr/inv_cdf.c", ],
                  include_dirs=[numpy.get_include()]),
    ]

def main():
    if sys.version_info.major <= 2:
        raise ValueError( "IDR requires Python version 3 or higher" )
    import idr
    setup(
        name = "idr",
        version = idr.__version__,

        author = "Nathan Boley",
        author_email = "npboley@gmail.com",

        ext_modules = extensions,     

        install_requires = [ 'scipy', 'numpy'  ],

        extra_requires=['matplotlib'],

        packages= ['idr',],

        scripts =  ['./bin/idr',],

        description = ("IDR is a method for measuring the reproducibility of " + 
                       "replicated ChIP-seq type experiments and providing a " +
                       "stable measure of the reproducibility of identified " +
                       "peaks."),

        license = "GPL2",
        keywords = "IDR",
        url = "https://github.com/nboley/idr",

        long_description="""
    The IDR (Irreproducible Discovery Rate) framework is a unified approach to measure the reproducibility of findings identified from replicate experiments and provide highly stable thresholds based on reproducibility. Unlike the usual scalar measures of reproducibility, the IDR approach creates a curve, which quantitatively assesses when the findings are no longer consistent across replicates. In layman's terms, the IDR method compares a pair of ranked lists of identifications (such as ChIP-seq peaks). These ranked lists should not be pre-thresholded i.e. they should provide identifications across the entire spectrum of high confidence/enrichment (signal) and low confidence/enrichment (noise). The IDR method then fits the bivariate rank distributions over the replicates in order to separate signal from noise based on a defined confidence of rank consistency and reproducibility of identifications i.e the IDR threshold.

    The method was developed by Qunhua Li and Peter Bickel's group and is extensively used by the ENCODE and modENCODE projects and is part of their ChIP-seq guidelines and standards.
    """,
        classifiers=[
            "Programming Language :: Python :: 3",
            "Development Status :: 4 - Beta",
            "Topic :: Scientific/Engineering :: Bio-Informatics",
            "License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)",
        ],
    )

if __name__ == '__main__':
    main()
