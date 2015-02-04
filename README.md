Irreproducible Discovery Rate (IDR)
===

<p align="justify">The IDR (Irreproducible Discovery Rate) framework is a uniﬁed approach to measure the reproducibility of ﬁndings identiﬁed from replicate experiments and provide highly stable thresholds based on reproducibility. Unlike the usual scalar measures of reproducibility, the IDR approach creates a curve, which quantitatively assesses when the ﬁndings are no longer consistent across replicates. In layman's terms, the IDR method compares a pair of ranked lists of identifications (such as ChIP-seq peaks). These ranked lists should not be pre-thresholded i.e. they should provide identifications across the entire spectrum of high confidence/enrichment (signal) and low confidence/enrichment (noise). The IDR method then fits the bivariate rank distributions over the replicates in order to separate signal from noise based on a defined confidence of rank consistency and reproducibility of identifications i.e the IDR threshold.</p>

<p align="justify">The method was developed by <a href="http://www.personal.psu.edu/users/q/u/qul12/index.html">Qunhua Li</a> and <a href="http://www.stat.berkeley.edu/~bickel/">Peter Bickel</a>'s group and is extensively used by the ENCODE and modENCODE  projects and is part of their ChIP-seq guidelines and standards.</p>

### Building IDR

* Get the current repo
```
git clone --recursive https://github.com/nboley/idr.git
```

* Then follow the commands below 
```
(sudo) python3 setup.py install
```

### Usage

List all the options
 
```
idr -h
```

Sample idr run using test peak files in the repo

```
idr --samples ../idr/test/data/peak1 ../idr/test/data/peak2
```

The main contributors of IDR code:

  * Nathan Boleu        - Kundaje Lab, Dept. of Genetics, Stanford University
  * Anshul Kundaje      - Assistant Professor, Dept. of Genetics, Stanford University
  * Peter J. Bickel     - Professor, Dept. of Statistics, University of California at Berkeley

### Issues

If you notice any problem with the code, please file an issue over [here](https://github.com/nboley/idr/issues)
