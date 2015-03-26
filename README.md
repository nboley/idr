Irreproducible Discovery Rate (IDR)
===

<p align="justify">The IDR (Irreproducible Discovery Rate) framework is a uniﬁed approach to measure the reproducibility of ﬁndings identiﬁed from replicate experiments and provide highly stable thresholds based on reproducibility. Unlike the usual scalar measures of reproducibility, the IDR approach creates a curve, which quantitatively assesses when the ﬁndings are no longer consistent across replicates. In layman's terms, the IDR method compares a pair of ranked lists of identifications (such as ChIP-seq peaks). These ranked lists should not be pre-thresholded i.e. they should provide identifications across the entire spectrum of high confidence/enrichment (signal) and low confidence/enrichment (noise). The IDR method then fits the bivariate rank distributions over the replicates in order to separate signal from noise based on a defined confidence of rank consistency and reproducibility of identifications i.e the IDR threshold.</p>

<p align="justify">The method was developed by <a href="http://www.personal.psu.edu/users/q/u/qul12/index.html">Qunhua Li</a> and <a href="http://www.stat.berkeley.edu/~bickel/">Peter Bickel</a>'s group and is extensively used by the ENCODE and modENCODE  projects and is part of their ChIP-seq guidelines and standards.</p>

### Building IDR

* Get the current repo
```
git clone --recursive https://github.com/nboley/idr.git
```

# Install the following dependencies
- python3
- python3 headers 
- numpy
- setuptools
- matplotlib (only required for plotting the results)

In Ubuntu 14.04+ one can run: 
(sudo) apt-get install python3-dev python3-numpy python3-setuptools python3-matplotlib

* Then run the following commands 
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

### Output format
The output format mimics the input file type, with some additional fields. 

We provide an example for narrow peak files - note that the first 6 columns
are a standard bed6, the first 10 columns are a standard narrowPeak. Broad peak 
output files are the same *except* that they do not include the the summit 
columns (10, 18, and 22 for samples with exactly 2 replicates)

1.  chrom             string  
Name of the chromosome for common peaks

2.  chromStart        int     
The starting position of the feature in the chromosome or scaffold for common 
peaks, shifted based on offset. The first base in a chromosome is numbered 0.

3.  chromEnd          int     
The ending position of the feature in the chromosome or scaffold for common 
peaks. The chromEnd base is not included in the display of the feature.

4.  name              string  
Name given to a region (preferably unique) for common peaks. Use '.' 
if no name is assigned.

5.  score             int     
Contains the scaled IDR value, int(1000*(1-IDR)). e.g. peaks with an IDR of 0 
have a score of 1000, idr 0.1 have a score 900, idr 1.0 have a score of 0.

6.  strand         [+-.]   Use '.' if no strand is assigned.

Note for columns 7-10: only the score that the IDR code used for rankign 
will be set - the remaining two columns will be -1

7.  signalValue       float   
Measurement of enrichment for the region for merged peaks

8.  p-value           float   
Merged peak p-value

9.  q-value           float   
Merged peak q-value

10. summit            int     
Merged peak summit

11. localIDR          float 
Local IDR value

12. globalIDR         float 
Global IDR value

#### 
# The remaining lines contain replicate specific information. We only present
# replicate 1 but a real file will always have at least 2 replicates

13. rep1_chromStart   int     
The starting position of the feature in the chromosome or scaffold for common 
replicate 1 peaks, shifted based on offset. The first base in a chromosome is 
numbered 0.

14. rep1_chromEnd     int     
The ending position of the feature in the chromosome or scaffold for common 
replicate 1 peaks. The chromEnd base is not included in the display of the 
feature.

15. rep1_signalValue  float   
Signal measure from replicate 1. Note that this is determined by the --rank 
option. e.g. if --rank is set to signal.value, this corresponds to the 7th 
column of the narrowPeak, whereas if it is set to p.value it corresponds to
the 8th column. 

18. rep1_summit       int     
The summit of this peak in replicate 1. 

[rep 2 data]

...

[rep N data]


### Contributors

The main contributors of IDR code:

  * Nathan Boleu        - Kundaje Lab, Dept. of Genetics, Stanford University
  * Anshul Kundaje      - Assistant Professor, Dept. of Genetics, Stanford University
  * Peter J. Bickel     - Professor, Dept. of Statistics, University of California at Berkeley

### Issues

If you notice any problem with the code, please file an issue over [here](https://github.com/nboley/idr/issues)

### Differences 
