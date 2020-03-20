# K-Means Clustering

This is a Python implementation of the K-Means clustering algorithm using K-Means++ centroid initialization strategy.

K-Means clustering is a form of unsupervised learning where you have data and know its attributes but not its classification. It relies on the fact that similar data will stick together to form clusters, and the KM algorithm is meant to identify these clusters.

Here's an example visualization:

<img src="https://github.com/stratzilla/k-means-clustering/blob/master/images/perfect.gif" width=35%/>

# Dependencies

- pandas
- matplotlib
- Python (3.6+)
- GNU/Linux

# Execution

You can clone these files to your computer with the below:

` $ git clone https://github.com/stratzilla/k-means-clustering`

Execute the script as the below:

` $ ./k-means.py <File> <Type> <K> <Epochs>`

Where the following are the arguments:

```
 <File> -- the data to perform k-means clustering upon
 <Type> -- the type of metric space to use:
        -- 1 - Euclidean Metric
        -- 2 - Manhattan Metric
        -- 3 - Chebyshev Metric
 <K> -- k-parameter, or how many clusters [2, 25]
 <Epochs> -- how many epochs for k-means to perform [1, 100]
```

Each epoch frame will be saved as an image in `./plots/`. This directory will be made if it does not exist.

Using no arguments will remind the user of this. If the arguments are appropriate, the output to console will be the Dunn Index of the clustering.

# Data

In `/data` are a few data sets as found <a href="http://cs.joensuu.fi/sipu/datasets/">here</a>, each with a nomimal `k=15`.

# Tutorial

I wrote a tutorial to go alongside this repository: <a href="https://github.com/stratzilla/k-means-tutorial">here</a>!
