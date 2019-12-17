---
title: "Customer Segmentation using K-Means Clustering"
author: "Subha Yoganandan"
date: "12/12/2019"
output: 
  html_document:
    keep_md: true
---
Market segmentation is the process of dividing a broad consumer or business market, normally consisting of existing and potential customers, into sub-groups of consumers (known as segments) based on some type of shared characteristics. In dividing or segmenting markets, researchers typically look for common characteristics such as shared needs, common interests, similar lifestyles or even similar demographic profiles. The overall aim of segmentation is to identify high yield segments – that is, those segments that are likely to be the most profitable or that have growth potential – so that these can be selected for special attention (i.e. become target markets). [Source: Wikipedia]

First a little introduction of the K-Means algorithm which I am going to implement here. The following process describes how this method is implemented.

1. k initial "means" are randomly generated within the data domain.

2. k clusters are created by associating every observation with the nearest mean.

3. The centroid of each of the k clusters becomes the new mean.

4. Steps 2 and 3 are repeated until convergence has been reached.

[Source: Wikipedia]

In this blog I am looking to implement a K-Means clustering algorithm to a dataset from the trusted UCI Machine Learning Repository. The Wholesale Customers dataset consists of 8 attributes and 440 Instances. It can be downloaded from here.

Attribute Information:

1) FRESH: annual spending (m.u.) on fresh products (Continuous);
2) MILK: annual spending (m.u.) on milk products (Continuous);
3) GROCERY: annual spending (m.u.)on grocery products (Continuous);
4) FROZEN: annual spending (m.u.)on frozen products (Continuous)
5) DETERGENTS_PAPER: annual spending (m.u.) on detergents and paper products (Continuous)
6) DELICATESSEN: annual spending (m.u.)on and delicatessen products (Continuous);
7) CHANNEL: customersâ€™ Channel - Horeca (Hotel/Restaurant/CafÃ©) or Retail channel (Nominal)
8) REGION: customersâ€™ Region â€“ Lisnon, Oporto or Other (Nominal)
Descriptive Statistics:

(Minimum, Maximum, Mean, Std. Deviation)
FRESH ( 3, 112151, 12000.30, 12647.329)
MILK (55, 73498, 5796.27, 7380.377)
GROCERY (3, 92780, 7951.28, 9503.163)
FROZEN (25, 60869, 3071.93, 4854.673)
DETERGENTS_PAPER (3, 40827, 2881.49, 4767.854)
DELICATESSEN (3, 47943, 1524.87, 2820.106)

REGION Frequency
Lisbon 77
Oporto 47
Other Region 316
Total 440

CHANNEL Frequency
Horeca 298
Retail 142
Total 440

Now let us jump straight into the code.


```r
#Call necessary libraries
library(corrplot)
```

```
## corrplot 0.84 loaded
```

```r
#Read the input file

wholecust <- read.csv(url("http://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv"))
```

Now, let's understand the dataset by having a peak of the data


```r
head(wholecust)
```

```
##   Channel Region Fresh Milk Grocery Frozen Detergents_Paper Delicassen
## 1       2      3 12669 9656    7561    214             2674       1338
## 2       2      3  7057 9810    9568   1762             3293       1776
## 3       2      3  6353 8808    7684   2405             3516       7844
## 4       1      3 13265 1196    4221   6404              507       1788
## 5       2      3 22615 5410    7198   3915             1777       5185
## 6       2      3  9413 8259    5126    666             1795       1451
```

We can also get a useful summary from the dataset


```r
summary(wholecust)
```

```
##     Channel          Region          Fresh             Milk      
##  Min.   :1.000   Min.   :1.000   Min.   :     3   Min.   :   55  
##  1st Qu.:1.000   1st Qu.:2.000   1st Qu.:  3128   1st Qu.: 1533  
##  Median :1.000   Median :3.000   Median :  8504   Median : 3627  
##  Mean   :1.323   Mean   :2.543   Mean   : 12000   Mean   : 5796  
##  3rd Qu.:2.000   3rd Qu.:3.000   3rd Qu.: 16934   3rd Qu.: 7190  
##  Max.   :2.000   Max.   :3.000   Max.   :112151   Max.   :73498  
##     Grocery          Frozen        Detergents_Paper    Delicassen     
##  Min.   :    3   Min.   :   25.0   Min.   :    3.0   Min.   :    3.0  
##  1st Qu.: 2153   1st Qu.:  742.2   1st Qu.:  256.8   1st Qu.:  408.2  
##  Median : 4756   Median : 1526.0   Median :  816.5   Median :  965.5  
##  Mean   : 7951   Mean   : 3071.9   Mean   : 2881.5   Mean   : 1524.9  
##  3rd Qu.:10656   3rd Qu.: 3554.2   3rd Qu.: 3922.0   3rd Qu.: 1820.2  
##  Max.   :92780   Max.   :60869.0   Max.   :40827.0   Max.   :47943.0
```

Let us now explore the dataset more and find out strong correlations among variables.


```r
#Identify strong correlations
w <- cor(wholecust)
corrplot(w, method="number")
```

![](cust_segmentation_kmeans_files/figure-html/unnamed-chunk-4-1.png)<!-- -->

From the above diagram we can see that there is strong correlation among the Detergents_Paper and Grocery.

Now we need to find the optimal number of cluster K. Determining the number of clusters in a data set, a quantity often labelled k as in the k-means algorithm, is a frequent problem in data clustering, and is a distinct issue from the process of actually solving the clustering problem. The elbow method looks at the percentage of variance explained as a function of the number of clusters: One should choose a number of clusters so that adding another cluster doesn't give much better modeling of the data. More precisely, if one plots the percentage of variance explained by the clusters against the number of clusters, the first clusters will add much information (explain a lot of variance), but at some point the marginal gain will drop, giving an angle in the graph. The number of clusters is chosen at this point, hence the "elbow criterion".  [Source: Wikipedia]


```r
set.seed(123)
k.max <- 15
data <- as.matrix(scale(wholecust[,(3:8)]))

wss <- sapply(1:k.max, 
 function(k){kmeans(data, k, nstart=50,iter.max = 15 )$tot.withinss})

plot(1:k.max, wss,
 type="b", pch = 19, frame = FALSE, 
 xlab="Number of clusters",
 ylab="Sum of squares")
```

![](cust_segmentation_kmeans_files/figure-html/unnamed-chunk-5-1.png)<!-- -->

From the above plot we can see that 3 or 5 is the optimal number of clusters, as we can see that after these numbers the curve remains less changing.

Let us plot out the clusters for both the scenarios


```r
kmm <- kmeans(wholecust[,(3:8)], 3)
kmm
```

```
## K-means clustering with 3 clusters of sizes 330, 60, 50
## 
## Cluster means:
##      Fresh      Milk   Grocery   Frozen Detergents_Paper Delicassen
## 1  8253.47  3824.603  5280.455 2572.661         1773.058   1137.497
## 2 35941.40  6044.450  6288.617 6713.967         1039.667   3049.467
## 3  8000.04 18511.420 27573.900 1996.680        12407.360   2252.020
## 
## Clustering vector:
##   [1] 1 1 1 1 2 1 1 1 1 3 1 1 2 1 2 1 1 1 1 1 1 1 2 3 2 1 1 1 3 2 1 1 1 2 1 1 2
##  [38] 1 3 2 2 1 1 3 1 3 3 3 1 3 1 1 2 1 2 1 3 1 1 1 1 3 1 1 1 3 1 1 1 1 1 1 1 1
##  [75] 1 1 1 3 1 1 1 1 1 1 1 3 3 2 1 2 1 1 3 1 1 1 1 1 1 1 1 1 1 2 1 1 1 1 1 3 1
## [112] 3 1 1 1 1 1 1 1 1 1 1 1 1 2 2 1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 2 2 1 1 3 1 1
## [149] 1 2 1 1 1 1 1 3 1 1 1 1 1 1 1 3 1 3 1 1 1 1 1 3 1 3 1 1 2 1 1 1 1 2 1 2 1
## [186] 1 1 1 1 1 1 1 1 1 1 1 2 1 1 1 3 3 2 1 1 3 1 1 1 3 1 3 1 1 1 1 3 1 1 1 1 1
## [223] 1 1 1 1 1 1 1 1 1 1 2 1 1 1 1 1 1 2 2 2 1 1 1 1 1 1 1 1 1 3 1 2 1 2 1 1 2
## [260] 2 1 1 2 1 1 3 3 1 3 1 1 1 1 2 1 1 2 1 1 1 1 1 2 2 2 2 1 1 1 2 1 1 1 1 1 1
## [297] 1 1 1 1 1 3 1 1 3 1 3 1 1 3 1 2 3 1 1 1 1 1 1 3 1 1 1 1 2 2 1 1 1 1 1 3 1
## [334] 3 1 2 1 1 1 1 1 1 1 3 1 1 1 2 1 3 1 3 1 3 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
## [371] 2 1 1 1 1 1 1 2 1 1 2 1 2 1 3 1 1 1 1 1 1 1 1 2 1 1 1 1 1 1 1 2 2 2 1 1 2
## [408] 3 1 1 1 1 1 1 1 1 1 1 3 1 1 1 2 1 1 1 1 2 1 1 1 1 1 1 1 2 2 3 1 1
## 
## Within cluster sum of squares by cluster:
## [1] 28184318853 25765310312 26382784678
##  (between_SS / total_SS =  49.0 %)
## 
## Available components:
## 
## [1] "cluster"      "centers"      "totss"        "withinss"     "tot.withinss"
## [6] "betweenss"    "size"         "iter"         "ifault"
```

Now let's implement K means for k=5


```r
 kmm <- kmeans(wholecust[,(3:8)], 5)
 kmm
```

```
## K-means clustering with 5 clusters of sizes 104, 10, 223, 80, 23
## 
## Cluster means:
##       Fresh      Milk   Grocery   Frozen Detergents_Paper Delicassen
## 1 21200.058  3886.423  5138.933 4119.856        1131.5192  1690.3365
## 2 21263.700 37443.300 46710.600 6287.200       21699.4000  8743.3000
## 3  6052.812  3266.314  4092.054 2459.682        1214.1300   990.6099
## 4  4738.762 11609.013 18198.775 1515.388        8003.7750  1603.8000
## 5 49296.087  4983.783  5590.304 8285.783         962.2609  2543.6957
## 
## Clustering vector:
##   [1] 3 3 3 3 1 3 3 3 3 4 4 3 1 1 1 3 4 3 1 3 1 3 1 2 1 1 3 1 4 5 1 3 1 1 3 3 1
##  [38] 4 4 5 1 1 4 4 3 4 4 2 3 4 3 3 5 4 1 3 4 4 3 3 3 2 3 4 3 2 3 1 3 3 1 1 3 1
##  [75] 3 1 3 4 3 3 3 4 4 1 3 2 2 5 3 1 3 3 2 1 4 3 3 3 3 3 4 4 3 5 1 1 4 4 3 4 3
## [112] 4 1 1 1 3 3 3 1 3 1 3 3 3 5 5 1 1 3 5 3 3 1 3 3 3 3 3 3 3 1 1 5 3 1 4 3 3
## [149] 3 1 1 3 1 3 3 4 4 1 3 4 3 3 1 4 3 4 3 3 3 3 4 4 3 4 3 4 5 3 3 3 3 5 4 2 3
## [186] 3 3 3 4 4 1 3 3 4 3 1 1 3 3 3 4 4 1 3 3 4 3 3 3 4 1 2 3 3 4 4 4 1 4 3 1 3
## [223] 3 3 3 3 1 3 3 3 3 3 1 3 1 3 3 1 3 5 1 1 1 3 3 4 3 3 1 3 3 4 3 1 3 1 3 3 5
## [260] 5 3 3 1 3 4 4 4 1 4 1 3 3 3 5 3 3 1 3 3 1 3 3 5 1 5 5 3 1 1 5 3 3 3 4 1 3
## [297] 1 3 3 3 1 4 3 4 4 4 4 1 3 4 3 1 4 3 3 4 3 3 3 4 3 3 1 3 1 5 3 3 1 3 3 4 1
## [334] 2 1 1 3 3 3 3 3 3 3 4 3 3 4 1 3 4 3 4 3 4 1 3 1 4 3 3 1 3 3 3 3 3 3 3 1 3
## [371] 5 1 3 1 3 3 4 5 3 3 1 1 1 3 4 3 3 1 3 3 3 3 3 1 3 3 4 3 3 3 3 1 1 1 1 3 1
## [408] 4 3 3 3 3 3 3 3 3 4 3 4 3 4 1 1 1 1 3 4 1 3 3 4 3 1 3 1 1 5 4 3 3
## 
## Within cluster sum of squares by cluster:
## [1]  8521349738 14108802241 10060038988  8835879467 11679101316
##  (between_SS / total_SS =  66.2 %)
## 
## Available components:
## 
## [1] "cluster"      "centers"      "totss"        "withinss"     "tot.withinss"
## [6] "betweenss"    "size"         "iter"         "ifault"
```

Let us closely look at the cluster means of both scenarios:

Scenario 1: k = 3

Cluster 1 - highest fresh-products.
Cluster 2 - low spenders.
Cluster 3 - highest milk, grocery, detergents_papers spenders.

Scenario 2: k = 5

Cluster 1 - low spenders
Cluster 2 - highest Fresh spenders
Cluster 3 - mediocre spenders
Cluster 4 - low spenders
Cluster 5 - mediocre Fresh, highest milk, Grocery, detergents_papers

From the above 2 analysis we can see that 3 clusters prove to be the base optimal number for quickly understanding the customer segmentation.

Possible extensions to the project is to use Principal Component Analysis for dimensionality reduction.
