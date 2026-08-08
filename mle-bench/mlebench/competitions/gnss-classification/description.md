# GNSS LOS/NLOS Classification

## Description

### Motivation

Global navigation satellite system (GNSS) positioning accuracy is degraded in urban canyons due to signal blockages and reflections. This competition aims to foster the machine learning algorithms in GNSS multipath/NLOS problem. There are three tasks for this problem: LOS/NLOS classification, Pseudorange Error Prediction, and Positioning Prediction.

One of the important characteristics of the satellite is their visibility: whether they are line-of-sight (LOS) or non-line-of-sight (NLOS). This is the page for GNSS LOS/NLOS classifier performance evaluation platform. You can upload your prediction result to evaluate the performance of your model in different scenarios.

### Acknowledgements

We thank the fellows who contributed to the project:
Prof. Hsu, Mr. Huang, Mr. Zhang, Mr. Ivan, Mr. Zhong, Mr. Max, Mr. Xu

## Evaluation

### Evaluation Metric

The evaluation metric for this competition is **Accuracy**.

Accuracy is calculated as:

$$\text{Accuracy} = \frac{TP + TN}{P + N}$$

where:
- TP is true positive samples
- TN is true negative samples
- P and N represent positive samples and negative samples, respectively

### Benchmark

The benchmark result is from a simple MLP model. The input of the network is the pseudorange residual, CN0, and elevation angle; output is the probability of the LOS and NLOS.

## Submission Format

For every GPS Time and Every Satellite in the dataset, submission files should contain two columns: `ID` and `Predict`.

- `ID` follows the same sequence with the testing dataset
- Within each GPS Time, the satellite sequence should also be the same as that in the testing set
- Every `Prediction` is an int value, either 1 or 0, where 1 means LOS, 0 means NLOS

The file should contain a header and have the following format:

```
ID,Predict
0,1
1,0
2,1
3,0
4,0
5,1
6,0
7,1
8,0
9,0
10,1
11,0
12,1
13,0
```

## Citation

POLYU IPNL. GNSS LOS/NLOS CLASSIFICATION. https://kaggle.com/competitions/gnss-classification, 2022. Kaggle.
