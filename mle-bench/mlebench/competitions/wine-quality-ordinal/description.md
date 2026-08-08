# Predicting Wine Quality: An Ordinal Regression Challenge

## Overview

Welcome to the 2023 Kaggle Playground Series! Thank you to everyone who participated in and contributed to Season 3 Playground Series so far!

**Your Goal:** For this Episode of the Series, your task is to use regression to predict the quality of wine based on various properties. Good luck!

## Evaluation

Submissions are scored based on the **quadratic weighted kappa**, which measures the agreement between two outcomes. This metric typically varies from 0 (random agreement) to 1 (complete agreement). In the event that there is less agreement than expected by chance, the metric may go below 0.

The quadratic weighted kappa is calculated as follows:

1. First, an N x N histogram matrix O is constructed, such that O_{i,j} corresponds to the number of Ids i (actual) that received a predicted value j.

2. An N-by-N matrix of weights, w, is calculated based on the difference between actual and predicted values:

$$w_{i,j} = \frac{(i-j)^2}{(N-1)^2}$$

3. An N-by-N histogram matrix of expected outcomes, E, is calculated assuming that there is no correlation between values. This is calculated as the outer product between the actual histogram vector of outcomes and the predicted histogram vector, normalized such that E and O have the same sum.

4. From these three matrices, the quadratic weighted kappa is calculated as:

$$\kappa = 1 - \frac{\sum_{i,j} w_{i,j} O_{i,j}}{\sum_{i,j} w_{i,j} E_{i,j}}$$

## Submission File

For each `Id` in the test set, you must predict the value for the target `quality`. The file should contain a header and have the following format:

```
Id,quality
2056,5
2057,7
2058,3
etc.
```

## Citation

Sujal Raj. Predicting Wine Quality: An Ordinal Regression Challenge. https://kaggle.com/competitions/wine-quality-ordinal, 2025. Kaggle.
