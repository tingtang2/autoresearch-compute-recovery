# Task

Predict a binary target for synthetic data.

## Description

# Metric

Area under the ROC curve between the predicted probability and the observed target.

# Submission Format

For each `id` in the test set, you must predict a probability for the `diagnosed_diabetes` variable. The file should contain a header and have the following format:

```
id,diagnosed_diabetes
700000,0.2
700001,0.4
700002,0.5
etc.
```

# Dataset 

- **train.csv** - the training dataset; `diagnosed_diabetes` is the binary target.
- **test.csv** - the test dataset; your objective is to predict the probability of `diagnosed_diabetes`
- **sample_submission.csv** - a sample submission file in the correct format
