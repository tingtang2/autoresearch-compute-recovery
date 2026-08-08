# Predicting Optimal Fertilizers

## Overview

Welcome to the 2025 Kaggle Playground Series! We plan to continue in the spirit of previous playgrounds, providing interesting and approachable datasets for our community to practice their machine learning skills, and anticipate a competition each month.

**Your Goal:** Your objective is to select the best fertilizer for different weather, soil conditions and crops.

## Evaluation

Submissions are evaluated according to the **Mean Average Precision @ 3 (MAP@3)**:

$$MAP@3 = \frac{1}{U} \sum_{u=1}^{U} \sum_{k=1}^{min(n,3)} P(k) \times rel(k)$$

where:
- U is the number of observations
- P(k) is the precision at cutoff k
- n is the number of predictions per observation
- rel(k) is an indicator function equaling 1 if the item at rank k is a relevant (correct) label, zero otherwise

Once a correct label has been scored for an observation, that label is no longer considered relevant for that observation, and additional predictions of that label are skipped in the calculation. For example, if the correct label is A for an observation, the following predictions all score an average precision of 1.0:

- `[A, B, C, D, E]`
- `[A, A, A, A, A]`
- `[A, B, A, C, A]`

## Submission File

For each `id` in the test set, you may predict up to 3 `Fertilizer Name` values, with the predictions space delimited. The file should contain a header and have the following format:

```
id,Fertilizer Name 
750000,14-35-14 10-26-26 Urea
750001,14-35-14 10-26-26 Urea 
...
```

## Timeline

- **Start Date:** June 1, 2025
- **Entry Deadline:** Same as the Final Submission Deadline
- **Team Merger Deadline:** Same as the Final Submission Deadline
- **Final Submission Deadline:** June 30, 2025

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted.

## About the Tabular Playground Series

The goal of the Tabular Playground Series is to provide the Kaggle community with a variety of fairly light-weight challenges that can be used to learn and sharpen skills in different aspects of machine learning and data science.

### Synthetically-Generated Datasets

Using synthetic data for Playground competitions allows us to strike a balance between having real-world data (with named features) and ensuring test labels are not publicly available.

## Prizes

- 1st Place - Choice of Kaggle merchandise
- 2nd Place - Choice of Kaggle merchandise
- 3rd Place - Choice of Kaggle merchandise

## Citation

Walter Reade and Elizabeth Park. Predicting Optimal Fertilizers. https://kaggle.com/competitions/playground-series-s5e6, 2025. Kaggle.
