# Feature Ranking evaluation curves

The code for constructing feature ranking evaluation curves.



## Forward feature addition curves

Once a feature ranking is obtained (represented as a list of indices, e.g., `[3, 1, 4, 0, 2]`), we choose a predictive model and compute the following qualities:

- quality `q1` of the model that uses feature `x3`,
- quality `q2` of the model that uses features `x3` and `x1`,
- ...
- quality `q5` of the model that uses feature `x3`, `x1`, ..., and `x2`

The qualities can be then plotted as a curve that consists of points `(<i>, q<i>)`, where `i` denotes the number of used features. This type of curve measures how close to the **top of the ranking** are relevant features.


## Backward feature addition curves

Once a feature ranking is obtained (represented as a list of indices, e.g., `[3, 1, 4, 0, 2]`), we choose a predictive model and compute the following qualities:

- quality `q1` of the model that uses feature `x2`,
- quality `q2` of the model that uses features `x2` and `x0`,
- ...
- quality `q5` of the model that uses feature `x2`, `x0`, ..., and `x3`

The qualities can be then plotted as a curve that consists of points `(<i>, q<i>)`, where `i` denotes the number of used features. This type of curve measures how close to the **bottom of the ranking** are relevant features.

## How to use the code?

The function `example` in `main.py` gives an example of how to 

- construct a curve
- plot the curve

The code supports regression, classification etc. problems. Instead of adding features one by one to the set of used features, we can follow a more general linear (adding 10 features per step), quadratic or exponential progression.
