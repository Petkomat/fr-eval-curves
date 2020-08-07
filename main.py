STEP_TYPE_LINEAR = "linear"
STEP_TYPE_QUADRATIC = "quadratic"
STEP_TYPE_EXPONENTIAL = "exponential"

ALLOWED_STEPS = [STEP_TYPE_LINEAR, STEP_TYPE_QUADRATIC, STEP_TYPE_EXPONENTIAL]


def compute_feature_set_sizes(n_features, basic_step, step_type):
    """
    Computes the sizes of feature subsets.

    :param n_features: number of features in data
    :param basic_step: int, Its meaning depends on the value of the step_type argument.
    :param step_type: str, An element of the list ALLOWED_STEPS.

            If linear, the sizes are 1 + k * basic_step, for 1 <= 1 + k * basic_step <= n_features
            If quadratic, the sizes are 1 + basic_step * k * (k + 1) / 2, in the same range
            If exponential, the sizes are basic_step ** k, in the same range.

    :return: The list of the computed feature sizes. If the last size in the sequence is smaller than n_features,
             the value n_features is appended to it.
    """
    sizes = []
    s0 = 1
    iterations = 0
    while s0 <= n_features:
        iterations += 1
        sizes.append(s0)
        if step_type == STEP_TYPE_LINEAR:
            s0 += basic_step
        elif step_type == STEP_TYPE_QUADRATIC:
            s0 += basic_step * iterations
        elif step_type == STEP_TYPE_EXPONENTIAL:
            s0 *= basic_step
        else:
            raise ValueError(
                f"Wrong step type: Expected an element of {ALLOWED_STEPS}, got {step_type}"
            )
        if len(sizes) >= 2 and sizes[-1] == sizes[-2]:
            raise ValueError(
                f"Improper argument combination: basic step and step type would result in an infinite loop."
            )
    if sizes[-1] < n_features:
        sizes.append(n_features)
    return sizes


def compute_curve(xs, y, model, model_quality_measure,
                  scores=None, feature_order=None, is_forward=True, basic_step=1, step_type=STEP_TYPE_LINEAR):
    """
    Computes a forward or backward feature addition curve.

    :param xs: a pair of 2D-arrays of features values (xs_train, xs_test), where, e.g.,  xs_train[j, i] = the value
           of the i-th feature for j-th training example; passed to the model
    :param y: a pair of a 1D-arrays (y_train, y_test) of target values; passed to the model
    :param model: presumably a scikit learn model
           (or something with .fit(xs_train, y_train) and .predict(xs_train) methods)
    :param model_quality_measure: one of the scikit quality measures, e.g., sklearn.metrics.mean_squared_error or
           sklearn.metrics.accuracy_score (or any other function(y_true, y_pred) -> quality).
    :param feature_order: None or a list-like of 0-based feature indices that corresponds to the feature ranking,
           e.g., in the ranking [3, 1, 4, 0, 2], feature 3 is top-ranked, feature 1 has rank 2, ..., and feature
           2 has rank 5. If None, scores must not be None.
    :param scores: None or a list-like of numeric values, scores[i] = importance of the i-th feature where more is
           better is assumed. Ignored if feature_order is not None. Otherwise, it must not be None.
    :param is_forward: bool, determines which type of the curve is built.
    :param basic_step: int, passed to the compute_feature_set_sizes basic step
           when computing the increasing sizes of features.
    :param step_type: str, An element of the list ALLOWED_STEPS. Passed to the compute_feature_set_sizes
    :return: A list of curve points (feature set size, quality of the corresponding model)
    """
    if feature_order is None:
        if scores is None:
            raise ValueError("Cannot determined the order of the features.")
        else:
            pairs = sorted(enumerate(scores), key=lambda pair: pair[1])
            feature_order = [i for i, _ in pairs]
    n_features = len(feature_order)
    feature_set_sizes = compute_feature_set_sizes(n_features, basic_step, step_type)
    xs_train, xs_test = xs
    y_train, y_test = y
    curve = []
    for size in feature_set_sizes:
        features = feature_order[:size] if is_forward else feature_order[-size:]
        model.fit(xs_train[:, features], y_train)
        y_predicted = model.predict(xs_test[:, features])
        quality = model_quality_measure(y_test, y_predicted)
        curve.append((len(features), quality))
    return curve


def plot_curve(curve):
    """
    Plots the curve
    :param curve: [(number of features1, quality1), (number of features2, quality2), ...]
    :return:
    """
    import matplotlib.pyplot as plt
    x = list(range(len(curve)))
    x_labels = [str(n) for n, _ in curve]
    y = [q for _, q in curve]
    plt.plot(x, y)
    plt.xticks(x, x_labels)
    plt.xlabel("feature subset size")
    plt.ylabel("quality")
    plt.show()


def example():
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    import numpy as np

    xs_all = np.random.rand(1000, 5)
    order = [3, 1, 4, 0, 2]
    y_regression = np.dot(xs_all, np.array([0.4, 0.8, 0.2, 1.0, 0.6]))  # importance ~ coefficient size
    y_bar = np.mean(y_regression)
    y_classification = [int(y > y_bar) for y in y_regression]
    model = KNeighborsClassifier()

    train_i = list(range(500))
    test_i = list(range(500, 1000))
    xs_train = xs_all[train_i, :]
    xs_test = xs_all[test_i, :]
    y_train = y_classification[:500]
    y_test = y_classification[500:]

    curve = compute_curve((xs_train, xs_test), (y_train, y_test), model, accuracy_score, feature_order=order)
    plot_curve(curve)


if __name__ == "__main__":
    example()
