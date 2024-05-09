from . import *

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import _check_sample_weight, check_is_fitted
from sklearn.utils import check_random_state
from sklearn.neighbors._ball_tree import BallTree, DTYPE
from sklearn.neighbors._kd_tree import KDTree
from sklearn.utils.extmath import row_norms


TREE_DICT = {"ball_tree": BallTree, "kd_tree": KDTree}

def tim_fit(self, X, y=None, sample_weight=None, percent_ignored=1, n_init=25, approach = "full-calculation"):
    """Fit the Kernel Density model on the data.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        List of n_features-dimensional data points.  Each row
        corresponds to a single data point.

    y : None
        Ignored. This parameter exists only for compatibility with
        :class:`~sklearn.pipeline.Pipeline`.

    sample_weight : array-like of shape (n_samples,), default=None
        List of sample weights attached to the data X.

        .. versionadded:: 0.20

    Returns
    -------
    self : object
        Returns the instance itself.
    """
    self._validate_params()

    algorithm = self._choose_algorithm(self.algorithm, self.metric)

    if not isinstance(y, str):
        if isinstance(self.bandwidth, str):
            if self.bandwidth == "scott":
                self.bandwidth_ = X.shape[0] ** (-1 / (X.shape[1] + 4))
            elif self.bandwidth == "silverman":
                self.bandwidth_ = (X.shape[0] * (X.shape[1] + 2) / 4) ** (
                    -1 / (X.shape[1] + 4))
            elif self.bandwidth == "nigh":
                if isinstance(y, pd.DataFrame) or isinstance(y, np.ndarray):
                    if isinstance(X, pd.DataFrame):
                        X = X.values
                    if isinstance(y, pd.DataFrame):
                        y = y.values
                    dist = np.linalg.norm(X - y[:, None], axis=-1)
                    self.bandwidth_ = np.percentile(dist.min(0), 100 - percent_ignored)
                elif y == None:
                    if isinstance(X, pd.DataFrame):
                        X_ = X.values
                    else:
                        X_ = X
                    if approach == "cross-validation":
                        bands = np.empty(shape=[1, 0])
                        for i in range(n_init):
                            xtr, xte = train_test_split(X_, train_size=.8, random_state=i)
                            small_chunksize, large_chunksize = 10, 100
                            tr_points = xtr.shape[0]
                            te_points = xte.shape[0]
                            min_dists = np.zeros(te_points)
                            for i in range(0, te_points, large_chunksize):
                                large_chunk = xte[i:i + large_chunksize]
                                large_dist = np.zeros(
                                    shape=[large_chunk.shape[0], int(np.ceil(tr_points / small_chunksize))])
                                for j in range(0, tr_points, small_chunksize):
                                    small_chunk = xtr[j:j + small_chunksize]
                                    small_dist = np.linalg.norm(large_chunk - small_chunk[:, None], axis=-1)
                                    small_dist[small_dist == 0] = np.inf
                                    large_dist[:, j // small_chunksize] = small_dist.min(0)
                                min_dists[i:i + large_chunk.shape[0]] = large_dist.min(1)
                            bands=np.append(bands,np.percentile(min_dists, 100 - percent_ignored))
                        #print(f"median : {np.median(bands)} mean:{np.mean(bands)}")
                        self.bandwidth_ = np.mean(bands)
                    elif approach == "full-calculation":
                        small_chunksize, large_chunksize = 10, 100
                        num_points = X_.shape[0]
                        min_dists = np.zeros(num_points)
                        for i in range(0, num_points, large_chunksize):
                            large_chunk = X_[i:i + large_chunksize]
                            large_dist = np.zeros(
                                shape=[large_chunk.shape[0], int(np.ceil(num_points / small_chunksize))])
                            for j in range(0, num_points, small_chunksize):
                                small_chunk = X_[j:j + small_chunksize]
                                small_dist = np.linalg.norm(large_chunk - small_chunk[:, None], axis=-1)
                                small_dist[small_dist == 0] = np.inf
                                large_dist[:, j // small_chunksize] = small_dist.min(0)
                            min_dists[i:i + large_chunk.shape[0]] = large_dist.min(1)
                        self.bandwidth_ = np.percentile(min_dists, 100-percent_ignored)




        else:
            self.bandwidth_ = self.bandwidth

    X = self._validate_data(X, order="C", dtype=DTYPE)
    if not isinstance(y, str):
        self.fitted_data = X

    if sample_weight is not None:
        sample_weight = _check_sample_weight(
            sample_weight, X, DTYPE, only_non_negative=True
        )

    kwargs = self.metric_params
    if kwargs is None:
        kwargs = {}
    self.tree_ = TREE_DICT[algorithm](
        X,
        metric=self.metric,
        leaf_size=self.leaf_size,
        sample_weight=sample_weight,
        **kwargs,
    )
    return self

def nigh_bandwidth(self, start_bandwidth=None, bandwidth_step=1, hold_out_threshold=0.01):
    check_is_fitted(self)
    if start_bandwidth == None:
        start_bandwidth = self.bandwidth_
    if self.kernel not in {"tophat", "linear", "epanechnikov"}: #"gaussian", "exponential",
        raise NotImplementedError()
    tune, end_next = True, False
    ho_inf = [np.inf]
    ho_thres = hold_out_threshold

    train, ho = train_test_split(self.fitted_data, train_size=.8, random_state=0)
    while tune:
        self.fit(train, y="omit_refit")
        ho_scores = self.score_samples(ho)
        ho_inf.append(np.mean(ho_scores == -np.inf))
        if end_next:
            tune = False
        elif (ho_inf[-2] <= ho_thres):
            if (ho_inf[-1] <= ho_thres):
                bandwidth_step *= .5
            elif (ho_inf[-3] <= ho_thres):
                bandwidth_step *= -1
                end_next = True
            else:
                bandwidth_step *= -.5
        elif (ho_inf[-1] <= ho_thres):
            bandwidth_step *= -.5
        self.bandwidth_ += bandwidth_step
        #print(self.bandwidth_, bandwidth_step)
    self.fit(self.fitted_data, y="omit_refit")

def sample(self, n_samples=1, random_state=None):
    """Generate random samples from the model.

    Currently, this is implemented only for gaussian and tophat kernels.

    Parameters
    ----------
    n_samples : int, default=1
        Number of samples to generate.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation used to generate
        random samples. Pass an int for reproducible results
        across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    X : array-like of shape (n_samples, n_features)
        List of samples.
    """
    check_is_fitted(self)
    # TODO: implement sampling for other valid kernel shapes
    if self.kernel not in {"gaussian", "tophat", "linear", "exponential", "epanechnikov"}:
        raise NotImplementedError()

    data = np.asarray(self.tree_.data)

    rng = check_random_state(random_state)
    u = rng.uniform(0, 1, size=n_samples)
    if self.tree_.sample_weight is None:
        i = (u * data.shape[0]).astype(np.int64)
    else:
        cumsum_weight = np.cumsum(np.asarray(self.tree_.sample_weight))
        sum_weight = cumsum_weight[-1]
        i = np.searchsorted(cumsum_weight, u * sum_weight)
    if self.kernel == "gaussian":
        return np.atleast_2d(rng.normal(data[i], self.bandwidth_))

    dim = data.shape[1]
    X = rng.normal(size=(n_samples, dim))
    lengths = row_norms(X, squared=False)
    if self.kernel == "tophat":
        rads = rng.uniform(size=(n_samples)) ** (1 / dim)
    elif self.kernel == "linear":
        rads = rng.beta(a=dim, b=2, size=(n_samples))
    elif self.kernel == "epanechnikov":
        rads = np.sqrt(rng.beta(a=dim / 2, b=2, size=(n_samples)))
    elif self.kernel == "exponential":
        rads = rng.gamma(dim, size=(n_samples))
    X = X * (self.bandwidth_ * rads / lengths).reshape(-1, 1)
    return np.atleast_2d(data[i] + X)

KernelDensity.fit = tim_fit
KernelDensity.sample = sample
KernelDensity.nigh_bandwidth = nigh_bandwidth