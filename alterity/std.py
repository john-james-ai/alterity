#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Enter Project Name in Workspace Settings                                            #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.10                                                                             #
# Filename   : /alterity/std.py                                                                    #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : Enter URL in Workspace Settings                                                     #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday May 19th 2023 07:17:40 am                                                    #
# Modified   : Friday May 19th 2023 12:48:43 pm                                                    #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import numpy as np

from alterity.base import Alterity


# ------------------------------------------------------------------------------------------------ #
class STDOutlier(Alterity):
    r"""STDOutlier Outlier Detection Algorithm

    Standard outlier detection method based upon the sample mean and the sample standard deviation;
    whereas, an observation is considered an outlier if its value is outside of the
    interval formed by the sample mean +/- a threshold value * the sample standard deviation.

    According to Chebyshev inequality, if a random variable X with mean $\mu$ and variance
    $\sigma^$ exists, then for any $k>0$,

    $$
    P[|X-\mu|\ge k\sigma]\le\frac{1}{k^2}
    $$

    This inequality enables us to determine what proportion of our data will reside within $k$
    standard deviations of the sample mean.

    Args:
        threshold (int): The threshold above which an observation is considered an outlier. This
            value is expressed in terms of the number of standard deviations from the
            sample mean

    """

    def __init__(self, threshold: int = 3) -> None:
        super().__init__()
        self._threshold = threshold
        self._mean = None
        self._std = None

    @property
    def threshold(self) -> int:
        return self._threshold

    @property
    def mean(self) -> np.float32:
        return self._mean

    @property
    def std(self) -> np.float32:
        return self._std

    def fit(self, X: np.float32, y: np.float32 = None) -> Alterity:
        """Fits the outlier detector

        Args:
            X (array-like): Array of shape (n_samples, n_features). Input uses np.float32
                for maximum efficiency.

            y (None): Ignored. In place for scikit-learn API conformity.

        Returns: self (Alterity): The fitted estimator.

        """
        self._mean = np.mean(X, axis=0)
        self._std = np.std(X, axis=0)
        return self

    def predict(self, X: np.float32) -> np.float32:
        """Predict whether each sample is an outlier.

        Args:
            X (np.float32): Array of shape (n_samples, n_features). Input samples
                will be converted to np.float32 dtype.

        Returns (np.float32): Binary ndarray of shape (n_samples,) indicating
            whether each sample is an outlier (+1) or not (0).

        """
        above = X > (self._mean - (self._threshold * self._std))
        below = X < (self._mean + (self._threshold * self._std))
        return np.where(above & below, 0, 1)
