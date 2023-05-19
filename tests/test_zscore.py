#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Enter Project Name in Workspace Settings                                            #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.10                                                                             #
# Filename   : /tests/test_zoutlier.py                                                             #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : Enter URL in Workspace Settings                                                     #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday May 19th 2023 08:07:06 am                                                    #
# Modified   : Friday May 19th 2023 10:56:18 am                                                    #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import inspect
from datetime import datetime
import pytest
import logging

import numpy as np

from alterity.zscore import ZScoreOutlier


# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------------------------ #
double_line = f"\n{100 * '='}"
single_line = f"\n{100 * '-'}"


@pytest.mark.zscore
class TestZScoreOutlier:  # pragma: no cover
    # ============================================================================================ #
    def test_threshold(self, caplog):
        start = datetime.now()
        logger.info(
            "\n\nStarted {} {} at {} on {}".format(
                self.__class__.__name__,
                inspect.stack()[0][3],
                start.strftime("%I:%M:%S %p"),
                start.strftime("%m/%d/%Y"),
            )
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        X = np.array(
            [[2, 1, 0, 2, 2, 1, 3, 5, 2, -1, -2, -99], [2, 1, 0, 2, 2, 1, 3, 5, 2, -1, -2, 99]]
        ).transpose()
        logger.debug(X)
        outlier = ZScoreOutlier(threshold=3)
        outlier.fit(X=X)
        assert outlier.threshold == 3
        assert outlier.mean.all() == np.mean(X, axis=0).all()
        assert outlier.std.all() == np.std(X, axis=0).all()

        logger.debug(outlier.mean)
        logger.debug(outlier.std)

        y = outlier.predict(X)
        logger.debug(y)
        assert y.shape == X.shape
        assert np.sum(y) == 4

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            "\nCompleted {} {} in {} seconds at {} on {}".format(
                self.__class__.__name__,
                inspect.stack()[0][3],
                duration,
                end.strftime("%I:%M:%S %p"),
                end.strftime("%m/%d/%Y"),
            )
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_default_threshold(self, caplog):
        start = datetime.now()
        logger.info(
            "\n\nStarted {} {} at {} on {}".format(
                self.__class__.__name__,
                inspect.stack()[0][3],
                start.strftime("%I:%M:%S %p"),
                start.strftime("%m/%d/%Y"),
            )
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        X = np.array(
            [[2, 1, 0, 2, 2, 1, 3, 5, 2, -1, -2, -99], [2, 1, 0, 2, 2, 1, 3, 5, 2, -1, -2, 99]]
        ).transpose()
        logger.debug(X)
        outlier = ZScoreOutlier()
        outlier.fit(X=X)
        assert outlier.threshold == 3
        assert outlier.mean.all() == np.mean(X, axis=0).all()
        assert outlier.std.all() == np.std(X, axis=0).all()

        logger.debug(outlier.mean)
        logger.debug(outlier.std)

        y = outlier.predict(X)
        logger.debug(y)
        assert y.shape == X.shape
        assert np.sum(y) == 4

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            "\nCompleted {} {} in {} seconds at {} on {}".format(
                self.__class__.__name__,
                inspect.stack()[0][3],
                duration,
                end.strftime("%I:%M:%S %p"),
                end.strftime("%m/%d/%Y"),
            )
        )
        logger.info(single_line)
