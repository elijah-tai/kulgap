# pylint: disable=E1101

import numpy as np

from kulgap.config import logger


def normalize(y, start):  # relativize
    return y / y[start] - 1


def centre(y, start):
    return y - y[start]


def calculate_p_value(observed, empirical):
    """
    Calculates the p_value of a given observation in empirical tests.
    """

    return (len([x for x in empirical if x >= observed]) + 1) / \
        (len(empirical) + 1)


def find_start_date_index(obs_times: np.ndarray, drug_start_day: int) -> int:
    """
    Returns the index in the array of the location of the drug's
    start day, + or - 1.

    :return: int
    """
    start = None
    start_found = False

    for i in range(len(obs_times.ravel())):
        if obs_times[i] - 1 <= drug_start_day <= obs_times[i] + \
                1 and start_found is False:
            start = i
            start_found = True
    return start


def normalize_data(
        obs_times: np.ndarray,
        obs_seqs: np.ndarray,
        drug_start_day: int = 0) -> np.array:
    """
    Normalizes all observations.

    :return:
    """
    logger.info("Normalizing data...")

    def normalize_treatment_start_day_and_log_transform(y, drug_start_day):
        """
        Normalize by dividing every y element-wise by the first day's median
        and then taking the log.

        :param y:
        :return:
        """
        return np.log(
            np.asarray(
                (y.T + 0.01) / y.T[drug_start_day],
                dtype=float).T) + 1

    # TODO: Need to normalize on treatment start_day
    return normalize_treatment_start_day_and_log_transform(
        obs_seqs,
        find_start_date_index(obs_times, drug_start_day)
    )
