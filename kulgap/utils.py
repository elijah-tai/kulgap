# pylint: disable=E1101

import numpy as np

from kulgap.config import logger

def find_start_date_index(obs_times: np.array, drug_start_day: int) -> int:
    """
    Returns the index in the array of the location of the drug's
    start day, + or - 1.

    :return: int
    """
    start = None
    start_found = False

    for i in range(len(obs_times.ravel())):
        if obs_times[i] - 1 <= drug_start_day <= obs_times[i] + 1 and start_found is False:
            start = i
            start_found = True
    return start

def normalize_data(obs_times, obs_seqs, drug_start_day) -> np.array:
    """
    Normalizes all observations.

    :return:
    """
    logger.log("Normalizing data...")

    def normalize_treatment_start_day_and_log_transform(y, treatment_start):
        """
        Normalize by dividing every y element-wise by the first day's median
        and then taking the log.

        :param y:
        :return:
        """
        return np.log(np.asarray((y.T + 0.01) / y.T[int(treatment_start)], dtype=float).T) + 1

    # TODO: Need to normalize on treatment start_day
    return normalize_treatment_start_day_and_log_transform(
        obs_seqs,
        find_start_date_index(obs_times, drug_start_day)
        )
