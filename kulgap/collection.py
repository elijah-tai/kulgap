import numpy as np

from kulgap.config import logger
from kulgap.errors import TimeValidationError
from kulgap.metadata import Metadata
import kulgap.utils as utils

class Collection:
    """
    Holds multiple sequences of observed data that was observed
    at specific times.

    The collection is what the GP is fit on.
    """

    def __init__(self, name: str, obs_times=None, obs_seqs=np.array([])) -> None:
        """
        :param name:        name of the collection
        :param metadata:    metadata information about each collection
        :param obs_times:   array of observation times
        :param obs_seqs:    array of array of observations
        """

        self.name = name
        
        if not isinstance(obs_times, list):
            raise TimeValidationError(str(obs_times) + " is not a list.")

        self._obs_times = obs_times
        self._obs_seqs = obs_seqs
        self._obs_seqs_norm = None

        # create metadata
        self.metadata = Metadata()


    @property
    def obs_times(self):
        return self._obs_times

    @property
    def obs_seqs(self):
        return self._obs_seqs

    @obs_seqs.setter
    def add_obs_seqs(self, sequence):
        """
        The new sequences that are added should have
        the same length as previous observation sequences, 
        or else this append will not work.
        """
        np.append(self._obs_seqs, sequence, axis=0)

    def normalize_obs_seqs(self):
        """
        Normalizes observations.
        """
        self._obs_seqs_norm = utils.normalize_data(self.obs_times, self.obs_seqs, self.metadata.gp_start)

    def create_full_data(self):
        pass