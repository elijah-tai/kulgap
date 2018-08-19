# pylint: disable=E0602

import numpy as np

from kulgap.errors import UnsupportedMetricError

from scipy.integrate import quad
from scipy.stats import norm

from GPy import plotting, models, kern

class Metrics:

    TYPES = (
        'KL_P_CVSC', 'KL_DIVERGENCE', 'KL_P_VALUE', 'EMPIRICAL_KL',

        'MRECIST', 'MRECIST_COUNTS',

        'RESPONSE_ANGLE', 'RESPONSE_ANGLE_RELATIVE',
        'RESPONSE_ANGLE_AVERAGE', 'RESPONSE_ANGLE_AVERAGE_RELATIVE',

        'AUC', 'AUC_NORM', 'AUC_GP', 
        
        'DIRECTION',

        'CREDIBLE_INTERVALS', 'PERCENT_CREDIBLE_INTERVALS',
        
        'RATES_LIST',
        
        'DELTA_LOG_LIKELIHOOD_H0_H1'
    )

    def __init__(self, collection):
        self._collection = collection

        self.types = {}
        for k in Metrics.TYPES: 
            self.types[k] = None


    @property
    def collection(self):
        return self._collection

    @classmethod
    def valid_types(cls, types: list):
        unsupported_metrics = set(map(str.lower, types)) - set(map(str.lower, cls.TYPES))
        
        if len(unsupported_metrics) > 0:
            raise UnsupportedMetricError("The following metrics are not supported: " + str(unsupported_metrics))
        
        return True

    def calculate(self, types: list):
        """
        Calculates metrics from the list of metrics available to calculate
        """

        if valid_types(types):
            for metric_type in types:
                pass

    def fit_gaussian_process(self):
        raise NotImplementedError

    def kl_divergence(self):
        raise NotImplementedError
    
    def kl_p_value(self):
        raise NotImplementedError

    def response_angle(self):
        raise NotImplementedError

    def auc(self):
        raise NotImplementedError

    def gp_auc(self):
        raise NotImplementedError

    def auc_norm(self):
        raise NotImplementedError

    def mrecist(self):
        raise NotImplementedError

    def credible_intervals(self):
        raise NotImplementedError

    def percent_credible_intervals(self):
        raise NotImplementedError

    def write_metrics(self, out_path):
        raise NotImplementedError
