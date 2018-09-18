# pylint: disable=E0602

import numpy as np

from kulgap.errors import UnsupportedMetricError
from kulgap.config import logger

from scipy.integrate import quad
from scipy.stats import norm

from GPy import plotting, models, kern


class Metrics:

    # Tuple of supported types
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

    # a mapping of Metric types to the function needed to calculate them
    # TYPE_TO_FUNCTION = {

    # }

    def __init__(self, collection):
        """
        Initializes class for calculating the metrics of a Collection

        A collection has:
        - observation times
        - sequences of observations
        - (optional) normalized sequences of observations
        """
        self._collection = collection

        self.types = {}
        for k in Metrics.TYPES:
            self.types[k] = None

        # GP settings
        self.GP_INPUT_DIM = 1.
        self.GP_VARIANCE = 1.
        self.GP_LENGTH_SCALE = 10.
        self.NUM_OPTIMIZE_RESTARTS = 7

        # save GP
        self.fit_gp = None
        self.gp_kernel = None

    @property
    def collection(self):
        return self._collection

    @classmethod
    def valid_types(cls, types: list):
        unsupported_metrics = set(map(str.lower, types)) - set(map(str.lower, cls.TYPES))

        if unsupported_metrics:
            raise UnsupportedMetricError(
                "The following metrics are not supported: " + str(unsupported_metrics)
                )

        return True

    def calculate(self, types: list):
        """
        Calculates metrics from the list of metrics available to calculate
        """

        if Metrics.valid_types(types):
            for metric_type in types:
                TYPE_TO_FUNCTION[metric_type]()

    def fit_gaussian_process(self):
        """
        Fits a single Gaussian process on the collection
        """

        num_replicates = self.collection.obs_seqs.shape[0]
        time_length = self.collection.obs_seqs.shape[1]

        x = np.tile(self.collection.obs_times, (1, num_replicates)).T
        y = np.resize(self.collection.obs_seqs_norm, (num_replicates * time_length, 1))

        # RBF = radial basis function / squared exponential by default
        self.gp_kernel = kern.RBF(
            input_dim=self.GP_INPUT_DIM,
            variance=self.GP_VARIANCE,
            lengthscale=self.GP_LENGTH_SCALE
        )

        self.fit_gp = models.GPRegression(x, y, self.gp_kernel)
        self.fit_gp.optimize_restarts(
            num_restarts=self.NUM_OPTIMIZE_RESTARTS, messages=False
            )

    def kl_divergence(self, other_metrics) -> float:
        """
        Compare the current GP with other_gp

        :param other_gp: Another GP to compare with
        :return: kl_divergence
        """

        if not self.fit_gp:
            logger.info("Currently no fit GP on %s, fitting a GP" % self.collection.name)
            self.fit_gaussian_process()
        
        if not other_metrics.fit_gp:
            logger.info("Currently no fit GP on %s, fitting a GP" % other_metrics.collection.name)
            other_metrics.fit_gaussian_process()

        logger.info("Calculating the KL Divergence between %s and %s" % \
            (self.collection.name, other_metrics.collection.name))

        def kl_integrand(t):
            control_mean, control_var = other_metrics.fit_gp.predict(np.asarray([[t]]))
            case_mean, case_var = self.fit_gp.predict(np.asarray([[t]]))

            return np.log10(case_var / control_var) + \
                ((control_var + (control_mean - case_mean) ** 2) / (2 * case_var))

        kl_divergence = abs(quad(kl_integrand, 0, max(self.collection.obs_times)) \
            - max(self.collection.obs_times) / 2)[0]
        
        logger.info("Calculated KL divergence is: %f" % kl_divergence)

        return kl_divergence

    def kl_p_value(self):
        if self.fit_gp:
            pass
        else:
            self.fit_gaussian_process()

    def response_angle(self):
        raise NotImplementedError

    def auc(self):
        raise NotImplementedError

    def gp_auc(self):
        if self.fit_gp:
            pass
        else:
            self.fit_gaussian_process()

    def auc_norm(self):
        raise NotImplementedError

    def mrecist(self):
        raise NotImplementedError

    def credible_intervals(self):
        if self.fit_gp:
            pass
        else:
            self.fit_gaussian_process()

    def percent_credible_intervals(self):
        if self.fit_gp:
            pass
        else:
            self.fit_gaussian_process()

    def write_metrics(self, out_path):
        raise NotImplementedError