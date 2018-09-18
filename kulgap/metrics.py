# pylint: disable=E0602

import itertools

import numpy as np

from kulgap.errors import UnsupportedMetricError
from kulgap.config import logger
from kulgap import utils

from scipy.integrate import quad
from scipy.stats import norm

from GPy import plotting, models, kern


class Metrics:

    # Tuple of supported types
    TYPES = (
        'KL_DIVERGENCE', 'KL_P_VALUE',

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

    @staticmethod
    def _prepare_inputs_for_fitting_gp(x, y):
        """
        For x - repeats each element in order by the number of observations per sequence in y.
        For y - resizes it to be a single column of elements in y
        """

        num_observations = x.shape[0]
        num_replicates = y.shape[0]

        new_x = np.tile(x, (1, num_replicates)).T
        new_y = np.resize(y, (num_replicates * num_observations, 1))

        return new_x, new_y

    @staticmethod
    def _fit_gaussian_process(x, y, kernel = kern.RBF(1, 1., 10.), num_restarts: int = 7):
        """
        Fits a Gaussian process
        """

        x, y = Metrics._prepare_inputs_for_fitting_gp(x, y)

        gp = models.GPRegression(x, y, kernel)
        gp.optimize_restarts(num_restarts=num_restarts, messages=False)

        return gp, kernel

    def fit_gaussian_process(self):
        """
        Fits a Gaussian process on the Collection while saving the GP and the kernel
        """

        # RBF = radial basis function / squared exponential by default
        self.gp_kernel = kern.RBF(
            input_dim=self.GP_INPUT_DIM,
            variance=self.GP_VARIANCE,
            lengthscale=self.GP_LENGTH_SCALE
        )

        x, y = self.collection.obs_times, self.collection.obs_seqs_norm

        self.fit_gp, self.gp_kernel = Metrics._fit_gaussian_process(
            x, y, self.gp_kernel, self.NUM_OPTIMIZE_RESTARTS
            )

    @staticmethod
    def _kl_divergence(gp_control, gp_case, obs_times, case_start: int = 0):
        """
        Calculates the KL divergence
        """

        def kl_integrand(t):
            control_mean, control_var = gp_control.predict(np.asarray([[t]]))
            case_mean, case_var = gp_case.predict(np.asarray([[t]]))

            return np.log10(case_var / control_var) + \
                ((control_var + (control_mean - case_mean) ** 2) / (2 * case_var))

        kl_divergence = abs(quad(kl_integrand, case_start, max(obs_times)) \
            - max(obs_times) / 2)[0]
        
        return kl_divergence

    def kl_divergence(self, other_metrics) -> float:
        """
        Compare the current GP with other_metric's GP

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

        kl_divergence = self._kl_divergence(other_metrics.fit_gp, self.fit_gp, self.collection.obs_times)
        
        logger.info("Calculated KL divergence is: %f" % kl_divergence)

        return kl_divergence

    def kl_p_value(self, other_metrics) -> float:
        """
        Calculates the empirical KL divergences between two collections
        by fitting a GP on randomized control and cases.

        :param other_metrics: The Metrics instance
        :return: kl_p_value: The p value of the calculated KL divergence
                             for the fit GP
        """

        if not self.fit_gp:
            logger.info("Currently no fit GP on %s, fitting a GP" % self.collection.name)
            self.fit_gaussian_process()

        if not other_metrics.fit_gp:
            logger.info("Currently no fit GP on %s, fitting a GP" % other_metrics.collection.name)
            other_metrics.fit_gaussian_process()

        all_pseudo_controls, all_pseudo_cases = self._randomize_controls_cases_procedural(other_metrics.collection)

        empirical_kl = []
        counter = 0
        for pseudo_controls, pseudo_cases in zip(all_pseudo_controls, all_pseudo_cases):
            logger.info("Processed %i out of %i cases" % (counter, len(
                all_pseudo_controls + all_pseudo_cases
            )))

            i = np.stack(pseudo_controls)
            j = np.stack(pseudo_cases)

            # clean up zeros
            i[i == 0] = np.finfo(np.float32).tiny
            j[j == 0] = np.finfo(np.float32).tiny

            control_x = other_metrics.collection.obs_times
            case_x = self.collection.obs_times

            gp_control, _ = self._fit_gaussian_process(control_x, i)
            gp_case, _ = self._fit_gaussian_process(case_x, j)

            empirical_kl.append((Metrics._kl_divergence(gp_control, gp_case, case_x)))
            
            counter += 1

        p_value = utils.calculate_p_value(self.kl_divergence(other_metrics), empirical_kl)

        logger.info("The calculated p value for %s is: %f." % (self.collection.name, p_value))

        return p_value

    def _randomize_controls_cases_procedural(self, control_collection):
        """
        Creates all possible pseudo controls and pseudo cases, with a one-to-one relationship.

        :param patient
        :return all_pseudo_controls, all_pseudo_cases
        """

        all_pseudo_controls, all_pseudo_cases = [], []

        control_num_replicates = len(control_collection.obs_seqs_norm) 
        case_num_replicates = len(self.collection.obs_seqs_norm)

        all_y_norm = np.append(self.collection.obs_seqs_norm, control_collection.obs_seqs_norm, axis=0)

        for pattern in itertools.product([True, False], repeat=len(all_y_norm)):
            all_pseudo_controls.append([x[1] for x in zip(pattern, all_y_norm) if x[0]])
            all_pseudo_cases.append([x[1] for x in zip(pattern, all_y_norm) if not x[0]])

        all_pseudo_controls = [x for x in all_pseudo_controls if control_num_replicates == len(x)]
        all_pseudo_cases = [x for x in all_pseudo_cases if case_num_replicates == len(x)]

        return all_pseudo_controls, all_pseudo_cases

    def response_angle(self):
        raise NotImplementedError

    def auc(self):
        raise NotImplementedError

    def auc_norm(self):
        raise NotImplementedError

    def gp_auc(self):
        if self.fit_gp:
            pass
        else:
            self.fit_gaussian_process()

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
        # TODO: This is the interface that will be used for the web UI
        raise NotImplementedError
