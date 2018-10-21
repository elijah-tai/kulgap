# pylint: disable=E0602

import itertools

import numpy as np

from kulgap.errors import UnsupportedMetricError, UnsupportedScalingError
from kulgap.config import logger
from kulgap import utils

from scipy.integrate import quad
from scipy.stats import norm

from GPy import plotting, models, kern

import statsmodels.api as sm


class Metrics:

    # Tuple of supported types
    TYPES = (
        'KL_DIVERGENCE', 'KL_P_VALUE',

        'MRECIST', 
        # 'MRECIST_COUNTS',

        'ANGLE', 
        
        # 'ANGLE_RELATIVE',
        # 'ANGLE_AVERAGE', 'ANGLE_AVERAGE_RELATIVE',

        # 'AUC', 'AUC_NORM', 
        'AUC_GP',

        # 'DIRECTION',

        # 'CREDIBLE_INTERVALS', 'PERCENT_CREDIBLE_INTERVALS',

        # 'RATES_LIST',

        # 'DELTA_LOG_LIKELIHOOD_H0_H1'
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
        unsupported_metrics = set(
            map(str.lower, types)) - set(map(str.lower, cls.TYPES))

        if unsupported_metrics:
            raise UnsupportedMetricError(
                "The following metrics are not supported: " +
                str(unsupported_metrics))

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
    def _fit_gaussian_process(
            x,
            y,
            kernel=kern.RBF(
                1,
                1.,
                10.),
            num_restarts: int = 7):
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

            return np.log10(case_var / control_var) + ((control_var + \
                            (control_mean - case_mean) ** 2) / (2 * case_var))

        kl_divergence = abs(quad(kl_integrand, case_start, max(obs_times))
                            - max(obs_times) / 2)[0]

        return kl_divergence

    def kl_divergence(self, other_metrics) -> float:
        """
        Compare the current GP with other_metric's GP

        :param other_gp: Another GP to compare with
        :return: kl_divergence
        """

        if not self.fit_gp:
            logger.info(
                "Currently no fit GP on %s, fitting a GP" %
                self.collection.name)
            self.fit_gaussian_process()

        if not other_metrics.fit_gp:
            logger.info(
                "Currently no fit GP on %s, fitting a GP" %
                other_metrics.collection.name)
            other_metrics.fit_gaussian_process()

        logger.info("Calculating the KL Divergence between %s and %s" %
                    (self.collection.name, other_metrics.collection.name))

        kl_divergence = self._kl_divergence(
            other_metrics.fit_gp, self.fit_gp, self.collection.obs_times)

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
            logger.info(
                "Currently no fit GP on %s, fitting a GP" %
                self.collection.name)
            self.fit_gaussian_process()

        if not other_metrics.fit_gp:
            logger.info(
                "Currently no fit GP on %s, fitting a GP" %
                other_metrics.collection.name)
            other_metrics.fit_gaussian_process()

        all_pseudo_controls, all_pseudo_cases = self._randomize_controls_cases_procedural(
            other_metrics.collection)

        empirical_kl = []
        counter = 0
        for pseudo_controls, pseudo_cases in zip(
                all_pseudo_controls, all_pseudo_cases):
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

            empirical_kl.append(
                (Metrics._kl_divergence(
                    gp_control, gp_case, case_x)))

            counter += 1

        p_value = utils.calculate_p_value(
            self.kl_divergence(other_metrics), empirical_kl)

        logger.info(
            "The calculated p value for %s is: %f." %
            (self.collection.name, p_value))

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

        all_y_norm = np.append(
            self.collection.obs_seqs_norm,
            control_collection.obs_seqs_norm,
            axis=0)

        for pattern in itertools.product(
                [True, False], repeat=len(all_y_norm)):
            all_pseudo_controls.append(
                [x[1] for x in zip(pattern, all_y_norm) if x[0]])
            all_pseudo_cases.append(
                [x[1] for x in zip(pattern, all_y_norm) if not x[0]])

        all_pseudo_controls = [
            x for x in all_pseudo_controls if control_num_replicates == len(x)]
        all_pseudo_cases = [
            x for x in all_pseudo_cases if case_num_replicates == len(x)]

        return all_pseudo_controls, all_pseudo_cases

    @staticmethod
    def _centre_around_start(y, start_index):
        """Centre data around value of starting index

        Arguments:
            y {np.array} -- Observation sequence
            start_index {int} -- Start time
        """
        return y - y[start_index]

    @staticmethod
    def _relativize_around_start(y, start_index):
        """Normalize each observation value by first observation value

        Arguments:
            y {np.array} -- Observation sequence
            start_index {int} -- Start time
        """
        return y / y[start_index] - 1

    @staticmethod
    def _compute_angle(x, y, start_index):
        min_length = min(len(x), len(y))
        model = sm.OLS(y[start_index:min_length],
                       x[start_index:min_length])
        results = model.fit()
        return np.arctan(results.params[0])

    def angles(self, start_index: int = 0):
        """
        Calculates a very simple angle for each series of observations.

        Arguments:
            start_index {int} -- Where to begin fitting linear model
        """

        angles = {}
        x = self.collection.obs_times
        y = self.collection.obs_seqs
        num_obs_seqs = y.shape[0]

        for i in range(num_obs_seqs):
            centred_y_i = Metrics._centre_around_start(y[i], start_index)
            angles[i] = Metrics._compute_angle(
                x.ravel(), centred_y_i, start_index)

        return angles

    def average_angle(
            self,
            method: str = 'centre',
            start_index: int = 0) -> float:
        """Returns the average angles of all of the obs_seqs for the collection.

        Keyword Arguments:
            method {str} -- Can be either 'centre' or 'relativize' (default: {'centre'})
            start_index {int} -- [description] (default: {0})

        Returns:
            float -- The angle fit to the observations after some scaling
        """
        x = self.collection.obs_times
        y = self.collection.obs_seqs

        if method is 'centre':
            y_means = Metrics._centre_around_start(
                np.nanmean(y, axis=0), start_index)
        elif method is 'relativize':
            y_means = Metrics._relativize_around_start(
                np.nanmean(y, axis=0), start_index)
        else:
            raise UnsupportedScalingError(
                "The given method: \"{}\", is not supported.".format(method))

        return Metrics._compute_angle(x.ravel(), y_means, start_index)

    @staticmethod
    def _calculate_AUC(x, y):
        area = 0
        l = min(len(x), len(y))
        for j in range(l - 1):
            area += (y[j + 1] - y[j]) / (x[j + 1] - x[j])
        return area 

    def auc(self):
        """
        Builds the AUC dict for y
        """
        # TODO: Not sure if we need this

        raise NotImplementedError
    
    def auc_norm(self):
        """
        Builds the AUC dict for y_norm
        """
        # TODO: Not sure if we need this
        raise NotImplementedError

    def gp_auc(self):
        """ 
        Builds AUC of the GP.
        """
        if not self.fit_gp:
            self.fit_gaussian_process()
        
        x = self.collection.obs_times
        return Metrics._calculate_AUC(x, self.fit_gp.predict(x)[0])

    # TODO: Things specific to cancer growth should be separate
    def mrecist(self, start_index: int = 0):
        """Builds the mRECIST dict.

        - **mCR**: BestResponse < -95% AND BestAverageResponse < -40%
        - **mPR**: BestResponse < -50% AND BestAverageResponse < -20%
        - **mSD**: BestResponse < 35% AND BestAverageResponse < 30%
        - **mPD**: everything else
        """
        x = self.collection.obs_times
        y = self.collection.obs_seqs
        num_obs_seqs = y.shape[0]

        mrecist = {}
        for i in range(num_obs_seqs):
            days_volume = zip(x, y[i])
            days = x 

            initial_volume = y[i][start_index]

            responses = []
            average_responses = []

            day_diff = 0

            for day, volume in days_volume:
                day_diff = day - start_index
                if day >= start_index and day_diff >= 3:
                    responses.append(((volume - initial_volume) / initial_volume) * 100)
                    average_responses.append(np.average(responses))

            if min(responses) < -95 and min(average_responses) < -40:
                mrecist[i] = 'mCR'
            elif min(responses) < -50 and min(average_responses) < -20:
                mrecist[i] = 'mPR'
            elif min(responses) < 35 and min(average_responses) < 30:
                mrecist[i] = 'mSD'
            else:
                mrecist[i] = 'mPD'

        return mrecist

    def credible_intervals(self):
        # TODO
        if self.fit_gp:
            pass
        else:
            self.fit_gaussian_process()

    def percent_credible_intervals(self):
        # TODO
        if self.fit_gp:
            pass
        else:
            self.fit_gaussian_process()

    def write_metrics(self, out_path):
        # TODO: This is the interface that will be used for the web UI
        raise NotImplementedError
