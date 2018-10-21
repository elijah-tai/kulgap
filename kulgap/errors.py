class TimeValidationError(Exception):
    def __init__(self, message):
        super().__init__(message)


class UnsupportedMetricError(Exception):
    def __init__(self, message):
        super().__init__(message)


class UnsupportedScalingError(Exception):
    def __init__(self, message):
        super().__init__(message)
