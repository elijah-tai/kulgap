from kulgap.collection import Collection
from kulgap.errors import TimeValidationError

import pytest


def test_raise_timevalidationerror_when_obs_times_not_list():
    with pytest.raises(TimeValidationError):
        c = Collection('testName', obs_times=2, obs_seqs=[0])
