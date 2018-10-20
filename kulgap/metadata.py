class Metadata:
    """
    Not required to fill any of this out, but helpful for
    keeping some annotations about the Collection.
    """

    def __init__(
            self,
            obs_start=None,
            obs_end=None,
            gp_start=None,
            notes_dict=None):
        self.obs_start = obs_start
        self.obs_end = obs_end
        self.gp_start = gp_start

        # could put tumour type and phlc_sample here
        self.notes_dict = notes_dict
