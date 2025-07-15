""" Landau-Zener model """

import numpy as np

from dykhne.models.model import Model

class LZ(Model, model_name='lz'):
    """
    Landau-Zener model

    """

    @classmethod
    def _define_default_model_parameters(cls):
        """
        Defines a dictionary cls._default_model_parameters

        """

        cls._default_model_parameters = {
                'a': 1.0,
                'b': 2.0
                }

    def __init__(self, **kwargs):
        """
        Sets the model parameters

        Parameters
        ----------
        **kwargs : dict
            Dictionary of model parameters

        """

        super().__init__(**kwargs)

    def energy_levels(self, t):
        """
        LZ energy levels

        """

        a = self._model_parameters['a']
        b = self._model_parameters['b']

        e0 = - np.sqrt(b**2 + a**2 * t**2)
        e1 = np.sqrt(b**2 + a**2 * t**2)

        levels = {'0': e0, '1': e1}

        return levels
