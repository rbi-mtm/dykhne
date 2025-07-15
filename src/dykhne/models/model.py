"""Abstract base class for models to calculate time-dependent energies"""

import abc

class Model(abc.ABC):
    """
    Base abstract class for a model to calculate time-dependent energies

    """

    def __init_subclass__(cls, model_name):
        """
        Sets a unique model name.

        """

        cls._model_name = model_name
        cls._define_default_model_parameters()

    @classmethod
    @abc.abstractmethod
    def _define_default_model_parameters(cls):
        """
        Defines a dictionary cls._default_model_parameters

        Example dictionary:

            {'a' : 2.0, 'b' : -1.0}

        """

    def __init__(self, **kwargs):
        """
        Sets the model parameters

        Keyword Arguments
        -----------------
        kwargs : dict
            Dictionary of model parameters

        """

        self._model_parameters = kwargs

    @abc.abstractmethod
    def energy_levels(self):
        """
        Returns a dictionary of analytic functions of the time-dependent energy levels

        The levels should be labeled starting with 0:

        {'0': a + b*t, '1': c + d*t**2, ...}

        Keyword Arguments
        -----------------
        kwargs : dict
            Dictionary of parameter values

        Returns
        -------
        levels : dict
            Dictionary of analytic functions defining time-dependent energy levels

        """
