"""Abstract base class for adding noise to model energy levels"""

import abc

import numpy as np


class Noise(abc.ABC):
    """
    Base abstract class for adding noise to model energy levels.

    Subclasses implement a specific noise algorithm and receive an
    ``energy_levels`` dict (as returned by ``Model.energy_levels``) together
    with a random seed and optional algorithm parameters.  After construction,
    the noisy levels are available as ``self.noisy_energy_levels``.

    ``energy_levels`` format
    ------------------------
    A ``dict[str, np.ndarray]`` mapping string labels to 1-D arrays of
    energy values evaluated at the same time points::

        {'0': array([...]), '1': array([...]), ...}

    Usage pattern
    -------------
    Subclass ``Noise`` and pass a unique ``noise_name`` keyword::

        class MyNoise(Noise, noise_name='my_algorithm'):
            ...

    Then instantiate with an ``energy_levels`` dict, a ``seed``, and any
    algorithm-specific keyword arguments::

        noise = MyNoise(energy_levels, seed=42, scale=0.1)
        noise.noisy_energy_levels   # dict with same keys, noisy values

    """

    def __init_subclass__(cls, noise_name):
        """
        Registers a unique noise algorithm name on the subclass.

        """

        cls._noise_name = noise_name
        cls._define_default_noise_parameters()

    @classmethod
    @abc.abstractmethod
    def _define_default_noise_parameters(cls):
        """
        Define ``cls._default_noise_parameters`` as a dict of algorithm
        parameters and their default values.

        Example::

            cls._default_noise_parameters = {'scale': 0.1}

        """

    def __init__(self, energy_levels, seed, **kwargs):
        """
        Store the energy levels, seed, and noise parameters, then generate
        noisy levels.

        Parameters
        ----------
        energy_levels : dict[str, np.ndarray]
            Energy levels as returned by ``Model.energy_levels``.  Keys are
            string labels; values are 1-D arrays of the same length.
        seed : int
            Seed for the random number generator, for reproducibility.
        **kwargs
            Algorithm parameters that override ``_default_noise_parameters``.

        """

        self._energy_levels = energy_levels
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._noise_parameters = {
            **self._default_noise_parameters,
            **kwargs,
        }
        self.noisy_energy_levels = self._apply_noise()

    @abc.abstractmethod
    def _apply_noise(self):
        """
        Generate and return the noisy energy levels dict.

        Returns
        -------
        dict[str, np.ndarray]
            A new dict with the same keys as ``_energy_levels`` and noisy
            values of the same shape.

        """
