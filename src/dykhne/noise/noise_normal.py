"""Gaussian (normal) noise for model energy levels"""

import numpy as np

from dykhne.noise.noise import Noise


class NoiseNormal(Noise, noise_name='normal'):
    """
    Add independent Gaussian noise to each energy level.

    For each level, noise is drawn from::

        N(loc, scale, size=len(level))

    using ``numpy.random.Generator.normal`` and added element-wise.

    Parameters
    ----------
    energy_levels : dict[str, np.ndarray]
        Energy levels as returned by ``Model.energy_levels``.
    seed : int
        Seed for ``numpy.random.default_rng``.
    **kwargs
        Overrides for ``_default_noise_parameters``:

        ``loc``   — mean of the Gaussian noise (default ``0.0``).
        ``scale`` — standard deviation of the Gaussian noise (default ``1.0``).

    Examples
    --------
    >>> import numpy as np
    >>> from dykhne.models.lz import LZ
    >>> from dykhne.noise.noise_normal import NoiseNormal
    >>> t = np.linspace(-5, 5, 200)
    >>> levels = LZ(a=1.0, b=1.0).energy_levels(t)
    >>> noisy = NoiseNormal(levels, seed=0, scale=0.05)
    >>> noisy.noisy_energy_levels
    {'0': array([...]), '1': array([...])}

    """

    @classmethod
    def _define_default_noise_parameters(cls):
        """
        Default parameters for ``numpy.random.Generator.normal``.

        ``loc``   — mean of the distribution (default ``0.0``).
        ``scale`` — standard deviation (default ``1.0``).

        """

        cls._default_noise_parameters = {
            'loc': 0.0,
            'scale': 1.0,
        }

    def __init__(self, energy_levels, seed, **kwargs):

        super().__init__(energy_levels, seed, **kwargs)

    def _apply_noise(self):
        """
        Draw Gaussian noise for each level and return the noisy dict.

        Returns
        -------
        dict[str, np.ndarray]
            Noisy energy levels with the same keys and array shapes as the
            input ``energy_levels``.

        """

        loc = self._noise_parameters['loc']
        scale = self._noise_parameters['scale']

        noisy = {}

        for label, level in self._energy_levels.items():
            noise = self._rng.normal(loc=loc, scale=scale, size=level.shape)
            noisy[label] = level + noise

        return noisy
