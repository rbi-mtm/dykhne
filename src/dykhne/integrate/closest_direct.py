"""Integration from the real axis directly to the nearest complex zero"""

import numpy as np
from scipy.integrate import quad_vec

from dykhne.integrate.integrate import Integrate


class ClosestDirect(Integrate, integrate_name='closest_direct'):
    """
    Integrate the fitted function along a vertical path from the real axis to
    the complex zero closest to the real axis.

    The contour is::

        z(y) = Re(z0) + i*y,   y in [0, Im(z0)]

    where ``z0`` is the selected root.  The real part of the path is held
    constant at ``Re(z0)`` throughout, so the path is a straight vertical
    segment.

    Parameters
    ----------
    fit : Fit
        A trained ``Fit`` instance.
    **kwargs
        Overrides for ``_default_integrate_parameters``.

    Examples
    --------
    >>> integrator = ClosestDirect(fit, upper_half_plane=True)
    >>> integral, root = integrator.integrate()

    """

    @classmethod
    def _define_default_integrate_parameters(cls):
        """
        Default parameters for ``ClosestDirect``.

        ``upper_half_plane`` — if ``True``, restrict candidate roots to
        those with ``Im(z) >= 0`` before selecting the closest one.

        """

        cls._default_integrate_parameters = {
            'upper_half_plane': True,
        }

    def __init__(self, fit, **kwargs):
        """
        Initialise with an optional ``upper_half_plane`` override.

        """

        super().__init__(fit, **kwargs)

    def _nearest_root(self):
        """
        Return the root closest to the real axis.

        Raises
        ------
        ValueError
            If no roots exist in the selected half-plane.

        """

        roots = self._fit.roots()

        if self._integrate_parameters['upper_half_plane']:
            roots = roots[roots.imag >= 0]

        if len(roots) == 0:
            raise ValueError("No roots found in the selected half-plane.")

        return roots[np.argmin(np.abs(roots.imag))]

    def integrate(self):
        """
        Integrate from the real axis to the nearest complex zero.

        The path is a vertical segment at constant ``Re(z0)`` from ``y = 0``
        to ``y = Im(z0)``.

        Returns
        -------
        integral : complex
            Value of the contour integral.
        root : complex
            The zero ``z0`` used as the endpoint of the path.

        """

        root = self._nearest_root()

        x0 = root.real
        y0 = root.imag

        def integrand(y):
            z = x0 + 1j * y
            return self._fit(z) * 1j

        integral, _ = quad_vec(integrand, 0.0, y0)

        return integral, root
