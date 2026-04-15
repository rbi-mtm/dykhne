"""AAA (Adaptive Antoulas-Anderson) rational approximation"""

from scipy.interpolate import AAA

from dykhne.fit.fit import Fit


class FitAAA(Fit, fit_name='aaa'):
    """
    Fitting via the AAA algorithm (``scipy.interpolate.AAA``).

    AAA constructs a rational approximant from real-axis samples and can be
    evaluated at complex points, making it well-suited for analytic
    continuation and locating branch-point singularities in the complex plane.

    Parameters
    ----------
    **kwargs
        Overrides for ``_default_fit_parameters``; passed directly to
        ``scipy.interpolate.AAA``.

    Examples
    --------
    >>> import numpy as np
    >>> from dykhne.fit.fit_aaa import FitAAA
    >>> t = np.linspace(-3, 3, 200)
    >>> y = np.sqrt(1 + t**2)           # real-axis data
    >>> f = FitAAA()
    >>> f.fit(t, y)
    >>> f(1j)                            # analytic continuation to Im axis
    >>> f.roots()                        # zeros of the approximant
    >>> f.poles()                        # poles of the approximant

    """

    @classmethod
    def _define_default_fit_parameters(cls):
        """
        Default parameters for ``scipy.interpolate.AAA``.

        ``rtol``         — relative tolerance for the stopping criterion.
        ``max_terms``    — maximum number of support points.
        ``clean_up``     — whether to remove spurious pole-zero pairs.
        ``clean_up_tol`` — tolerance for the clean-up step.

        """

        cls._default_fit_parameters = {
            'rtol': None,
            'max_terms': 100,
            'clean_up': True,
            'clean_up_tol': 1e-13,
        }

    def __init__(self, **kwargs):
        """
        Initialise with optional parameter overrides.

        """

        super().__init__(**kwargs)

    def fit(self, x, y):
        """
        Build the AAA rational approximant from real-axis data.

        Parameters
        ----------
        x : array_like
            Sample points (real).
        y : array_like
            Function values at *x*.

        """

        self._fit_result = AAA(x, y, **self._fit_parameters)

    def __call__(self, z):
        """
        Evaluate the AAA approximant at *z*.

        Parameters
        ----------
        z : array_like
            Evaluation points; may be complex for analytic continuation.

        Returns
        -------
        array_like
            Approximated function values at *z*.

        Raises
        ------
        RuntimeError
            If ``fit`` has not been called yet.

        """

        if self._fit_result is None:
            raise RuntimeError("Call fit(x, y) before evaluating.")

        return self._fit_result(z)

    def roots(self):
        """
        Return the zeros of the AAA approximant.

        Returns
        -------
        ndarray of complex
            Zeros of the rational approximant.

        """

        if self._fit_result is None:
            raise RuntimeError("Call fit(x, y) before querying roots.")

        return self._fit_result.roots()

    def poles(self):
        """
        Return the poles of the AAA approximant.

        Returns
        -------
        ndarray of complex
            Poles of the rational approximant.

        """

        if self._fit_result is None:
            raise RuntimeError("Call fit(x, y) before querying poles.")

        return self._fit_result.poles()

    def residues(self):
        """
        Return the residues at the poles of the AAA approximant.

        Returns
        -------
        ndarray of complex
            Residues corresponding to each pole.

        """

        if self._fit_result is None:
            raise RuntimeError("Call fit(x, y) before querying residues.")

        return self._fit_result.residues()
