"""Abstract base class for fitting numerical data to functions"""

import abc


class Fit(abc.ABC):
    """
    Base abstract class for fitting numerical data to a function.

    Subclasses implement a specific fitting algorithm (e.g. AAA, polynomial)
    and may be evaluated at arbitrary (including complex) points to perform
    analytic continuation using only real-axis data.

    Usage pattern
    -------------
    Subclass ``Fit`` and pass a unique ``fit_name`` keyword in the class
    definition::

        class MyFit(Fit, fit_name='my_algorithm'):
            ...

    Then instantiate with optional algorithm parameters, call ``fit`` to
    train on data, and call the object to evaluate::

        f = MyFit(tol=1e-10)
        f.fit(x, y)
        values = f(z)          # z may be complex

    """

    def __init_subclass__(cls, fit_name):
        """
        Registers a unique algorithm name on the subclass.

        """

        cls._fit_name = fit_name
        cls._define_default_fit_parameters()

    @classmethod
    @abc.abstractmethod
    def _define_default_fit_parameters(cls):
        """
        Define ``cls._default_fit_parameters`` as a dict of algorithm
        parameters and their default values.

        Example::

            cls._default_fit_parameters = {'tol': 1e-13, 'mmax': 100}

        """

    def __init__(self, **kwargs):
        """
        Store algorithm parameters, falling back to defaults for any that
        are not supplied.

        Keyword Arguments
        -----------------
        **kwargs
            Algorithm parameters that override ``_default_fit_parameters``.

        """

        self._fit_parameters = {**self._default_fit_parameters, **kwargs}
        self._fit_result = None

    @abc.abstractmethod
    def fit(self, x, y):
        """
        Fit the supplied real-axis data.

        Parameters
        ----------
        x : array_like
            Sample points (real).
        y : array_like
            Function values at *x*.

        """

    @abc.abstractmethod
    def __call__(self, z):
        """
        Evaluate the fitted function at *z*.

        Parameters
        ----------
        z : array_like
            Evaluation points; may be complex for analytic continuation.

        Returns
        -------
        array_like
            Approximated function values at *z*.

        """

    @abc.abstractmethod
    def roots(self):
        """
        Return the complex zeros of the fitted function.

        Returns
        -------
        ndarray of complex
            Zeros of the fitted function in the complex plane.

        """
