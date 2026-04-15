"""Abstract base class for integrating fitted functions in the complex plane"""

import abc

from dykhne.fit.fit import Fit


class Integrate(abc.ABC):
    """
    Base abstract class for integrating a fitted function along a contour in
    the complex plane.

    Subclasses implement a specific integration strategy (e.g. integrating
    directly to the nearest complex zero) and receive a ``Fit`` instance that
    supplies both the function values and its complex zeros.

    Usage pattern
    -------------
    Subclass ``Integrate`` and pass a unique ``integrate_name`` keyword in the
    class definition::

        class MyIntegrate(Integrate, integrate_name='my_strategy'):
            ...

    Then instantiate with a fitted ``Fit`` object and optional parameters, and
    call ``integrate`` to obtain the result::

        integrator = MyIntegrate(fit, upper_half_plane=True)
        result = integrator.integrate()

    """

    def __init_subclass__(cls, integrate_name):
        """
        Registers a unique integration strategy name on the subclass.

        """

        cls._integrate_name = integrate_name
        cls._define_default_integrate_parameters()

    @classmethod
    @abc.abstractmethod
    def _define_default_integrate_parameters(cls):
        """
        Define ``cls._default_integrate_parameters`` as a dict of strategy
        parameters and their default values.

        Example::

            cls._default_integrate_parameters = {'upper_half_plane': True}

        """

    def __init__(self, fit, **kwargs):
        """
        Store the fitted function and integration parameters.

        Parameters
        ----------
        fit : Fit
            A trained ``Fit`` instance whose ``__call__`` evaluates the
            function and whose ``roots`` returns its complex zeros.
        **kwargs
            Integration parameters that override
            ``_default_integrate_parameters``.

        """

        if not isinstance(fit, Fit):
            raise TypeError(f"fit must be a Fit instance, got {type(fit)}")

        self._fit = fit
        self._integrate_parameters = {
            **self._default_integrate_parameters,
            **kwargs,
        }

    @abc.abstractmethod
    def integrate(self):
        """
        Perform the contour integration.

        Returns
        -------
        integral : complex
            Value of the contour integral.

        """
