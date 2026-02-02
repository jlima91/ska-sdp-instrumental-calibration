from typing import Tuple

import numpy as np


class Solver:
    """
    Base class for gain solvers.

    This class provides a common interface and shared configuration for
    calibration solvers. Subclasses should implement the `solve` method
    for specific algorithms. This class also acts as a central registry,
    allowing for the dynamic retrieval and instantiation of solver classes
    based on string identifiers.

    Parameters
    ----------
    niter : int, optional
        Maximum number of iterations for the solver. Default is 50.
    tol : float, optional
        Convergence tolerance for the iterative solver. Default is 1e-6.
    **_ : dict
        Additional keyword arguments (ignored by the base class).

    Attributes
    ----------
    niter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance.

    Class Attributes
    ----------
    _solvers : dict
        A registry dictionary mapping unique solver names (str) to their
        corresponding classes (type).

    Examples
    --------
    >>> # Assuming GainSubstitution is defined with _SOLVER_NAME_
    >>> solver = Solver.get_solver("gain_substitution", niter=10)
    >>> print(type(solver))
    <class 'GainSubstitution'>

    """

    _solvers = {}

    def __init_subclass__(cls, **kwargs):
        """
        Hook that runs when a new subclass is defined.

        Registers the subclass if it has a `_SOLVER_NAME_` attribute.
        """
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "_SOLVER_NAME_"):
            cls._solvers[cls._SOLVER_NAME_] = cls

    @classmethod
    def get_solver(cls, solver="gain_substitution", **kwargs):
        """
        Retrieve and instantiate a solver by name.

        Parameters
        ----------
        solver : str, optional
            The unique identifier of the solver to instantiate.
            Default is "gain_substitution".
        **kwargs
            Keyword arguments passed directly to the solver's constructor.

        Returns
        -------
        object
            An instance of the requested solver class.

        Raises
        ------
        ValueError
            If the requested solver name is not found in the registry.
        """
        if solver not in cls._solvers:
            raise ValueError(
                f"{solver} not definebd."
                f" Supported solvers: {', '.join(cls._solvers)}"
            )
        return cls._solvers[solver](**kwargs)

    def __init__(self, niter=50, tol=1e-6, **_):
        self.niter = niter
        self.tol = tol

    def solve(
        self,
        vis_vis: np.ndarray,
        vis_flags: np.ndarray,
        vis_weight: np.ndarray,
        model_vis: np.ndarray,
        model_flags: np.ndarray,
        gain_gain: np.ndarray,
        gain_weight: np.ndarray,
        gain_residual: np.ndarray,
        ant1: np.ndarray,
        ant2: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the calibration solver (abstract method).

        This method defines the standard signature for all solver
        implementations. It accepts observed data, model data, and initial
        gain estimates, returning updated gains.

        Parameters
        ----------
        vis_vis : np.ndarray
            Observed visibility data. Shape: (time, baseline, freq, pol).
        vis_flags : np.ndarray
            Flags for observed visibilities. Shape matches `vis_vis`.
        vis_weight : np.ndarray
            Weights for observed visibilities. Shape matches `vis_vis`.
        model_vis : np.ndarray
            Model visibility data. Shape matches `vis_vis`.
        model_flags : np.ndarray
            Flags for model visibilities. Shape matches `vis_vis`.
        gain_gain : np.ndarray
            Initial gain estimates. Shape: (time, ant, freq, rec1, rec2).
        gain_weight : np.ndarray
            Storage for gain weights. Shape matches `gain_gain`.
        gain_residual : np.ndarray
            Storage for gain residuals. Shape matches `gain_gain`.
        ant1 : np.ndarray
            Indices of antenna 1 for each baseline. Shape: (nbl,).
        ant2 : np.ndarray
            Indices of antenna 2 for each baseline. Shape: (nbl,).

        Returns
        -------
        tuple of np.ndarray
            A tuple containing:

            - Updated gain array.
            - Gain weights.
            - Gain residuals.

        Raises
        ------
        NotImplementedError
            This method must be overridden by subclasses.
        """
        raise NotImplementedError("solve not implemented")
