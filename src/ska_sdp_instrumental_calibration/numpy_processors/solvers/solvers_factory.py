class SolverFactory(type):
    """
    Metaclass for registering and instantiating calibration solvers.

    This factory uses the metaclass mechanism to automatically register any
    class that defines a `_SOLVER_NAME_` attribute. It acts as a central
    registry, allowing for the dynamic retrieval and instantiation of solver
    classes based on string identifiers.

    Attributes
    ----------
    _solvers : dict
        A registry dictionary mapping unique solver names (str) to their
        corresponding classes (type).

    Examples
    --------
    >>> # Assuming GainSubstitution is defined with _SOLVER_NAME_
    >>> solver = SolverFactory.get_solver("gain_substitution", niter=10)
    >>> print(type(solver))
    <class 'GainSubstitution'>
    """

    _solvers = {}

    def __new__(cls, name, bases, attrs):
        """
        Create a new class and register it if a solver name is defined.

        Parameters
        ----------
        name : str
            Name of the class being created.
        bases : tuple
            Base classes of the class being created.
        attrs : dict
            Attributes defined in the class body.

        Returns
        -------
        type
            The newly created class.
        """
        new_class = super(SolverFactory, cls).__new__(cls, name, bases, attrs)
        if "_SOLVER_NAME_" in attrs:
            cls._solvers[attrs["_SOLVER_NAME_"]] = new_class

        return new_class

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
