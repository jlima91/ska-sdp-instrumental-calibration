import xarray as xr


class VisibilityFilter(type):
    _data_filters = {}

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
        new_class = super(VisibilityFilter, cls).__new__(
            cls, name, bases, attrs
        )
        if "_FILTER_NAME_" in attrs:
            cls._data_filters[attrs["_FILTER_NAME_"]] = new_class

        return new_class

    @classmethod
    def filter(cls, filters: dict, vis: xr.Dataset):
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

        filters = {} if filters is None else filters

        for key, filter_expr in filters.items():
            if key not in cls._data_filters:
                raise ValueError(f"Strategy for {key} filter not known")

            if filter_expr is not None:
                vis = vis.assign(
                    {"flag": cls._data_filters[key].filter(filter_expr, vis)}
                )

        return vis


class AbstractVisibilityFilter(metaclass=VisibilityFilter):

    @classmethod
    def filter(cls, filters: str, vis: xr.Dataset):
        raise NotImplementedError("Filter function not implemented")
