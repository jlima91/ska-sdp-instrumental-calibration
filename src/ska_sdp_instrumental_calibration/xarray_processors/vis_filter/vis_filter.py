from abc import ABC, abstractclassmethod

import xarray as xr


class VisibilityFilter(ABC):
    """
    Abstract base class for implementing specific visibility filters.

    Classes inheriting from this must define a `_FILTER_NAME_` attribute to be
    registered by the `VisibilityFilter` metaclass and implement the `filter`
    class method.
    """

    _data_filters = {}

    def __init_subclass__(cls, **kwargs):
        """
        Hook that runs when a new subclass is defined.

        Registers the subclass if it has a `_FILTER_NAME_` attribute.
        """
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "_FILTER_NAME_"):
            cls._data_filters[cls._FILTER_NAME_] = cls

    @abstractclassmethod
    def _filter(cls, filters: str, vis: xr.Dataset):
        """
        Apply specific filtering logic to the dataset.

        This method must be overridden by subclasses to define the specific
        filtering behavior.

        Parameters
        ----------
        filters : str
            The filter expression or configuration string specific to this
            filter implementation.
        vis : xr.Dataset
            The visibility dataset to apply the filter on.

        Returns
        -------
        xr.DataArray
            The updated visibility flag.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError("Filter function not implemented")

    @classmethod
    def filter(cls, filters: dict, vis: xr.Dataset):
        """
        Apply a set of registered filters to the visibility dataset.

        Iterates through the provided dictionary of filters, looks up the
        corresponding strategy in the registry, and updates the dataset's flags

        Parameters
        ----------
        filters : dict
            A dictionary where keys correspond to registered filter names
            (defined via `_FILTER_NAME_` in subclasses) and values are the
            filter expressions or configurations to be passed to the specific
            filter implementation.
        vis : xr.Dataset
            The input visibility dataset containing the data to be filtered.

        Returns
        -------
        xr.Dataset
            The visibility dataset with updated "flag" variables based on the
            applied filters.

        Raises
        ------
        ValueError
            If a key in the `filters` dictionary does not correspond to a
            registered filter strategy.
        """
        filters = {} if filters is None else filters

        for key, filter_expr in filters.items():
            if key not in cls._data_filters:
                raise ValueError(f"Strategy for {key} filter not known")

            if filter_expr is not None:
                vis = vis.assign(
                    {"flag": cls._data_filters[key]._filter(filter_expr, vis)}
                )

        return vis
