class MissingDependency(ImportError):
    """An optional package is needed. The message names the extra."""


class ConsistencyError(ValueError):
    """OpenQuake realizations, weights and mean curves disagree."""


class PituplotsWarning(UserWarning):
    """The figure is valid but incomplete."""
