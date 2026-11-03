import importlib


class MissingDependency(ImportError):
    """An optional package is needed. The message names the extra."""


class ConsistencyError(ValueError):
    """OpenQuake realizations, weights and mean curves disagree."""


class PituplotsWarning(UserWarning):
    """The figure is valid but incomplete."""


def need(module, extra):
    """Import an optional package, or raise naming its extra."""
    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise MissingDependency(f'{module} is needed: pip install "pituplots[{extra}]"') from e
