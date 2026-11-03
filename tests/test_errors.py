import pituplots


def test_error_bases():
    assert issubclass(pituplots.MissingDependency, ImportError)
    assert issubclass(pituplots.ConsistencyError, ValueError)
    assert issubclass(pituplots.PituplotsWarning, Warning)
