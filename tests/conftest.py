import socket
from pathlib import Path

import cartopy
import matplotlib
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

DATA = Path(__file__).parent / "data"


def pytest_configure(config):
    cartopy.config["pre_existing_data_dir"] = str(DATA / "natural_earth")


@pytest.fixture(scope="session", autouse=True)
def empty_cartopy_cache(tmp_path_factory):
    cartopy.config["data_dir"] = str(tmp_path_factory.mktemp("cartopy"))
    cartopy.config["cache_dir"] = str(tmp_path_factory.mktemp("cartopy_cache"))


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*a, **k):
        raise OSError("tests must not use the network")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="session")
def data():
    return DATA
