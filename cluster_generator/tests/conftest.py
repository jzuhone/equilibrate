"""
``cluster_generator``'s ``Pytest`` configuration tools.
"""
import pytest
import os


def pytest_addoption(parser):
    """Adds ``--answer_dir`` and ``--answer_store`` to the command line arguments."""
    parser.addoption("--answer_dir",
                     help="Directory where answers are stored.")
    parser.addoption("--answer_store", action="store_true",
                     help="Generate new answers, but don't test.")


@pytest.fixture()
def answer_store(request):
    """fetches the ``--answer_store`` option."""
    return request.config.getoption('--answer_store')


@pytest.fixture()
def answer_dir(request):
    """fetches the ``--answer_dir`` option."""
    ad = os.path.abspath(request.config.getoption('--answer_dir'))
    if not os.path.exists(ad):
        os.makedirs(ad)
    return ad
