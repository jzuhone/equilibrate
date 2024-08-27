"""
Configuration and Fixture Setup
================================

This module configures the pytest environment for the `cluster_generator` testing suite. It provides custom
command-line options, fixtures for setting up and tearing down test environments, and utility functions to manage
complex test scenarios, such as ordering tests, packaging results, and fetching remote test data.

Command Line Options
--------------------

- Adds several command-line options to pytest to control test behavior:
  - ``--answer_dir``: Specifies the directory where test answers are stored.
  - ``--answer_store``: Enables generation of new answers without running tests.
  - ``--tmp``: Defines the temporary directory to use during tests. If not provided, a ``/tmp`` directory is used.
  - ``--package``: Packages the test results for external use.
  - ``--fetch_remote``: Fetches answers from a remote directory.
  - ``--remote_file``: Specifies an explicit remote file to fetch.

Example
-------
.. code-block:: bash

    pytest --answer_dir="./test_answers" --answer_store --package

This command will store test answers in the ``./test_answers`` directory, generate new answers, and package the
results into a distributable format.

How to Use This Configuration
-----------------------------
1. **Modify the `test_configuration.yaml` File**:
   - Ensure the configuration file is properly set up with all necessary settings. Example content:

   .. code-block:: yaml

       REMOTE:
         HOST: "https://astro.utah.edu/~u1281896/cluster_generator"

       SETUP:
         RUN_BEFORE_ALL:
           - ["cluster_generator.tests.test_model", "test_model_build"]

2. **Run Pytest with Custom Options**:
   - Use the provided command-line options to customize test runs according to your needs.

   Example
   -------
   .. code-block:: bash

       pytest --answer_dir="./test_answers" --fetch_remote

   This command will fetch test answers from the remote host specified in the configuration and extract them into the
   ``./test_answers`` directory.

3. **Package Results for Distribution**:
   - Enable the ``--package`` option to automatically compress test results into a tarball for sharing or archiving.

   Example
   -------
   .. code-block:: bash

       pytest --answer_dir="./test_answers" --answer_store --package

   This command will package all files located in the ``./test_answers`` directory into a tarball with a name based on
   the operating system, Python version, and ``cluster_generator`` version.
"""
import os
from pathlib import Path
from typing import Generator, List

import pytest
from _pytest.config.argparsing import Parser
from _pytest.main import Session
from _pytest.nodes import Item

from cluster_generator.tests.utils.environment import download_file, load_config
from cluster_generator.utils import cgparams

# INFRASTRUCTURE SETUP
# --------------------
#
# This section of the conftest.py file loads configuration files
# for the testing environment and incorporates them into the testing infrastructure.
# We load the fixtures, enforce test order, etc.
CONFIG_FILE: str = os.path.join(Path(__file__).parents[0], "test_configuration.yaml")
CONFIG: dict = load_config(CONFIG_FILE)

# Turning off progress bars to maintain efficiency when non-terminal interfaces are used
cgparams["system"]["display"]["progress_bars"] = False

# Setting various configuration settings.
_REMOTEHOST: str = CONFIG["REMOTE"]["HOST"]
""" str: The remote host from which to fetch the testing answers.

Cluster generator seeks answers with the format ``answers_{OS}_{pyversion}_{cgversion}.tar.gz``.
"""


# TEST ORDERING:
# In some cases, tests have to run in a particular order to ensure that structures are
# generated in time for a test that requires them. To facilitate this, the CONFIG has a
# RUN_BEFORE_ALL list for adding tests that must run before any other tests can start.


def pytest_collection_modifyitems(
    session: Session, config: pytest.Config, items: List[Item]
) -> None:
    """
    Modify the order of test items collected by pytest to optimize test performance.

    This function ensures that certain tests are run in a specific order to avoid redundant
    operations, such as rebuilding models for every test.

    Args:
        session (Session): The pytest session object.
        config (pytest.Config): The pytest configuration object.
        items (List[Item]): The list of collected test items.

    Returns:
        None
    """
    # Use the RUN_BEFORE_ALL setting from the YAML configuration file to enforce test ordering
    _enforced_test_ordering = [
        (test[0], test[1]) for test in CONFIG["SETUP"]["RUN_BEFORE_ALL"]
    ]

    _doc_tests, _standard_tests = (
        [test for test in items if isinstance(test, pytest.DoctestItem)],
        [test for test in items if not isinstance(test, pytest.DoctestItem)],
    )

    _test_dictionary = {
        (test.module.__name__, test.name): test for test in _standard_tests
    }

    items[:] = (
        [
            _test_dictionary.pop(test_name)
            for test_name in _enforced_test_ordering
            if test_name in _test_dictionary
        ]
        + list(_test_dictionary.values())
        + _doc_tests
    )


def pytest_addoption(parser: Parser) -> None:
    """
    Add custom command-line options to pytest for controlling test behavior.

    Args:
        parser (Parser): The pytest parser object.

    Returns:
        None
    """
    parser.addoption("--answer_dir", help="Directory where answers are stored.")
    parser.addoption(
        "--answer_store",
        action="store_true",
        help="Generate new answers, but don't test.",
    )
    parser.addoption("--tmp", help="The temporary directory to use.", default=None)
    parser.addoption(
        "--package",
        help="Package the results of this test for use externally.",
        action="store_true",
    )
    parser.addoption(
        "--fetch_remote",
        help="Fetch answers from remote directory.",
        action="store_true",
    )
    parser.addoption(
        "--remote_file",
        help="Explicitly set the remote file to fetch.",
        default=None,
    )


@pytest.fixture(scope="session", autouse=True)
def packager(request: pytest.FixtureRequest) -> Generator[None, None, None]:
    """
    Package test results into a ``.tar.gz`` file for external use after all tests have been executed.

    This fixture runs automatically for the entire testing session (``scope="session"``) and is utilized
    for packaging the results of tests into a compressed file format. The packaging occurs only if both
    the ``--package`` and ``--answer_store`` options are enabled when running pytest.

    Notes
    -----
    **Output Management**: To ensure clean output in environments like CI/CD systems, the ``capmanager`` plugin
     is used to manage standard output and disable it globally while packaging. Effectively overriding pytest's
     default output style.

    Parameters
    ----------
    request : :py:class:`pytest.FixtureRequest`
        The fixture request object that provides access to the configuration options set by the user via
        the command line.

    Yields
    ------
    None
        The fixture yields control back to pytest after setting up, then resumes after all tests have finished running.


    Example
    -------
    .. code-block:: python

        pytest --answer_dir="./test_answers" --answer_store --package

    This will package all files located in the ``./test_answers`` directory into a tarball with a name
    based on the operating system, Python version, and ``cluster_generator`` version.

    """
    import sys
    import tarfile
    from importlib.metadata import version
    from pathlib import Path
    from platform import system

    capmanager = request.config.pluginmanager.getplugin("capturemanager")

    # Fetch the options
    _answer_dir = Path(os.path.abspath(request.config.getoption("--answer_dir")))
    _answer_store = request.config.getoption("--answer_store")
    _package = request.config.getoption("--package")

    yield None

    if not _package:
        return None

    if not _answer_store:
        print("\0337", end="", flush=True)
        print(
            "\n[cluster_generator tests]: Skipping --package directive because --answer_store was False."
        )
        print("\0338", end="", flush=True)
        return None

    # Fetching system info
    _system = system()
    _py_version = sys.version.split(" ")[0]
    _cg_version = version("cluster_generator")

    package_name = f"cg_answers_{_system}_{_py_version}_{_cg_version}.tar.gz"

    with tarfile.open(package_name, "w:gz") as tar:
        for file in os.listdir(_answer_dir):
            tar.add(os.path.join(_answer_dir, file), arcname=file)

    with capmanager.global_and_fixture_disabled():
        print("\0337", end="", flush=True)
        print(
            f"\n[cluster_generator tests]: Packaged answers for distribution: {package_name}.",
            end="\n",
        )
        print("\0338", end="", flush=True)


@pytest.fixture(scope="session", autouse=True)
def fetch_remote(request: pytest.FixtureRequest) -> Generator[None, None, None]:
    """
    Load remote answers into the local answer directory if required.

    This fixture fetches test answers from a remote host and unpacks them into the local answer
    directory. It is triggered automatically for the entire testing session (``scope="session"``) and
    runs before any tests are executed. The fetching process is conditional, based on the options
    provided when running pytest:

    - The fixture fetches answers if the ``--fetch_remote`` option is enabled.
    - It will only proceed with fetching if ``--answer_store`` is **not** enabled, to prevent overwriting
      locally stored answers.

    **Conditional Fetching**: If the conditions for fetching are met, the fixture constructs the appropriate file
     name and remote  path for the answers based on the system's operating system,
     Python version, and ``cluster_generator`` version.


    Parameters
    ----------
    request : :py:class:`pytest.FixtureRequest`
        The fixture request object that provides access to the configuration options set by the user via
        the command line.

    Yields
    ------
    None
        The fixture yields control back to pytest after setting up, then resumes after the pre-test execution
        phase is complete.

    Example
    -------
    .. code-block:: python

        pytest --answer_dir="./test_answers" --fetch_remote

    This command will fetch the answers from the remote host specified in the configuration and extract
    them into the ``./test_answers`` directory.

    Raises
    ------
    :py:class:`pytest.UsageError`
        If the remote file download fails, a ``pytest.UsageError`` is raised to indicate an issue with
        the fetching process.

    """
    import sys
    import tarfile
    from importlib.metadata import version
    from pathlib import Path
    from platform import system

    capmanager = request.config.pluginmanager.getplugin("capturemanager")

    # Fetch the options from the command line
    _answer_dir = Path(os.path.abspath(request.config.getoption("--answer_dir")))
    _answer_store = request.config.getoption("--answer_store")
    _fetch_remote = request.config.getoption("--fetch_remote")
    _remote_file = request.config.getoption("--remote_file")

    if (not _answer_store) and _fetch_remote:
        # The fetching process will proceed. Determining the
        # relevant parameters to ensure that the correct package is
        # fetched.
        _system = system()
        _py_version = sys.version.split(" ")[0]
        _cg_version = version("cluster_generator")

        # Determine the package name to fetch
        if _remote_file is None:
            package_name = f"cg_answers_{_system}_{_py_version}_{_cg_version}.tar.gz"
        else:
            package_name = f"{_remote_file}.tar.gz"

        remote_path = os.path.join(_REMOTEHOST, package_name)

        with capmanager.global_and_fixture_disabled():
            print("\0337", end="", flush=True)
            print(
                f"\n[cluster_generator tests]: Attempting to fetch answers from {remote_path}..."
            )
            print("\0338", end="", flush=True)

            # Ensure the answer directory exists
            if not os.path.exists(_answer_dir):
                Path(_answer_dir).mkdir(parents=True, exist_ok=True)

            try:
                download_file(remote_path, os.path.join(_answer_dir, package_name))
            except Exception as e:
                raise pytest.UsageError(
                    f"Failed to download answers {remote_path}. Error = {e.__repr__()}"
                )

            print("\0337", end="", flush=True)
            print(
                f"\n[cluster_generator tests]: Obtained answers for distribution: {package_name}."
            )
            print("\0338", end="", flush=True)

            # Extract the answers into the answer directory
            with tarfile.open(os.path.join(_answer_dir, package_name), "r:gz") as tar:
                tar.extractall(path=_answer_dir)
            print("\0337", end="", flush=True)
            print(
                f"\n[cluster_generator tests]: Dumped answers into {_answer_dir}. Ready to proceed."
            )
            print("\0338", end="", flush=True)
    yield None


@pytest.fixture()
def answer_store(request) -> bool:
    """Fetches the ``--answer_store`` option."""
    return request.config.getoption("--answer_store")


@pytest.fixture()
def answer_dir(request) -> str:
    """Fetches the ``--answer_dir`` option."""
    ad = os.path.abspath(request.config.getoption("--answer_dir"))
    if not os.path.exists(ad):
        os.makedirs(ad)
    return ad


@pytest.fixture()
def temp_dir(request) -> str:
    """Pull the temporary directory.

    If this is specified by the user, then it may be a non-temp directory which is not
    wiped after runtime. If not specified, then a temp directory is generated and wiped
    after runtime.
    """
    td = request.config.getoption("--tmp")

    if td is None:
        from tempfile import TemporaryDirectory

        td = TemporaryDirectory()

        yield td.name

        td.cleanup()
    else:
        yield os.path.abspath(td)
