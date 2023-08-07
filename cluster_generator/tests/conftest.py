"""
``cluster_generator``'s ``Pytest`` configuration tools.
"""
import pytest
import os
from .utils import generate_model

def pytest_addoption(parser):
    """Adds ``--answer_dir`` and ``--answer_store`` to the command line arguments."""
    parser.addoption("--answer_dir",
                     help="Directory where answers are stored.")
    parser.addoption("--answer_store", action="store_true",
                     help="Generate new answers, but don't test.")

def pytest_collection_modifyitems(session,config,items):
    _module_order = ["cluster_generator.tests."+i for i in ["test_particles","test_profile","test_gravity","test_ics"]]
    _doc_test,its = [it for it in items if isinstance(it,pytest.DoctestItem)],[it for it in items if not isinstance(it,pytest.DoctestItem)]

    print(f"\nFound \u001b[31m{len(_doc_test)}\u001b[0m doctests: \u001b[35m{_doc_test}\u001b[0m")
    print(f"Found \u001b[31m{len(its)}\u001b[0m tests: \u001b[35m{its}\u001b[0m")
    module_mapping = {item: item.module.__name__ for item in its}


    sorted_items = its.copy()

    for module in _module_order:
        sorted_items = [it for it in sorted_items if module_mapping[it] != module] + [it for it in sorted_items if module_mapping[it] == module]

    its[:] = sorted_items

    items[:] = its + _doc_test
    print(f"\nPrescribed order \u001b[35m{_module_order}\u001b[0m.")
    print(f"Asserted order \u001b[35m{[item.module.__name__.replace('cluster_generator.tests.','') for item in its] + _doc_test}\u001b[0m.\n")

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
