"""
Testing structure for the collections available in cluster_collections.py
"""
import inspect
import os
import sys

import pytest

import cluster_generator.cluster_collections as cc

_ignored_classes = ["ProtoCluster", "Collection"]
print([k for k in sys.modules if "cluster_generator" in k])


def _locate_collections():
    """Locates the currently implemented collections"""
    class_members = inspect.getmembers(
        sys.modules["cluster_generator.cluster_collections"], inspect.isclass
    )

    collection_names = [
        k[0]
        for k in class_members
        if (k[1].__module__ == "cluster_generator.cluster_collections")
        and (k[0] not in _ignored_classes)
    ]

    return collection_names


@pytest.mark.parametrize("collection_name", _locate_collections())
def test_exists(collection_name):
    """Checks that the necessary files actually exist for each of the collections"""
    import numpy as np

    assert hasattr(cc, collection_name)

    assert os.path.exists(getattr(cc, collection_name)._data)
    assert os.path.exists(getattr(cc, collection_name)._schema_loc)

    cls = getattr(cc, collection_name)()

    n = np.random.randint(0, len(cls))

    cls.clusters[list(cls.clusters.keys())[n]].load(1, 10000)
