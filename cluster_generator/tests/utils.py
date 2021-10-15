from pathlib import Path
from cluster_generator.cluster_model import ClusterModel
from numpy.testing import assert_allclose, assert_equal


def answer_testing(model, filename, answer_store, answer_dir):
    p = Path(answer_dir) / filename
    if answer_store:
        model.write_model_to_h5(p, overwrite=True)
    else:
        old_model = ClusterModel.from_h5_file(p)
        for field in old_model.fields:
            assert_equal(old_model[field], model[field])