import numpy as np

from cluster_generator.tests.utils import generate_model, model_answer_testing


def test_model(answer_store, answer_dir):
    m = generate_model()
    model_answer_testing(m, "profile.h5", answer_store, answer_dir)
    assert np.all(m.check_hse() < 1.0e-4)
