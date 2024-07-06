"""
Testing suite for generating and checking the :py:mod:`cluster_generator.model` module.

"""

import os

import numpy as np

from cluster_generator.tests.utils import generate_model, model_answer_testing


def test_model_build(answer_store, answer_dir, temp_dir):
    """
    Test the basic construction mechanisms for a model.

    Parameters
    ----------
    answer_store: bool
        ``True`` to store and not check.
    answer_dir: str
        The location of the answers on disk.
    temp_dir: str
        The temporary directory at which to store artifacts.

    Notes
    -----
    This test generates using ``.from_dens_and_tden``.
    """
    # Construct the model
    m = generate_model()

    # Save model to tmp location -- regardless of answer_store (it's needed by later tests).
    m.write_model_to_h5(os.path.join(temp_dir, "base_model.h5"), overwrite=True)

    # Check the model against the existing base model in the answer directory.
    model_answer_testing(m, "base_model.h5", answer_store, answer_dir)

    # Check HSE
    assert np.all(m.check_hse() < 1.0e-4)
