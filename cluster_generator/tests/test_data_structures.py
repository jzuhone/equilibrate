"""
Testing module for the :py:mod:``data_structures`` module.
"""
import os

import pytest

from cluster_generator.data_structures import YTHDF5
from cluster_generator.tests.utils import generate_model


@pytest.mark.usefixtures("answer_dir", "answer_store")
class Test_YTHDF5:
    """
    Tests for the :py:class:`data_structures.YTHDF5`` class.
    """

    @pytest.fixture()
    def class_objects(self, answer_dir, answer_store):
        # Produce the model and crate the correct dataset.
        model = generate_model()
        model.write_model_to_h5(
            os.path.join(answer_dir, "YTHDF5_model_tmp.h5"), overwrite=True
        )
        model.create_dateset(os.path.join(answer_dir, "YTHDF5_struct_tmp.h5"))

        # Yield the files.
        yield os.path.join(answer_dir, "YTHDF5_model_tmp.h5"), os.path.join(
            answer_dir, "YTHDF5_struct_tmp.h5"
        )

        # Teardown
        if answer_store:
            os.system(
                f"mv {os.path.join(answer_dir,'YTHDF5_struct_tmp.h5')} {os.path.join(answer_dir,'YTHDF5_struct_answer.h5')}"
            )
        else:
            os.remove(os.path.join(answer_dir, "YTHDF5_struct_tmp.h5"))

        os.remove(os.path.join(answer_dir, "YTHDF5_model_tmp.h5"))

    def test_construction(self, answer_dir, answer_store, class_objects):
        """
        Test the construction process. Only checks that the attributes are the same across to enforce reasonable runtimes.
        """

        mpath, dpath = class_objects

        if not answer_store:
            # We need to actually check.
            answer_path = os.path.join(answer_dir, "YTHDF5_struct_answer.h5")

            ans_struct, new_struct = YTHDF5(answer_path), YTHDF5(dpath)

            with ans_struct.open(mode="r") as ans_fo, new_struct.open(
                mode="r"
            ) as new_fo:
                for k, v in ans_fo.attrs.items():
                    assert (
                        k in new_fo.attrs.keys()
                    ), f"Attribute {k} is not in the new structure's attribute list."
                    assert (
                        new_fo.attrs[k] == v
                    ), f"Attribute {k} appears to have changed between old and new test answers: {v} -> {new_fo.attrs[k]}."
