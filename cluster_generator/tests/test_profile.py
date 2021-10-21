from cluster_generator.tests.utils import model_answer_testing, \
    generate_model


def test_model(answer_store, answer_dir):
    m = generate_model()
    model_answer_testing(m, "profile.h5", answer_store, answer_dir)
    #profile_answer_testing(vd, "virial_dm.h5", answer_store, answer_dir)
    #profile_answer_testing(vs, "virial_star.h5", answer_store, answer_dir)