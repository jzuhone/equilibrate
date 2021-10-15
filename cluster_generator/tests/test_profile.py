from cluster_generator.tests.utils import profile_answer_testing, \
    generate_profile


def test_profile(answer_store, answer_dir):
    p, vd, vs = generate_profile()
    profile_answer_testing(p, "profile.h5", answer_store, answer_dir)
    profile_answer_testing(vd, "virial_dm.h5", answer_store, answer_dir)
    profile_answer_testing(vs, "virial_star.h5", answer_store, answer_dir)