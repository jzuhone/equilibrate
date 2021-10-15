from cluster_generator.tests.utils import particle_answer_testing, \
    generate_profile
from numpy.random import RandomState


prng = RandomState(25)


def test_particles(answer_store, answer_dir):
    p, vd, vs = generate_profile()
    dp = vd.generate_particles(100000, prng=prng)
    sp = vs.generate_particles(100000, prng=prng)
    hp = p.generate_particles(100000, prng=prng)
    parts = hp+dp+sp
    particle_answer_testing(parts, "particles.h5", answer_store, answer_dir)