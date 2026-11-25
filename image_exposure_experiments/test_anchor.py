import numpy as np

from image_exposure_experiments.train_anchor import causal_extrapolate


def test_uses_actual_interval_and_known_midpoint():
    records = [{'sequence': 'a', 'order': j} for j in range(3)]
    times = np.array([0, 100_000_000, 300_000_000], dtype=np.int64)
    anchors = np.array([[0.,0,0], [1,0,0], [3,0,0]])
    age = np.tile([0., .05], (3,1))
    p = causal_extrapolate(anchors, records, times, age, 7.8, 1.)
    np.testing.assert_allclose(p[1:,0,0], [1.078,3.078])
    np.testing.assert_allclose(p[1:,1,0], [1.578,3.578])
    np.testing.assert_array_equal(p[0], np.zeros((2,3)))


def test_future_changes_do_not_change_previous_predictions():
    r = [{'sequence':'a','order':j} for j in range(4)]
    t = np.arange(4,dtype=np.int64)*100_000_000
    a = np.zeros((4,3))
    ages = np.full((4,10),.03)
    p = causal_extrapolate(a,r,t,ages,7.8,.5)
    a[2:] = 100
    q = causal_extrapolate(a,r,t,ages,7.8,.5)
    np.testing.assert_array_equal(p[:2],q[:2])


def test_gap_resets_velocity_and_zero_motion_is_exact():
    r = [{'sequence':'a','order':j} for j in (0,1,3)]
    t = np.array([0,100_000_000,300_000_000],dtype=np.int64)
    a = np.array([[0.,0,0],[1.,0,0],[20.,0,0]])
    ages = np.full((3,10),.04)
    p = causal_extrapolate(a,r,t,ages,7.8,1.)
    np.testing.assert_array_equal(p[2],np.tile(a[2],(10,1)))
    q = causal_extrapolate(a,r,t,ages,7.8,0.)
    np.testing.assert_array_equal(q,np.repeat(a[:,None],10,axis=1))
