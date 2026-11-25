import numpy as np

from filter_stationary_cache import contact_history, select_records, xy_diameter_mm


def record(identifier, xy, kind='slide', order=0, sequence='trial'):
    target = np.zeros((len(xy), 11))
    target[:, :2] = xy
    target[:, 2] = np.arange(len(xy)) * 100  # Z motion must not affect the filter.
    return {'id': identifier, 'kind': kind, 'target': target.tolist(), 'sequence': sequence,
            'order': order, 'events': len(xy), 'file': identifier + '.npz', 'image': identifier + '.png'}


def test_xy_filter_uses_full_window_and_keeps_equality_and_press():
    still = record('still', [(0, 0), (.01, .01)], order=2)
    boundary = record('boundary', [(0, 0), (.05, 0)], order=1)
    returning = record('returning', [(0, 0), (0, .2), (0, 0)], order=0)
    moving_tail = record('moving_tail', [(0, 0), (.5, .5)], order=3)
    press = record('press', [(0, 0)], kind='press')
    retained, removed = select_records([returning, boundary, still, moving_tail, press], .05)
    assert [r['id'] for r in retained] == ['returning', 'boundary', 'press']
    assert [r['id'] for r in removed] == ['still', 'moving_tail']
    assert [r['reason'] for r in removed] == ['first_stationary_xy', 'after_first_stationary_xy']
    assert xy_diameter_mm(returning) == .2


def test_xy_filter_combines_both_axes():
    diagonal = record('diagonal', [(0, 0), (.04, .04)])
    retained, removed = select_records([diagonal], .05)
    assert len(retained) == 1 and not removed


def test_cutoff_is_per_trajectory_and_uses_time_order_not_manifest_order():
    first_still = record('first_still', [(0, 0), (.01, 0)], order=2)
    later_still = record('later_still', [(0, 0), (.02, 0)], order=4)
    earlier_moving = record('earlier_moving', [(0, 0), (.5, 0)], order=1)
    middle_moving = record('middle_moving', [(0, 0), (.5, 0)], order=3)
    other = record('other', [(0, 0), (.5, 0)], order=3, sequence='other_trial')
    retained, removed = select_records([later_still, earlier_moving, middle_moving, first_still, other], .05)
    assert [r['id'] for r in retained] == ['earlier_moving', 'other']
    assert {r['cutoff_order'] for r in removed} == {2}


def test_contact_gap_marks_only_first_event_unknown_without_mutating_source():
    arrays = {'previous_contact': np.ones(10, dtype=np.int8),
              'transition_valid': np.ones(10, dtype=bool),
              'target': np.arange(110).reshape(10, 11)}
    repaired = contact_history(arrays, continuous=False)
    assert repaired['previous_contact'].tolist() == [-1] + [1] * 9
    assert repaired['transition_valid'].tolist() == [False] + [True] * 9
    assert arrays['previous_contact'].tolist() == [1] * 10
    assert arrays['transition_valid'].all()
    np.testing.assert_array_equal(repaired['target'], arrays['target'])
    continuous = contact_history(arrays, continuous=True)
    np.testing.assert_array_equal(continuous['previous_contact'], arrays['previous_contact'])
