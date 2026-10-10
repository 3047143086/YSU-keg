"""Strict structure checks and descriptive summaries, independent of Torch."""
import math
import statistics


def distribution(values, allow_undefined=False):
    if not values:
        raise ValueError('An error distribution is required')
    finite = [value for value in values if value is not None]
    if (not allow_undefined and len(finite) != len(values)) or any(
            not isinstance(value, (int, float)) or not math.isfinite(value) for value in finite):
        raise ValueError('A complete finite error distribution is required')
    return {'count': len(values), 'undefined_count': len(values) - len(finite),
            'minimum': min(finite) if finite else None, 'median': statistics.median(finite) if finite else None,
            'maximum': max(finite) if finite else None, 'nonzero_count': sum(value != 0 for value in finite)}


def family(comparisons):
    if not comparisons or any(not isinstance(row.get('exact'), bool) or not isinstance(row.get('passed'), bool)
                              for row in comparisons):
        raise ValueError('Complete exactness and inherited comparison columns are required')
    if any('relative_frobenius_error' not in row or (row['relative_frobenius_error'] is None and row['passed'])
           for row in comparisons):
        raise ValueError('An undefined relative error cannot pass the inherited comparison')
    return {'count': len(comparisons), 'exact_count': sum(row['exact'] for row in comparisons),
            'nonexact_count': sum(not row['exact'] for row in comparisons),
            'exceeds_inherited_bounds_count': sum(not row['passed'] for row in comparisons),
            'relative_frobenius_error': distribution([row['relative_frobenius_error'] for row in comparisons], True),
            'maximum_absolute_error': distribution([row['maximum_absolute_error'] for row in comparisons])}


def summarize(rows, losses):
    if len(rows) != 12 or [row.get('capture_index') for row in rows] != list(range(12)):
        raise ValueError('Exactly twelve aligned original capture comparisons required')
    if any(row.get('first_module') != row.get('second_module') or not row.get('first_module') for row in rows):
        raise ValueError('Observed module order must agree between original forwards')
    if any(len(row.get('bands', [])) != 3 for row in rows):
        raise ValueError('Exactly three bands per original call required')
    if not losses or any(not isinstance(row.get('exact'), bool) or not isinstance(row.get('within_inherited_bounds'), bool)
                         or not math.isfinite(row.get('absolute_error', float('nan'))) for row in losses):
        raise ValueError('Complete finite loss comparisons required')
    inputs = family([row['input'] for row in rows])
    bands = family([band for row in rows for band in row['bands']])
    loss_summary = {'count': len(losses), 'exact_count': sum(row['exact'] for row in losses),
                    'nonexact_count': sum(not row['exact'] for row in losses),
                    'exceeds_inherited_bounds_count': sum(not row['within_inherited_bounds'] for row in losses),
                    'absolute_error': distribution([row['absolute_error'] for row in losses])}
    all_exact = inputs['exact_count'] == 12 and bands['exact_count'] == 36 and loss_summary['nonexact_count'] == 0
    return {'inputs': inputs, 'bands': bands, 'losses': loss_summary, 'all_observed_values_exact': all_exact,
            'status': 'completed_exact' if all_exact else 'completed_different',
            'candidate_accepted': False, 'training_authorized': False,
            'exact_historical_crash_replayed': False, 'projector_equivalence_established': False}
