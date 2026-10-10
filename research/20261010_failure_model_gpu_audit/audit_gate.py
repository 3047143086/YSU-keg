"""Explicit conjunctions for the preregistered numerical audit."""


def archived_candidate_passed(row):
    forward = row.get('exact_forward_bands', [])
    if len(forward) != 3 or not all(forward) or not row.get('candidate_gradient_finite'):
        return False
    return not row.get('generic_gradient_finite') or bool(row.get('generic_comparison', {}).get('passed'))


def model_candidate_passed(result):
    if not (result.get('state_unchanged') and result.get('loss_matches')
            and result.get('candidate_all_gradients_finite')
            and result.get('participating_parameters_match')
            and result.get('participating_parameter_count', 0) > 0):
        return False
    rows = result.get('rows', [])
    if len(rows) != 12:
        return False
    for row in rows:
        bands = row.get('candidate_vs_original_bands', [])
        if not row.get('candidate_vs_original_input', {}).get('passed') or len(bands) != 3:
            return False
        if not all(band.get('passed') for band in bands):
            return False
    return (not result.get('original_all_gradients_finite')
            or bool(result.get('generic_gradient_equivalence_passed')))
