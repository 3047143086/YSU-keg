"""Report warnings without changing solvers; reject reported SVD fallback."""
from contextlib import contextmanager
import warnings


def reported_svd_fallback(rows):
    for row in rows:
        message = row.get('message', '').lower()
        if ('svd' in message or 'gesvd' in message or 'cusolver' in message) and any(
                term in message for term in ('converg', 'fallback', 'fall back', 'recomput', 'failed')):
            return True
    return False


@contextmanager
def observed_warnings(rows, phase):
    original = warnings.showwarning

    def observe(message, category, filename, lineno, file=None, line=None):
        rows.append({'phase': phase['name'], 'message': str(message), 'category': category.__name__,
                     'filename': filename, 'lineno': lineno})
        return original(message, category, filename, lineno, file=file, line=line)

    try:
        warnings.showwarning = observe
        yield
    finally:
        warnings.showwarning = original
