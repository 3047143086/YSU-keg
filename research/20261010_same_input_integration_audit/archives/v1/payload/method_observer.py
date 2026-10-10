"""Exception-safe temporary observation with exact method restoration."""
from contextlib import contextmanager


@contextmanager
def observed_methods(replacements):
    originals = []
    try:
        for owner, name, replacement in replacements:
            original = getattr(owner, name)
            originals.append((owner, name, original))
            setattr(owner, name, replacement)
        yield
    finally:
        for owner, name, original in reversed(originals):
            setattr(owner, name, original)
