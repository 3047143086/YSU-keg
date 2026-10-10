import unittest
from method_observer import observed_methods


class MethodObserverTests(unittest.TestCase):
    def test_exception_restores_both_original_objects(self):
        class Owner: first=object(); second=object()
        first,second=Owner.first,Owner.second
        with self.assertRaises(RuntimeError):
            with observed_methods(((Owner,'first',object()),(Owner,'second',object()))):
                raise RuntimeError('failed forward')
        self.assertIs(Owner.first,first); self.assertIs(Owner.second,second)

    def test_normal_exit_restores_original_object(self):
        class Owner: method=object()
        original=Owner.method; replacement=object()
        with observed_methods(((Owner,'method',replacement),)): self.assertIs(Owner.method,replacement)
        self.assertIs(Owner.method,original)

    def test_partial_installation_failure_restores_prior_object(self):
        class Owner: method=object()
        original=Owner.method
        with self.assertRaises(AttributeError):
            with observed_methods(((Owner,'method',object()),(Owner,'missing',object()))): pass
        self.assertIs(Owner.method,original)


if __name__ == '__main__': unittest.main()
