"""Independent, single-evaluation-pass cache for BRIDGE encoder outputs."""
from contextlib import contextmanager
from functools import update_wrapper
from types import FunctionType, MethodType
from threading import Lock, get_ident
from weakref import WeakKeyDictionary

_ACTIVE = WeakKeyDictionary()
_ACTIVE_LOCK = Lock()


def _torch():
    import torch
    return torch


def _rng_snapshot():
    torch = _torch()
    result = {"cpu": torch.get_rng_state().clone()}
    if torch.cuda.is_available():
        result["cuda"] = tuple(state.clone() for state in torch.cuda.get_rng_state_all())
    else:
        result["cuda"] = ()
    return result


def _same_rng(left, right):
    torch = _torch()
    if not torch.equal(left["cpu"], right["cpu"]):
        return False
    if len(left["cuda"]) != len(right["cuda"]):
        return False
    return all(torch.equal(a, b) for a, b in zip(left["cuda"], right["cuda"]))


def _state_token(model):
    """Cheap mutation token; PyTorch parameter/buffer versions change on loads."""
    token = []
    for name, value in list(model.named_parameters()) + list(model.named_buffers()):
        token.append((name, id(value), int(getattr(value, "_version", -1)), tuple(value.shape),
                      str(value.dtype), str(value.device)))
    generated = {"id_rep", "id_preference", "v_rep", "v_preference", "t_rep", "t_preference"}
    for module_name, module in model.named_modules():
        for name, value in vars(module).items():
            if name in generated:
                continue
            key = module_name + "." + name
            if _torch().is_tensor(value):
                token.append((key, id(value), int(getattr(value, "_version", -1)),
                              tuple(value.shape), str(value.dtype), str(value.device)))
            elif isinstance(value, (bool, int, float, str, type(None))):
                token.append((key, value))
    return tuple(token)


def _output_token(outputs):
    """Reject mutation of the forward result between score batches."""
    torch = _torch()
    values = []
    def visit(value):
        if torch.is_tensor(value):
            values.append((id(value), int(value._version), tuple(value.shape),
                           str(value.dtype), str(value.device)))
        elif isinstance(value, (tuple, list)):
            for child in value:
                visit(child)
        else:
            raise EvalCacheInvalidated("forward outputs contain unsupported values")
    visit(outputs)
    return tuple(values)


class EvalCacheInvalidated(RuntimeError):
    """Raised when a pass is reused after model state or context changed."""


class EvalForwardCache:
    """Cache one encoder forward result for exactly one eval/no-grad pass."""

    def __init__(self, model):
        self.model = model
        self.generation = 0
        self.hits = 0
        self.misses = 0
        self._outputs = None
        self._token = None
        self._rng = None
        self._entered = False
        self._thread = None
        self._sealed = False
        self._output_versions = None

    def __enter__(self):
        torch = _torch()
        if self.model.training:
            raise EvalCacheInvalidated("cache requires model.eval()")
        if torch.is_grad_enabled():
            raise EvalCacheInvalidated("cache requires torch.no_grad()")
        if self._entered or self._sealed:
            raise EvalCacheInvalidated("a cache object belongs to exactly one pass")
        with _ACTIVE_LOCK:
            if self.model in _ACTIVE:
                raise EvalCacheInvalidated("another cache pass already owns this model")
            _ACTIVE[self.model] = self
        self._token = _state_token(self.model)
        self._rng = _rng_snapshot()
        self._entered = True
        self._thread = get_ident()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.invalidate("pass_exit")
        self._entered = False
        self._sealed = True
        with _ACTIVE_LOCK:
            if _ACTIVE.get(self.model) is self:
                del _ACTIVE[self.model]
        return False

    def invalidate(self, reason="manual"):
        self._outputs = None
        self._token = None
        self._rng = None
        self._output_versions = None
        self.generation += 1
        self.invalidation_reason = reason

    def _check(self, model):
        torch = _torch()
        if not self._entered or model is not self.model or get_ident() != self._thread:
            raise EvalCacheInvalidated("cache belongs to another or closed pass")
        if model.training or torch.is_grad_enabled():
            raise EvalCacheInvalidated("eval/no-grad context was lost")
        if _state_token(model) != self._token:
            self.invalidate("model_state_changed")
            raise EvalCacheInvalidated("model parameters or buffers changed")
        if self._outputs is not None and _output_token(self._outputs) != self._output_versions:
            self.invalidate("forward_output_changed")
            raise EvalCacheInvalidated("cached encoder outputs changed")

    def forward(self, model, encoder_class):
        self._check(model)
        before = _rng_snapshot()
        if self._outputs is None:
            self.misses += 1
            outputs = encoder_class.forward(model)
            if not isinstance(outputs, tuple) or len(outputs) != 5:
                raise EvalCacheInvalidated("encoder output contract is not five-tuple")
            after = _rng_snapshot()
            if not _same_rng(before, after):
                self.invalidate("forward_changed_rng")
                raise EvalCacheInvalidated("encoder forward changed RNG state")
            self._outputs = outputs
            self._output_versions = _output_token(outputs)
            self._token = _state_token(model)
            self._rng = after
        else:
            self.hits += 1
            if not _same_rng(before, self._rng):
                self.invalidate("hit_changed_rng")
                raise EvalCacheInvalidated("cached hit observed changed RNG state")
        after = _rng_snapshot()
        if not _same_rng(before, after):
            self.invalidate("cache_changed_rng")
            raise EvalCacheInvalidated("cache access changed RNG state")
        return self._outputs

    def receipt(self):
        return {"generation": self.generation, "forward_generations": self.misses,
                "cache_hits": self.hits, "cache_misses": self.misses,
                "valid_only_inside_eval_no_grad": True,
                "cache_scope": "one_model_one_eval_pass",
                "mode_guard": "current_mode_only; intermediate_mode_transitions_prohibited"}


def bind_cached_full_sort_predict(model, original_predict, encoder_class, cache):
    """Bind original predictor code with only its encoder global redirected."""
    if original_predict.__code__.co_freevars:
        raise TypeError("A closure-bearing predictor cannot be safely rebound")

    class _ForwardShim:
        @staticmethod
        def forward(instance):
            return cache.forward(instance, encoder_class)

    globals_copy = dict(original_predict.__globals__)
    symbol = getattr(encoder_class, "__name__", "DualFrequencyEncoder")
    if globals_copy.get(symbol) is not encoder_class:
        raise TypeError("original predictor does not resolve the supplied encoder class")
    globals_copy[symbol] = _ForwardShim
    rebound = FunctionType(original_predict.__code__, globals_copy,
                           original_predict.__name__, original_predict.__defaults__,
                           original_predict.__closure__)
    update_wrapper(rebound, original_predict)
    return MethodType(rebound, model)


@contextmanager
def cached_eval_pass(model, original_predict, encoder_class):
    """Yield a predictor with isolated cache state, without patching model/class."""
    with EvalForwardCache(model) as cache:
        yield bind_cached_full_sort_predict(model, original_predict, encoder_class, cache), cache


def masked_topk(scores, positives, k=50):
    """Apply the existing positive-item mask and return deterministic Top-k IDs."""
    torch = _torch()
    masked = scores.clone()
    for row, items in enumerate(positives):
        if len(items):
            masked[row, torch.as_tensor(items, device=masked.device, dtype=torch.long)] = -torch.inf
    return torch.topk(masked, k=min(k, masked.shape[1]), dim=1, sorted=True).indices
