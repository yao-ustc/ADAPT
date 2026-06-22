"""Lightweight PyTorch forward-hook manager.

Allows registering callbacks that fire at named hook points (e.g. 'after_softmax')
within nn.Module forward passes.
"""

from typing import Dict, Callable, List
from collections import defaultdict


class HookManager:
    """Manage named hooks on a module.

    Usage:
        module.hook_manager = HookManager()
        module.hook_manager.register('after_softmax', my_callback)

    The module's forward should call:
        ret = self.hook_manager('after_softmax', ret=ret)
    """

    def __init__(self, hook_dict: Dict[str, List[Callable]] = None):
        self.hook_dict = hook_dict or defaultdict(list)
        self.called = defaultdict(int)
        self.forks: Dict[str, "HookManager"] = {}

    def register(self, name: str, func: Callable):
        assert name, "Hook name must be non-empty"
        self.hook_dict[name].append(func)

    def unregister(self, name: str, func: Callable):
        assert name
        if func in self.hook_dict[name]:
            self.hook_dict[name].remove(func)

    def __call__(self, name: str, **kwargs):
        if name in self.hook_dict:
            self.called[name] += 1
            ret = kwargs.get("ret")
            for fn in self.hook_dict[name]:
                ret = fn(ret)
            return ret
        return kwargs.get("ret")

    def unregister_all(self):
        self.hook_dict.clear()
        self.called.clear()

    def fork(self, name: str) -> "HookManager":
        if name in self.forks:
            raise ValueError(f"Already forked with '{name}'.")
        prefix = name + "."
        filtered = {k[len(prefix):]: v for k, v in self.hook_dict.items()
                    if k.startswith(prefix)}
        new = HookManager(defaultdict(list, filtered))
        self.forks[name] = new
        return new

    def finalize(self):
        for name in self.hook_dict:
            if self.called[name] == 0:
                raise ValueError(f"Hook '{name}' was registered but never called!")


def init_hookmanager(module: "torch.nn.Module"):
    """Attach a HookManager to a module if not already present."""
    if not hasattr(module, "hook_manager"):
        module.hook_manager = HookManager()
