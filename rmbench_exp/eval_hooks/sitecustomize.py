"""Stage oracle hook for the RMBench evaluation client (stage-memory experiment).

Active ONLY when LILA_STAGE_ORACLE=1 (set by scripts/step24_eval_with_oracle.sh, which also
puts this folder on PYTHONPATH). Nothing in RMBench or XPolicyLab is edited; instead, when
their modules are imported, two methods are wrapped:

  envs._base_task.Base_Task.get_obs     remembers the task object of the current episode
  <XPolicyLab model client>.call         adds obs["oracle"] = {"stages_done": n} to every
                                         update_obs message (n from rmbench_lila/oracle.py)

The oracle only READS simulator state; the task's own success check is untouched.
In any other process (or without the variable) this file does nothing.
"""
import importlib.abc
import importlib.util
import os
import sys

if os.environ.get("LILA_STAGE_ORACLE") == "1":
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _spec = importlib.util.spec_from_file_location(
        "lila_stage_oracle", os.path.join(_HERE, "..", "rmbench_lila", "oracle.py"))
    _oracle_mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_oracle_mod)
    _state = {"env": None, "key": None, "oracle": None}

    def _oracle_value():
        env = _state["env"]
        if env is None:
            return None
        task = type(env).__name__
        key = (id(env), id(getattr(env, "block1", None)))   # new actors -> new episode
        if _state["key"] != key:
            _state["key"], _state["oracle"] = key, _oracle_mod.make_oracle(task, env)
        return int(_state["oracle"]())

    def _patch_base_task(mod):
        orig = mod.Base_Task.get_obs

        def get_obs(self, *a, **k):
            _state["env"] = self
            return orig(self, *a, **k)
        mod.Base_Task.get_obs = get_obs

    def _patch_client(mod):
        for name in dir(mod):
            cls = getattr(mod, name)
            if isinstance(cls, type) and "call" in cls.__dict__ and cls.__module__ == mod.__name__:
                orig = cls.call

                def call(self, func_name=None, obs=None, *a, _orig=orig, **k):
                    if func_name == "update_obs" and isinstance(obs, dict):
                        n = _oracle_value()
                        if n is not None:
                            obs = dict(obs, oracle={"stages_done": n})
                    return _orig(self, func_name, obs, *a, **k)
                cls.call = call

    _TARGETS = {"envs._base_task": _patch_base_task,
                "client_server.ws.model_client": _patch_client,
                "client_server.tcp.model_client": _patch_client}

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname not in _TARGETS:
                return None
            for finder in sys.meta_path:
                if finder is self:
                    continue
                spec = getattr(finder, "find_spec", lambda *x: None)(fullname, path, target)
                if spec is not None and spec.loader is not None:
                    loader, patch = spec.loader, _TARGETS[fullname]
                    orig_exec = loader.exec_module

                    def exec_module(module, _orig=orig_exec, _patch=patch):
                        _orig(module)
                        _patch(module)
                        print(f"[stage-oracle] patched {module.__name__}", flush=True)
                    loader.exec_module = exec_module
                    return spec
            return None

    sys.meta_path.insert(0, _Finder())
