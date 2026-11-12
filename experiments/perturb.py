"""Perturbation operators: removal, alteration, injection, on requirements, datasheet passages, criteria.

All operators work on a *state* = {"inputs": ..., "requirements": [...]} (see agents.chain.state_of)
and return a new state, or None when the operator is not applicable to the element.
"""
from __future__ import annotations

import copy
import re

KINDS = ("removal", "alteration", "injection")
ALTER = 0.67     # 30 min -> 20 min
INJECT = 1.5

NUM_RE = re.compile(r"(?<![A-Za-z0-9.])([−-]?)(\d{1,3}(?:[  ]\d{3})+|\d+(?:\.\d+)?)")


def _fmt(x: float, template: str) -> str:
    if " " in template or " " in template:
        return f"{round(x):,}".replace(",", " ")
    if "." in template:
        return f"{x:.{len(template.split('.')[1])}f}"
    return str(round(x))


def scale_text(text: str, f: float) -> str | None:
    if not NUM_RE.search(text):
        return None

    def sub(m):
        sign, num = m.group(1), m.group(2)
        val = float(num.replace(" ", "").replace(" ", "")) * f
        return sign + _fmt(val, num)
    return NUM_RE.sub(sub, text)


def scale_value(v, f):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        x = v * f
        return round(x) if isinstance(v, int) else round(x, 2)
    if isinstance(v, list) and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v):
        return [scale_value(x, f) for x in v]
    return v


def scale_facts(facts: dict, f: float) -> dict:
    return {k: scale_value(v, f) for k, v in facts.items()}


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_range(v):
    return isinstance(v, list) and len(v) == 2 and all(_is_num(x) for x in v)


def _find_passage(state, pid):
    for cid, ds in state["inputs"]["datasheets"].items():
        for i, p in enumerate(ds["passages"]):
            if p["id"] == pid:
                return cid, i
    return None, None


def element_kind(node: str) -> str:
    return node.split("_", 1)[0]          # req / ds / crit / spec


def element_key(node: str) -> str:
    return node.split("_", 1)[1]


# ---------------------------------------------------------------- requirements
def _req(state, scen, rid, kind):
    reqs = state["requirements"]
    idx = next((i for i, r in enumerate(reqs) if r["id"] == rid), None)
    if idx is None:
        return None
    r = reqs[idx]
    gt = scen.gt_req(rid) or {}
    if kind == "removal":
        del reqs[idx]
        return f"removed {rid}"
    if kind == "alteration":
        v = r["value"]
        if _is_num(v):
            new = scale_value(v, ALTER)
        elif "alt_value" in gt:
            new = gt["alt_value"]
        elif _is_range(v):
            new = scale_value(v, ALTER)
        else:
            return None
        if new == v:
            return None
        r["value"] = new
        return f"{rid}.value {v} -> {new}"
    if kind == "injection":
        v, op = r["value"], r["operator"]
        if _is_num(v):
            new = scale_value(v, 0.5 if op in ("<=", "<") else INJECT)
        elif _is_range(v):
            w = (v[1] - v[0]) * 0.25
            new = [round(v[0] - w, 1), round(v[1] + w, 1)]
        elif "alt_value" in gt:
            new = gt["alt_value"]
        else:
            return None
        fake = dict(r, id=f"{rid}X", value=new, critical=True)
        reqs.insert(idx + 1, fake)
        return f"injected {rid}X: {r['attribute']} {op} {new} (critical)"
    raise ValueError(kind)


# ---------------------------------------------------------------- datasheet passages
def _passage(state, pid, kind):
    cid, i = _find_passage(state, pid)
    if cid is None:
        return None
    ps = state["inputs"]["datasheets"][cid]["passages"]
    p = ps[i]
    if kind == "removal":
        del ps[i]
        return f"removed {pid}"
    if kind == "alteration":
        if p.get("alt_text"):
            p["text"], p["facts"] = p["alt_text"], copy.deepcopy(p["alt_facts"])
        else:
            t = scale_text(p["text"], ALTER)
            if t is None:
                return None
            p["text"], p["facts"] = t, scale_facts(p.get("facts", {}), ALTER)
        return f"{pid} -> {p['text']!r}"
    if kind == "injection":
        if p.get("alt_text"):
            text, facts = p["alt_text"], copy.deepcopy(p["alt_facts"])
        else:
            text = scale_text(p["text"], INJECT)
            if text is None:
                return None
            facts = scale_facts(p.get("facts", {}), INJECT)
        ps.insert(i + 1, {"id": f"{pid}x", "text": "Revision note: " + text, "facts": facts})
        return f"injected {pid}x: {text!r}"
    raise ValueError(kind)


# ---------------------------------------------------------------- criteria
def _criterion(state, kid, kind):
    ks = state["inputs"]["criteria"]
    idx = next((i for i, k in enumerate(ks) if k["id"] == kid), None)
    if idx is None:
        return None
    k = ks[idx]
    if kind == "removal":
        del ks[idx]
        return f"removed {kid}"
    if kind == "alteration":
        if not k.get("alt"):
            return None
        k.update({x: k["alt"][x] for x in ("text", "attribute", "direction")})
        return f"{kid} -> {k['text']}"
    if kind == "injection":
        if not k.get("inject"):
            return None
        ks.insert(idx, copy.deepcopy(k["inject"]))
        return f"injected {k['inject']['id']} before {kid}: {k['inject']['text']}"
    raise ValueError(kind)


def apply(state: dict, scen, node: str, kind: str):
    """Return (new_state, description) or None if not applicable."""
    s = copy.deepcopy(state)
    t, key = element_kind(node), element_key(node)
    if t == "req":
        d = _req(s, scen, key, kind)
    elif t == "ds":
        d = _passage(s, key, kind)
    elif t == "crit":
        d = _criterion(s, key, kind)
    elif t == "spec" and kind == "removal":
        before = len(s["inputs"]["spec_passages"])
        s["inputs"]["spec_passages"] = [p for p in s["inputs"]["spec_passages"] if p["id"] != key]
        d = f"removed {key}" if len(s["inputs"]["spec_passages"]) < before else None
    else:
        d = None
    return None if d is None else (s, d)


def apply_group_removal(state: dict, scen, nodes: list[str]):
    s, descs = copy.deepcopy(state), []
    for n in nodes:
        r = apply(s, scen, n, "removal")
        if r is not None:
            s, d = r
            descs.append(d)
    return s, "; ".join(descs)


def apply_error(state: dict, err: dict):
    """Exp. 3: inject a known error with ground truth."""
    s = copy.deepcopy(state)
    if err["target"] == "requirement":
        r = next((r for r in s["requirements"] if r["id"] == err["element"]), None)
        if r is None:
            return None
        r[err["field"]] = err["new_value"]
        return s
    cid, i = _find_passage(s, err["element"])
    if cid is None:
        return None
    p = s["inputs"]["datasheets"][cid]["passages"][i]
    p["text"], p["facts"] = err["new_text"], copy.deepcopy(err["new_facts"])
    return s
