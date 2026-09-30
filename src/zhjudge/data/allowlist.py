"""Training-data allowlist (configs/datasets.yaml), cross-checked against the licence audit (configs/licenses.yaml).

Every dataset the spike census looked at has an entry. Only entries with `enabled: true` AND a converter are
built. `enabled` follows the audit: true only for converter entries whose licence class is ALLOW or CONDITIONAL.
Each entry names its audit entry in `license:`; `zhjudge build` refuses an enabled source whose audit class is not
allowed by the release profile (`data.license_profile`, profiles defined in licenses.yaml), and the
`permissive_strict` profile additionally leaves out share-alike TRAINING sources (eval-only sets are unaffected).
"""
from __future__ import annotations

from pathlib import Path

import yaml


def load_allowlist(path: Path) -> dict:
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    ds = doc.get("datasets") or {}
    for k, v in ds.items():
        if not isinstance(v, dict) or not isinstance(v.get("enabled"), bool):
            raise SystemExit(f"{path}: datasets.{k}.enabled must be true or false")
    return ds


def load_licenses(path: Path) -> dict | None:
    """The licence audit (configs/licenses.yaml): {"profiles": {...}, "datasets": {...}}, or None if absent."""
    p = Path(path)
    if not p.exists():
        return None
    doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return {"profiles": doc.get("profiles") or {}, "datasets": doc.get("datasets") or {}}


def licence_entry(allowlist: dict, licenses: dict | None, key: str) -> dict | None:
    """Audit entry of allowlist entry `key` (via its `license:` field, else the same key)."""
    if not licenses:
        return None
    return licenses["datasets"].get((allowlist.get(key) or {}).get("license") or key)


def check_licence(cfg: dict, allowlist: dict, licenses: dict | None, converter: str) -> tuple[bool, str | None]:
    """(keep, problem). keep=False with problem=None means: dropped by the profile (not an error)."""
    if licenses is None:
        return True, None  # no audit file (e.g. a stripped-down checkout): the allowlist alone decides
    profile_name = cfg["data"].get("license_profile") or "default"
    profile = licenses["profiles"].get(profile_name)
    if profile is None:
        raise SystemExit(f"data.license_profile={profile_name!r} is not a profile in {cfg['data']['licenses']} "
                         f"(known: {sorted(licenses['profiles'])})")
    key = next((k for k, v in allowlist.items() if v.get("converter") == converter), converter)
    lic = licence_entry(allowlist, licenses, key)
    if lic is None:
        return False, f"{converter}: no entry in {cfg['data']['licenses']} (every trained source must be audited)"
    cls = lic.get("class")
    if cls not in profile.get("classes", []):
        return False, f"{converter}: licence class {cls} is not allowed by profile {profile_name!r}"
    role = (allowlist.get(key) or {}).get("role")
    if profile.get("share_alike") is False and lic.get("share_alike") and role != "eval_only":
        return False, None
    return True, None


def select_sources(cfg: dict, allowlist: dict, registry: dict, licenses: dict | None = None
                   ) -> tuple[list[str], list[str]]:
    """Return (selected converter names, notes). Refuses disabled sources unless the smoke override is on, and
    sources the licence audit does not allow under data.license_profile (same smoke override)."""
    want = cfg["data"]["sources"]
    by_conv = {v.get("converter"): k for k, v in allowlist.items() if v.get("converter")}
    notes = []
    smoke = bool(cfg["data"].get("allow_disabled") and cfg["run"].get("smoke"))
    if want == "enabled":
        chosen = [v["converter"] for v in allowlist.values() if v["enabled"] and v.get("converter")]
        for k, v in allowlist.items():
            if v["enabled"] and not v.get("converter"):
                notes.append(f"{k}: enabled but has no converter yet (ignored)")
    else:
        chosen = list(want)
    out = []
    for c in chosen:
        if c not in registry:
            raise SystemExit(f"unknown converter {c!r}; known: {sorted(registry)}")
        entry = allowlist.get(by_conv.get(c, c))
        if entry is None:
            raise SystemExit(f"{c}: not listed in the allowlist ({cfg['data']['allowlist']})")
        if not entry["enabled"]:
            if smoke:
                notes.append(f"{c}: NOT enabled in the allowlist; used only because this is a smoke test")
            else:
                raise SystemExit(f"{c}: enabled: false in {cfg['data']['allowlist']} (see DATA_LICENSES.md)")
        keep, problem = check_licence(cfg, allowlist, licenses, c)
        if problem:
            if smoke:
                notes.append(f"{problem}; used only because this is a smoke test")
            else:
                raise SystemExit(f"{problem}. Set enabled: false in {cfg['data']['allowlist']} (see DATA_LICENSES.md)")
        elif not keep:
            notes.append(f"{c}: share-alike source left out by licence profile "
                         f"{cfg['data'].get('license_profile')!r}")
            continue
        out.append(c)
    # stable order = registry order
    order = list(registry)
    return sorted(dict.fromkeys(out), key=order.index), notes


def is_enabled(allowlist: dict, converter: str) -> bool:
    for v in allowlist.values():
        if v.get("converter") == converter:
            return bool(v["enabled"])
    return False
