"""From unified JSONL records to model inputs.

A *Laya record* is one decision in Laya's own question schema (what the runtime and the spike trainers consume):
  {"id", "source", "lang", "state", "question": {"type", "instructions", "criteria"?}, "target": [p per option],
   "gold": index of the gold option}
Tokenization uses upstream laya's own prompt builder (`laya.common.build_sequence`), i.e. the exact runtime format.
Batching and padding are the spike's (length-grouped, token-capped).
"""
from __future__ import annotations

import json
import pickle
import random
import time
from pathlib import Path

import numpy as np

QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {v: k for k, v in QTYPES.items()}


def onehot(i, k, smooth=0.0):
    t = [smooth / k] * k
    t[i] += 1.0 - smooth
    return t


def ordinal(i, k, neighbour=0.1):
    """Spike's ordinal soft target for score questions: most mass on the gold level, a little on each neighbour."""
    t = [0.0] * k
    t[i] = 1.0 - 2 * neighbour
    for nb in (i - 1, i + 1):
        if 0 <= nb < k:
            t[nb] = neighbour
    s = sum(t)
    return [v / s for v in t]


def to_laya(rec: dict, smooth=0.05, neighbour=0.1) -> dict:
    """Unified record -> Laya record. Options become list criteria (rendered as bare labels / level texts); a format
    variant (zhjudge.data.views.variant) adds `option_desc` ({option text: description}; noul: {"false", "true"}), turned
    into dict criteria in the record's current option order here, and `noul_labels` (Laya's custom noul labels)."""
    t = rec["type"]
    q = {"type": t, "instructions": rec["question"]}
    desc = rec.get("option_desc")
    if t == "choice":
        q["criteria"] = list(rec["options"])
        if desc:
            assert set(desc) == set(rec["options"]), (rec["id"], desc, rec["options"])
            q["criteria"] = {o: desc[o] for o in rec["options"]}
            assert list(q["criteria"]) == rec["options"], rec["id"]
        gold = rec["label"]
        target = onehot(gold, len(rec["options"]), smooth)
    elif t == "score":
        q["criteria"] = list(rec["options"])
        gold = rec["label"]
        target = ordinal(gold, len(rec["options"]), neighbour)
    else:  # noul: options are always [false, true]
        if desc:
            assert set(desc) <= {"false", "true"}, (rec["id"], desc)
            q["criteria"] = {k: desc[k] for k in ("false", "true") if k in desc}
        if rec.get("noul_labels"):
            q["labels"] = dict(rec["noul_labels"])
        gold = int(bool(rec["label"]))
        target = onehot(gold, 2, smooth)
    return {"id": rec["id"], "source": rec["source"], "lang": rec["lang"], "state": rec.get("context", ""),
            "question": q, "target": target, "gold": gold}


def augment(rec: dict, t: dict, seed) -> dict:
    """Training-time rendering of one unified TRAIN record (deterministic per record id and seed):
    - train.format_aug:    share of records with a view written Laya-style (JSON state with named fields, a
                           field-aware instruction; train.format_cross_lingual of them in the other language,
                           train.format_bare_labels of the NLI ones with bare labels); see zhjudge.data.views
    - train.format_variants: share of those also written in another Laya question shape (zhjudge.data.views.variant;
                           its own random stream, so the renderings and shuffles above do not move)
    - train.option_shuffle: share of `choice` records whose options are put in a new random order
    The gold option never changes, nor does the id. Evaluation and dev records are never augmented."""
    fa, osh = float(t.get("format_aug") or 0), float(t.get("option_shuffle") or 0)
    if not fa and not osh:
        return rec
    from .data import views

    rng = random.Random(f"{seed}:aug:{rec['id']}")
    if fa and rec.get("view") and rng.random() < fa:
        rec, lang = views.render(rec, rng, float(t.get("format_cross_lingual") or 0),
                                 float(t.get("format_bare_labels") or 0), with_lang=True)
        fv = float(t.get("format_variants") or 0)
        if fv:
            vrng = random.Random(f"{seed}:variant:{rec['id']}")
            if vrng.random() < fv:
                rec = views.variant(rec, lang, vrng)
    if osh and rec["type"] == "choice" and rng.random() < osh:
        rec = views.shuffle_options(rec, rng)
    return rec


def load_unified(unified_dir: Path, split: str, sources=None) -> list[dict]:
    """All records of one split, in stable (file name, line) order."""
    out = []
    for p in sorted((Path(unified_dir) / split).glob("*.jsonl")):
        if sources is not None and p.stem not in sources:
            continue
        with open(p, encoding="utf-8") as f:
            out.extend(json.loads(line) for line in f if line.strip())
    return out


def cap_per_source(recs, cap, seed, key="source"):
    """Deterministic per-source subsample (keeps file order of the kept records)."""
    if not cap:
        return recs
    by = {}
    for i, r in enumerate(recs):
        by.setdefault(r[key], []).append(i)
    keep = set()
    for s, idx in sorted(by.items()):
        if len(idx) > cap:
            idx = random.Random(f"{seed}:{s}").sample(idx, cap)
        keep.update(idx)
    return [r for i, r in enumerate(recs) if i in keep]


def cap_total(recs, cap, seed):
    if not cap or len(recs) <= cap:
        return recs
    keep = set(random.Random(f"{seed}:total").sample(range(len(recs)), cap))
    return [r for i, r in enumerate(recs) if i in keep]


def split_calibration(recs, frac, cap, minimum, seed, groups=None):
    """Hold out a calibration slice before training (upstream notebook fix #186; the spike used 10%/cap 400).
    groups: one key per record; records with the same key (e.g. the unified `group` of one oasst2 message tree) are
    held out together or not at all (the last group taken can make the slice a few records larger). With every key
    distinct (no `group` in the corpus) this is the per-record split."""
    keys = range(len(recs)) if groups is None else groups
    members = {}
    for i, k in enumerate(keys):
        members.setdefault(k, []).append(i)
    units = list(members.values())
    order = list(range(len(units)))
    random.Random(f"{seed}:calib").shuffle(order)
    n = min(cap, max(minimum, int(round(frac * len(recs)))), len(recs) // 2)
    cal = set()
    for u in order:
        if len(cal) >= n:
            break
        cal.update(units[u])
    return [r for i, r in enumerate(recs) if i not in cal], [recs[i] for i in sorted(cal)]


def encode_records(records, tok, max_len, head_max_len, log_every=20000):
    """Tokenize Laya records with upstream laya's own prompt builder. Drops a record only when its options do not
    all fit (markers beyond max_len), exactly like the spike. Items carry fmt: laya (dict state) | plain."""
    from laya.agent import Agent
    from laya.common import build_sequence, render_options

    items, dropped = [], 0
    t0 = time.time()
    for n, r in enumerate(records):
        qdef = r["question"]
        Agent._check_question(r["id"], qdef)
        q = Agent._to_internal(qdef)
        ids, markers = build_sequence(tok, r["state"], q, max_len, head_max_len,
                                      truncate_left=isinstance(r["state"], list))
        k = len(render_options(q))
        if len(markers) != k or len(r["target"]) != k:
            dropped += 1
            continue
        items.append({"id": r["id"], "source": r["source"], "lang": r["lang"], "ids": np.asarray(ids, dtype=np.int32),
                      "markers": list(markers), "qtype": QTYPES[q["t"]], "target": [float(x) for x in r["target"]],
                      "gold": int(r["gold"]), "fmt": "laya" if isinstance(r["state"], dict) else "plain"})
        if log_every and (n + 1) % log_every == 0:
            print(f"  tokenized {n + 1}/{len(records)} ({time.time() - t0:.0f}s)", flush=True)
    return items, dropped


def option_fit(items, records, tok, head_max_len) -> dict:
    """How laya.common.build_sequence fitted each question into its head (report only: the inputs are not changed).
    Per source: options whose token span was cut (48 tokens per option, and once the options overflow head_max_len
    each keeps [MASK] + max(4, (head_max_len - 16) // K) - 1 tokens), options whose kept span equals another option's
    (the model cannot tell them apart), records whose instruction (incl. the '<type> question:' prefix) was cut to
    max(8, what the options leave), and records dropped because not every option fitted in max_len. Spans are read
    from the encoded items; only a span that may have been cut is re-tokenized in full."""
    from laya.agent import Agent
    from laya.common import render_options

    mask = tok.mask_token
    full_opt, full_ins = {}, {}

    def n_opt(o):
        if o not in full_opt:
            full_opt[o] = 1 + len(tok(" " + o.replace(mask, " "), add_special_tokens=False)["input_ids"])
        return full_opt[o]

    by_id = {it["id"]: it for it in items}
    out = {}
    for r in records:
        s = out.setdefault(r["source"], {"records": 0, "dropped": 0, "max_options": 0, "options": 0, "options_cut": 0,
                                         "records_options_cut": 0, "options_colliding": 0, "records_colliding": 0,
                                         "records_instruction_cut": 0, "instruction_kept_min": None, "examples": []})
        s["records"] += 1
        it = by_id.get(r["id"])
        if it is None:
            s["dropped"] += 1
            continue
        q = Agent._to_internal(r["question"])
        opts = render_options(q)
        ids, mk = np.asarray(it["ids"]), list(it["markers"])
        sep = np.flatnonzero(ids[mk[-1]:] == tok.sep_token_id)
        ends = mk[1:] + [mk[-1] + int(sep[0]) if len(sep) else len(ids)]
        spans = [tuple(ids[a:b].tolist()) for a, b in zip(mk, ends)]
        per = max(4, (head_max_len - 16) // max(1, len(spans)))
        cut = [len(sp) in (49, per) and len(sp) < n_opt(o) for sp, o in zip(spans, opts)]
        seen = {}
        for sp in spans:
            seen[sp] = seen.get(sp, 0) + 1
        coll = [o for sp, o in zip(spans, opts) if seen[sp] > 1]
        kept = mk[0] - 2  # [CLS] instruction [SEP] precede the first option marker
        ins_cut = False
        if kept == max(8, head_max_len - sum(map(len, spans))):
            key = (q["t"], str(q["ins"]))
            if key not in full_ins:
                full_ins[key] = len(tok("%s question: %s" % (q["t"], key[1].replace(mask, " ")),
                                        add_special_tokens=False)["input_ids"])
            ins_cut = kept < full_ins[key]
        s["max_options"] = max(s["max_options"], len(spans))
        s["options"] += len(spans)
        s["options_cut"] += sum(cut)
        s["records_options_cut"] += any(cut)
        s["options_colliding"] += len(coll)
        s["records_colliding"] += bool(coll)
        s["records_instruction_cut"] += ins_cut
        if ins_cut:
            s["instruction_kept_min"] = min(kept, s["instruction_kept_min"] or kept)
        for o in coll:
            if o[:60] not in s["examples"] and len(s["examples"]) < 3:
                s["examples"].append(o[:60])
    return {k: out[k] for k in sorted(out)}


def option_fit_lines(fit: dict, head_max_len) -> list[str]:
    """Short summary of option_fit(): one line per source where anything was cut, collided or dropped."""
    L = []
    for src, s in fit.items():
        if not (s["options_cut"] or s["options_colliding"] or s["records_instruction_cut"] or s["dropped"]):
            continue
        L.append(f"  {src}: {s['records_options_cut']}/{s['records']} records with cut options ({s['options_cut']} of "
                 f"{s['options']} options, up to {s['max_options']} per question), {s['records_colliding']} with "
                 f"options rendered identically ({s['options_colliding']} options, e.g. {s['examples'][:2]}), "
                 f"{s['records_instruction_cut']} with the instruction cut"
                 + (f" (to {s['instruction_kept_min']} tokens)" if s["records_instruction_cut"] else "")
                 + (f", {s['dropped']} dropped (options beyond max_len)" if s["dropped"] else ""))
    return L or [f"  every option and instruction fits in head_max_len {head_max_len}"]


def cached_items(cache_path: Path, records, tok, max_len, head_max_len, key: str):
    """Tokenize once per run (pickle cache keyed by record ids + tokenizer settings)."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            blob = pickle.load(f)
        if blob.get("key") == key:
            return blob["items"]
    t0 = time.time()
    items, dropped = encode_records(records, tok, max_len, head_max_len)
    print(f"tokenized {len(items)}/{len(records)} records in {time.time() - t0:.1f}s (dropped {dropped} whose "
          f"options do not fit)", flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump({"key": key, "items": items}, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(cache_path)
    return items


def length_grouped_batches(items, batch_size, seed, pad_multiple=32, pool=50, max_tokens=0):
    """Shuffle, then sort inside pools of `pool` batches by length to cut padding waste (spike).

    max_tokens > 0 caps padded tokens per batch (count x padded max length), like upstream's
    `max_tokens_per_batch`: long sequences get smaller batches so peak memory stays bounded."""
    rng = random.Random(seed)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    batches = []
    span = batch_size * pool

    def padded(n):
        return ((n + pad_multiple - 1) // pad_multiple) * pad_multiple

    for s in range(0, len(idx), span):
        chunk = sorted(idx[s:s + span], key=lambda i: len(items[i]["ids"]))
        if not max_tokens:
            batches += [chunk[j:j + batch_size] for j in range(0, len(chunk), batch_size)]
            continue
        cur = []
        for i in chunk:  # ascending length, so the newest item sets the padded length
            L = padded(len(items[i]["ids"]))
            if cur and (len(cur) + 1 > batch_size or (len(cur) + 1) * L > max_tokens):
                batches.append(cur)
                cur = []
            cur.append(i)
        if cur:
            batches.append(cur)
    rng.shuffle(batches)
    return batches


def eval_batches(items, batch_size, max_tokens=0, pad_multiple=32):
    """Deterministic length-sorted batches for inference (returns lists of indices into items)."""
    order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))
    batches, cur = [], []
    for i in order:
        L = ((len(items[i]["ids"]) + pad_multiple - 1) // pad_multiple) * pad_multiple
        if cur and (len(cur) + 1 > batch_size or (max_tokens and (len(cur) + 1) * L > max_tokens)):
            batches.append(cur)
            cur = []
        cur.append(i)
    if cur:
        batches.append(cur)
    return batches


def collate_np(items, pad_id, pad_multiple=32):
    n = len(items)
    L = max(len(it["ids"]) for it in items)
    L = ((L + pad_multiple - 1) // pad_multiple) * pad_multiple
    K = max(2, max(len(it["markers"]) for it in items))
    b = {
        "input_ids": np.full((n, L), pad_id, dtype=np.int64),
        "attention_mask": np.zeros((n, L), dtype=np.int64),
        "marker_pos": np.zeros((n, K), dtype=np.int64),
        "marker_mask": np.zeros((n, K), dtype=bool),
        "qtype": np.array([it["qtype"] for it in items], dtype=np.int64),
        "target": np.zeros((n, K), dtype=np.float32),
    }
    for i, it in enumerate(items):
        l, k = len(it["ids"]), len(it["markers"])
        b["input_ids"][i, :l] = it["ids"]
        b["attention_mask"][i, :l] = 1
        b["marker_pos"][i, :k] = it["markers"]
        b["marker_mask"][i, :k] = True
        b["target"][i, :k] = it["target"]
    b["real_tokens"] = int(b["attention_mask"].sum())
    b["padded_tokens"] = n * L
    return b


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
