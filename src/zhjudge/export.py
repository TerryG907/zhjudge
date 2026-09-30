"""Package the calibrated checkpoint as a release-ready Laya checkpoint + model card.

  uv run zhjudge export --config ...
  HF_TOKEN=... uv run zhjudge export --config ... --push-to <user>/<repo>     # optional, private repo by default

Writes runs/<name>/export/<export.name>/:
  model.safetensors, rl_agent_config.json, encoder/, tokenizer/   (the Laya checkpoint; `zhjudge compat` checks it in
                                                                   laya, laya's /v1/systemone server and, on Apple
                                                                   silicon, laya-mlx)
  README.md            model card (Chinese + English): data, per-source licences and attribution duties (from
                       configs/licenses.yaml), metrics vs laya-multilingual, usage, limits
  DATA_LICENSES.md     the repository's data-licence audit (attribution / share-alike duties), if present
  training_data.json   sources, per-file sha256, allowlist notes, split sources and data.train_sample of the corpus it
                       was trained on
  eval.json            full evaluation (if `zhjudge eval` ran)
The token for --push-to is read from the HF_TOKEN environment variable only and never written to disk.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from .baselines import HEADLINE, LAYA_MULTILINGUAL
from .config import add_common_args, load_config, paths
from .data.allowlist import licence_entry, load_allowlist, load_licenses
from .stats import fmt_d

REQUIRED = ["model.safetensors", "rl_agent_config.json", "encoder/config.json", "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json"]


def structural_check(d: Path) -> list[str]:
    problems = [f"missing {f}" for f in REQUIRED if not (d / f).exists()]
    if problems:
        return problems
    cfg = json.loads((d / "rl_agent_config.json").read_text())
    for k in ("encoder", "head_layers", "max_len", "head_max_len", "act_costs", "temperature", "temperature_by_options"):
        if k not in cfg:
            problems.append(f"rl_agent_config.json lacks {k}")
    if not (isinstance(cfg.get("temperature"), list) and len(cfg["temperature"]) == 3):
        problems.append("temperature must be a list of 3 floats")
    from safetensors import safe_open

    with safe_open(str(d / "model.safetensors"), "np") as f:
        keys = list(f.keys())
        dtypes = {f.get_slice(k).get_dtype() for k in keys}
    for prefix in ("encoder.", "type_emb.", "scorer.", "act_head.", "head."):
        if not any(k.startswith(prefix) for k in keys):
            problems.append(f"no {prefix}* tensors")
    if "temperature" not in keys:
        problems.append("no temperature buffer")
    if not dtypes <= {"F16", "F32"}:
        problems.append(f"unexpected dtypes {sorted(dtypes)}")
    return problems


def _cell(m, k="acc"):
    return "-" if not m or m.get(k) is None else f"{m[k]:.3f}"


LICENCE_URLS = {
    "apache-2.0": "https://www.apache.org/licenses/LICENSE-2.0",
    "mit": "https://opensource.org/license/mit",
    "cc0-1.0": "https://creativecommons.org/publicdomain/zero/1.0/",
    "cc-by-3.0": "https://creativecommons.org/licenses/by/3.0/",
    "cc-by-4.0": "https://creativecommons.org/licenses/by/4.0/",
    "cc-by-sa-3.0": "https://creativecommons.org/licenses/by-sa/3.0/",
    "cc-by-sa-4.0": "https://creativecommons.org/licenses/by-sa/4.0/",
}

BASE_LICENCES = {"jhu-clsp/mmBERT-base": "MIT", "jhu-clsp/mmBERT-small": "MIT",
                 "answerdotai/ModernBERT-large": "Apache-2.0"}

DISCLAIMER = ("Independent open-source project. Not affiliated with, endorsed by or sponsored by TypeSafe (Jev) or "
              "Convai (Laya); those names only identify the model category (Jev-style typed decisions) and the Laya "
              "checkpoint format. Jev was never run or measured and no compatibility with Jev's API is claimed. / "
              "独立开源项目，与 TypeSafe（Jev）和 Convai（Laya）没有任何隶属、合作、赞助或背书关系；提及其名称只为说明模型类别"
              "（Jev 式类型化决策）和 Laya 检查点格式。本项目从未运行或测量过 Jev，也不声称与 Jev 的接口兼容。")


def attribution_lines(cfg: dict, P: dict, trained, manifest) -> list[str]:
    """Per-source licence + attribution / disclosure duties, read from the licence audit (configs/licenses.yaml),
    so the duties of every CONDITIONAL (and ALLOW) source travel with the weights in the card itself."""
    licenses = load_licenses(P["licenses"])
    if not licenses:
        return ["## Data licences and attribution / 数据许可与署名", "",
                "See DATA_LICENSES.md (copied next to this card).", ""]
    allow = load_allowlist(P["allowlist"])
    by_conv = {v.get("converter"): k for k, v in allow.items() if v.get("converter")}
    built = (manifest or {}).get("sources") or {}
    eval_only = [s for s, m in built.items() if m.get("eval_only")]

    def row(src):
        key = by_conv.get(src, src)
        a = allow.get(key) or {}
        lic = licence_entry(allow, licenses, key) or {}
        lic_name = str(lic.get("license") or "see DATA_LICENSES.md")
        url = LICENCE_URLS.get(lic_name.lower())
        lic_txt = f"[{lic_name}]({url})" if url else lic_name
        ev = (lic.get("evidence") or [None])[0]
        dataset = f"{a.get('name') or src} (`{a.get('hf_id') or lic.get('hf_id') or '-'}`)"
        notice = lic.get("card_notice")
        # the audit's duties, minus the two that this card discharges elsewhere (share-alike paragraph, notices)
        duties = "; ".join(str(d) for d in lic.get("duties") or []
                           if "share-alike question" not in str(d)
                           and not (notice and str(d).startswith("disclose in model card"))) or "-"
        cls = lic.get("class", "?") + (", share-alike" if lic.get("share_alike") else "")
        return (f"| {src} | {dataset} | {lic_txt} | {cls} | {duties}"
                f"{' **Notice:** ' + notice if notice else ''}{f' ([source]({ev}))' if ev else ''} |")

    L = ["## Data licences and attribution / 数据许可与署名", "",
         "Every training source below was converted into Laya typed-decision records (reformatted, filtered, "
         "de-duplicated, decontaminated and re-split): **the data were modified**. Licence classes and duties come "
         "from the repository's licence audit (`configs/licenses.yaml`, DATA_LICENSES.md, copied next to this card). "
         "/ 下列训练来源都已转换为 Laya 类型化决策格式（改写、过滤、去重、去污染、重新划分），即**已修改**。",
         "", f"### Trained on / 训练数据 ({len(trained or [])})", "",
         "| source | dataset | licence | class | attribution and duties |", "|---|---|---|---|---|"]
    L += [row(s) for s in trained or []]
    sa = [s for s in trained or [] if (licence_entry(allow, licenses, by_conv.get(s, s)) or {}).get("share_alike")]
    if sa:
        wl = str(cfg["export"].get("license") or "other")
        L += ["", f"Share-alike sources in the training data: {', '.join(sa)}. Whether trained weights are "
              "\"Adapted Material\" of these datasets is unsettled; "
              + ("this release follows the conservative reading and licenses the weights under CC BY-SA 4.0 "
                 "(DATA_LICENSES.md section 6)." if wl.lower().startswith("cc-by-sa") else
                 f"**the weights licence ({wl}) does not follow the audit's conservative reading (CC BY-SA 4.0); "
                 "see DATA_LICENSES.md section 6 before releasing.**")]
    if eval_only:
        L += ["", f"### Evaluated on, never trained on / 仅评测 ({len(eval_only)})", "",
              "| source | dataset | licence | class | attribution and duties |", "|---|---|---|---|---|"]
        L += [row(s) for s in eval_only]
    if cfg["eval"].get("probes") or cfg["data"].get("probes"):
        L += ["", "Baseline probe sets (CLUE, LCQMC, ChnSentiCorp, Amazon reviews, XNLI, AG News, GLUE, DREAM, "
              "LocalLLaMA/typed-decisions) were downloaded at run time and used only to compute the probe scores "
              "and to decontaminate the training split; several are non-commercial or unlicensed and were "
              "never trained on or redistributed."]
    base = cfg["model"]["base"]
    base_lic = BASE_LICENCES.get(base, "see the base model's card")
    L += ["", f"Base encoder: `{base}` ({base_lic}; its licence notice applies to the encoder weights). Decision "
          "head and checkpoint format: upstream Laya (`laya` on PyPI, Apache-2.0).", ""]
    return L


def status_lines(ev) -> list[str]:
    """Where the model stands against the re-measured baseline, from eval.json's paired differences (the repository's
    registered rule: better / worse only when the paired 95% interval excludes 0)."""
    pairs = (ev or {}).get("paired_vs_baseline") or {}
    if not pairs:
        return []
    rows = list(HEADLINE)
    rows += [("test:ALL", "held-out test, all"), ("test:group:eval-only", "never-trained eval sources")]
    L = ["## Status / 状态", "",
         "**Preview, not a final release.** One training run: the 95% intervals below cover which items were drawn, "
         "not training-seed variance. The probe sets guided the design of this version's training data and format "
         "(no probe item was trained on), so probe gains are not a clean held-out measure; the never-trained eval "
         "sources are the least targeted yardstick. / **预览版，不是正式版。** 只训练了一次，区间只反映抽题的随机性；"
         "训练数据和写法是参考探针结果设计的（探针题目本身没有进训练），探针上的提升不能全算作泛化。", "",
         "| set | this model | laya-multilingual (re-measured) | Δ acc, paired 95% interval | verdict |",
         "|---|---|---|---|---|"]
    for key, label in rows:
        p = pairs.get(key)
        if not p or p.get("d_acc_ci95") is None:
            continue
        lo, hi = p["d_acc_ci95"]
        verdict = "better / 更好" if lo > 0 else "worse / 更差" if hi < 0 else "no clear difference / 分不出差别"
        L.append(f"| {label} | {p['acc_a'] * 100:.1f} | {p['acc_b'] * 100:.1f} | {fmt_d(p)} | {verdict} |")
    return L + [""]


def model_card(cfg, acfg, ev, manifest, name, P=None):
    smoke = bool(cfg["run"].get("smoke"))
    tr = acfg.get("training", {})
    srcs, sample = tr.get("sources", []), (manifest or {}).get("train_sample")
    lines = ["---", f"license: {cfg['export'].get('license') or 'other'}", "language:", "- zh", "- en",
             f"base_model: {acfg['encoder']}", "library_name: laya", "tags:", "- typed-decisions", "- laya",
             "- calibration", "- chinese", "---", ""]
    if smoke:
        lines += ["> **SMOKE TEST — NOT FOR RELEASE.** Trained for a handful of steps on a tiny sample to test the "
                  "pipeline; the numbers below are meaningless. / 冒烟测试产物，仅用于验证流程，指标没有意义。", ""]
    src_url = cfg["export"].get("source_url")
    code = [f"Code, training data pipeline and licence audit / 代码、数据流程与许可审核: <{src_url}>", ""] if src_url else []
    lines += [f"# {name}", "", f"> {DISCLAIMER}", ""] + code + ([] if smoke else status_lines(ev)) + [
              "中文优先的类型化决策模型（System One）：给定状态（文本 / JSON）和一个带选项的问题，一次前向计算输出每个选项的"
              "校准概率（choice / score / noul），不生成文本。检查点使用 Laya 的格式（决策头就是上游 `laya` 的模块），"
              "`zhjudge compat` 检查它能被 `laya` 参考运行时加载（装了 `serve` 附加依赖时还检查 Laya 的 `/v1/systemone` 服务，在 Apple 芯片上还检查 `laya-mlx`）。", "",
              "A Chinese-first typed-decision model: given a state and a typed question, one forward pass returns "
              "calibrated probabilities over the options. Laya checkpoint format (the head is upstream `laya`'s own "
              "module); `zhjudge compat` checks that it loads in the `laya` reference runtime (and in Laya's "
              "`/v1/systemone` server with the `serve` extra installed, and `laya-mlx` on Apple silicon).", "",
              "## Training", "",
              f"- Base encoder: `{acfg['encoder']}` @ `{tr.get('base_revision', '?')}`",
              f"- Updates: {tr.get('updates')}, epochs: {tr.get('epochs_completed')}, hours: {tr.get('hours')}, "
              f"device: {tr.get('device')}, precision: {tr.get('precision')}",
              "- Recipe: RLCD (policy gradient on a strictly proper score) + soft cross-entropy, as in the upstream "
              "Laya notebook; temperatures fitted on a held-out calibration slice.",
              f"- Temperatures: {acfg.get('temperature')} (by options: {acfg.get('temperature_by_options') or '{}'})",
              f"- Training sources ({len(srcs)}): {', '.join(srcs) if srcs else '-'}",
              f"- Corpus sha256: `{(manifest or {}).get('corpus_sha256', '?')}` (per-file hashes in training_data.json)"
              + (f"; train records changed by data.train_sample `{json.dumps(sample)}` (per source: `split_sources` "
                 "in training_data.json)" if sample else ""),
              "- Licences and attribution duties of every source: the section below and DATA_LICENSES.md (copied "
              "next to this card). / 各数据集的许可与署名义务见下文及随附的 DATA_LICENSES.md。",
              "- **No outputs of Jev (or any other proprietary decision API) were used for training or labels.** "
              "/ 训练数据与标签均不含 Jev 等闭源决策接口的输出。", ""]
    if P is not None:
        lines += attribution_lines(cfg, P, srcs, manifest)
    if ev:
        m = ev.get("model", {})
        meas = ev.get("baseline_measured") or {}
        lines += ["## Evaluation (accuracy)", "",
                  "| set | this model | laya-multilingual (recorded) | laya-multilingual (re-measured) |",
                  "|---|---|---|---|"]
        for key, label in HEADLINE:
            if key in m:
                lines.append(f"| {label} | {_cell(m.get(key))} | {_cell(LAYA_MULTILINGUAL.get(key))} | "
                             f"{_cell(meas.get(key))} |")
        for key in ("test:ALL", "test:lang:zh", "test:lang:en"):
            if key in m:
                lines.append(f"| held-out {key[5:]} (n={m[key]['n']}) | {_cell(m[key])} | - | {_cell(meas.get(key))} |")
        lines += ["", "Brier, ECE-15, NLL and per-task slices: eval.json.", ""]
    lines += ["## Usage", "", "```python", "import laya", f'agent = laya.load("path/to/{name}")',
              'agent.predict("客户说：三月的账单被重复扣款了，今天不退款我们就取消订阅。", {',
              '    "churn": {"type": "noul", "instructions": "客户是否威胁要取消？"},',
              '    "dept": {"type": "choice", "instructions": "应由哪个部门处理？",',
              '             "criteria": {"billing": "账单、付款、退款", "technical": "故障", "sales": "新合同"}},',
              "})", "```", "",
              "## Limitations", "",
              "- Work in progress. Probabilities are calibrated on the training distribution; re-check calibration on "
              "your own data before gating decisions on confidence.",
              "- Chinese and English only were targeted; other languages inherit whatever mmBERT gives.", ""]
    return "\n".join(lines)


def share_alike_sources(cfg: dict, P: dict, sources) -> list[str]:
    """Trained sources whose licence audit entry is share-alike (their terms may attach to the weights)."""
    licenses = load_licenses(P["licenses"])
    if not licenses or not sources:
        return []
    allow = load_allowlist(P["allowlist"])
    by_conv = {v.get("converter"): k for k, v in allow.items() if v.get("converter")}
    return [s for s in sources if (licence_entry(allow, licenses, by_conv.get(s, s)) or {}).get("share_alike")]


def export(cfg: dict, push_to=None, private=True) -> Path:
    P = paths(cfg)
    src = P["model"]
    if not (src / "model.safetensors").exists():
        raise SystemExit(f"{src} has no checkpoint: run `zhjudge train` (and `zhjudge calibrate`) first")
    acfg = json.loads((src / "rl_agent_config.json").read_text(encoding="utf-8"))
    if acfg.get("temperature") == [1.0, 1.0, 1.0] and not acfg.get("training", {}).get("calibration"):
        print("[export] warning: checkpoint is not calibrated (run `zhjudge calibrate`)", flush=True)
    name = cfg["export"].get("name") or cfg["model"]["model_name"]
    weights_lic = str(cfg["export"].get("license") or "other")
    sa = share_alike_sources(cfg, P, acfg.get("training", {}).get("sources"))
    lic_problem = None
    if sa and not weights_lic.lower().startswith("cc-by-sa"):
        lic_problem = (f"export.license={weights_lic} but the model was trained on share-alike sources {sa}; "
                       f"use cc-by-sa-4.0, or retrain with data.license_profile=permissive_strict (DATA_LICENSES.md "
                       f"section 6)")
        print(f"[export] WARNING: {lic_problem}", flush=True)
    dst = P["export"] / name
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for f in ("model.safetensors", "rl_agent_config.json"):
        shutil.copy2(src / f, dst / f)
    for d in ("encoder", "tokenizer"):
        shutil.copytree(src / d, dst / d)
    ev_path = P["run"] / "eval.json"
    ev = json.loads(ev_path.read_text(encoding="utf-8")) if ev_path.exists() else None
    if ev:
        shutil.copy2(ev_path, dst / "eval.json")
    man_path = P["unified"] / "manifest.json"
    manifest = json.loads(man_path.read_text(encoding="utf-8")) if man_path.exists() else None
    if manifest:
        (dst / "training_data.json").write_text(json.dumps({
            "sources": acfg.get("training", {}).get("sources"), "corpus_sha256": manifest.get("corpus_sha256"),
            "sha256": manifest.get("sha256"), "allowlist": manifest.get("allowlist"),
            "decontamination": manifest.get("decontamination"), "revisions": manifest.get("revisions"),
            "split_sources": manifest.get("split_sources"), "train_sample": manifest.get("train_sample") or {}},
            indent=1, ensure_ascii=False), encoding="utf-8")
    lic = P["root"] / "DATA_LICENSES.md"
    if lic.exists():  # attribution / share-alike duties travel with the weights
        shutil.copy2(lic, dst / "DATA_LICENSES.md")
    (dst / "README.md").write_text(model_card(cfg, acfg, ev, manifest, name, P), encoding="utf-8")
    problems = structural_check(dst)
    if problems:
        raise SystemExit(f"export {dst} failed the structural check: {problems}")
    size = sum(p.stat().st_size for p in dst.rglob("*") if p.is_file()) / 2**20
    print(f"[export] {dst} ({size:.0f} MB) passes the structural check", flush=True)
    if push_to:
        if cfg["run"].get("smoke"):
            raise SystemExit("refusing to push a smoke-test model")
        if lic_problem:
            raise SystemExit(f"refusing to push: {lic_problem}")
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise SystemExit("set HF_TOKEN in the environment to push (it is never written to disk)")
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(push_to, repo_type="model", private=private, exist_ok=True)
        api.upload_folder(folder_path=str(dst), repo_id=push_to, repo_type="model",
                          commit_message=f"upload {name}")
        print(f"[export] pushed to https://huggingface.co/{push_to} ({'private' if private else 'public'})", flush=True)
    return dst


def main(argv=None):
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--push-to", default=None, help="Hugging Face repo id to upload to (token from $HF_TOKEN)")
    ap.add_argument("--public", action="store_true", help="create the Hugging Face repo as public (default private)")
    a = ap.parse_args(argv)
    export(load_config(a.config, a.set), a.push_to, private=not a.public)


if __name__ == "__main__":
    main()
