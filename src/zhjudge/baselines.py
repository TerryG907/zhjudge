"""Recorded baseline: convaiinnovations/laya @ 5e7b2b1 `multilingual` (laya-multilingual, mmBERT-base) as measured by
the phase-0 spike on the probe sets that `zhjudge probes` rebuilds (official `laya` 0.3.7 PyTorch runtime, Apple M5 Pro
MPS, fp32). Same metric definitions as zhjudge.metrics. These are MEASURED numbers from the spike, not targets.

`zhjudge eval` prints them next to our model's numbers, and (with eval.baseline: true) also re-measures the baseline
in the current environment on the same records, including our unified test split.
"""

SOURCE = "phase-0 spike, baselines/results/summary.json, run torch-multilingual-mps (2026-09-24)"

HEADLINE = [
    ("probe:zh/zh-prompt", "zh probe, Chinese prompt (600)"),
    ("probe:zh/en-prompt", "zh probe, English prompt (600)"),
    ("probe:en-control", "EN control, same task families (360)"),
    ("probe:xnli/zh/zh-prompt", "XNLI parallel, zh (150)"),
    ("probe:xnli/en", "XNLI parallel, en (150)"),
    ("probe:typed-decisions", "typed-decisions test (2,000 decisions)"),
]

LAYA_MULTILINGUAL = {
    'probe:en-control': {'n': 360, 'acc': 0.6806, 'brier': 0.4904, 'ece15': 0.1898, 'nll': 1.3809, 'score_mae': 1.5306},
    'probe:en-control/family:nli': {'n': 60, 'acc': 0.8167, 'brier': 0.2566, 'ece15': 0.137, 'nll': 0.4746},
    'probe:en-control/family:paraphrase': {'n': 60, 'acc': 0.8333, 'brier': 0.2713, 'ece15': 0.1617, 'nll': 0.4254},
    'probe:en-control/family:rating': {'n': 60, 'acc': 0.2333, 'brier': 1.3404, 'ece15': 0.6351, 'nll': 5.4478, 'score_mae': 1.5306},
    'probe:en-control/family:reading_mc': {'n': 60, 'acc': 0.4, 'brier': 0.7702, 'ece15': 0.275, 'nll': 1.411},
    'probe:en-control/family:sentiment': {'n': 60, 'acc': 0.8833, 'brier': 0.1785, 'ece15': 0.0778, 'nll': 0.2921},
    'probe:en-control/family:topic': {'n': 60, 'acc': 0.9167, 'brier': 0.1257, 'ece15': 0.0562, 'nll': 0.2345},
    'probe:en-control/qtype:choice': {'n': 240, 'acc': 0.7542, 'brier': 0.3327, 'ece15': 0.1205, 'nll': 0.603},
    'probe:en-control/qtype:noul': {'n': 60, 'acc': 0.8333, 'brier': 0.2713, 'ece15': 0.1617, 'nll': 0.4254},
    'probe:en-control/qtype:score': {'n': 60, 'acc': 0.2333, 'brier': 1.3404, 'ece15': 0.6351, 'nll': 5.4478, 'score_mae': 1.5306},
    'probe:typed-decisions': {'n': 2000, 'acc': 0.352, 'brier': 0.8904, 'ece15': 0.3139, 'nll': 1.8269, 'soft_brier': 0.5354, 'score_mae': 0.7604},
    'probe:typed-decisions/family:agent_trace_observability': {'n': 500, 'acc': 0.286, 'brier': 0.9596, 'ece15': 0.3612, 'nll': 1.9694, 'soft_brier': 0.4577, 'score_mae': 0.965},
    'probe:typed-decisions/family:customer_service': {'n': 500, 'acc': 0.42, 'brier': 0.7843, 'ece15': 0.27, 'nll': 1.7732, 'soft_brier': 0.4696, 'score_mae': 0.6287},
    'probe:typed-decisions/family:invoice_processing': {'n': 500, 'acc': 0.308, 'brier': 1.0477, 'ece15': 0.3958, 'nll': 2.0899, 'soft_brier': 0.8975, 'score_mae': 0.7849},
    'probe:typed-decisions/family:security_incidents': {'n': 500, 'acc': 0.394, 'brier': 0.77, 'ece15': 0.2365, 'nll': 1.4754, 'soft_brier': 0.3166, 'score_mae': 0.6629},
    'probe:typed-decisions/qtype:choice': {'n': 600, 'acc': 0.295, 'brier': 0.9885, 'ece15': 0.3623, 'nll': 2.3945, 'soft_brier': 0.4994},
    'probe:typed-decisions/qtype:noul': {'n': 600, 'acc': 0.4967, 'brier': 0.8233, 'ece15': 0.3757, 'nll': 1.4206, 'soft_brier': 0.5714},
    'probe:typed-decisions/qtype:score': {'n': 800, 'acc': 0.2863, 'brier': 0.8672, 'ece15': 0.2313, 'nll': 1.706, 'score_mae': 0.7604},
    'probe:xnli/en': {'n': 150, 'acc': 0.8467, 'brier': 0.2544, 'ece15': 0.1163, 'nll': 0.4939},
    'probe:xnli/zh/en-prompt': {'n': 150, 'acc': 0.7533, 'brier': 0.4304, 'ece15': 0.1706, 'nll': 1.008},
    'probe:xnli/zh/en-prompt/family:nli': {'n': 150, 'acc': 0.7533, 'brier': 0.4304, 'ece15': 0.1706, 'nll': 1.008},
    'probe:xnli/zh/en-prompt/qtype:choice': {'n': 150, 'acc': 0.7533, 'brier': 0.4304, 'ece15': 0.1706, 'nll': 1.008},
    'probe:xnli/zh/zh-prompt': {'n': 150, 'acc': 0.7, 'brier': 0.4641, 'ece15': 0.166, 'nll': 0.8822},
    'probe:zh/en-prompt': {'n': 600, 'acc': 0.5017, 'brier': 0.7915, 'ece15': 0.3339, 'nll': 2.4269, 'score_mae': 1.5819},
    'probe:zh/en-prompt/family:nli': {'n': 90, 'acc': 0.6444, 'brier': 0.5986, 'ece15': 0.2744, 'nll': 1.2693},
    'probe:zh/en-prompt/family:paraphrase': {'n': 100, 'acc': 0.61, 'brier': 0.5814, 'ece15': 0.2854, 'nll': 1.0672},
    'probe:zh/en-prompt/family:rating': {'n': 90, 'acc': 0.2667, 'brier': 1.2868, 'ece15': 0.5873, 'nll': 5.6058, 'score_mae': 1.5819},
    'probe:zh/en-prompt/family:reading_mc': {'n': 100, 'acc': 0.35, 'brier': 0.895, 'ece15': 0.3185, 'nll': 1.8083},
    'probe:zh/en-prompt/family:sentiment': {'n': 70, 'acc': 0.8571, 'brier': 0.2382, 'ece15': 0.1171, 'nll': 0.4735},
    'probe:zh/en-prompt/family:topic': {'n': 150, 'acc': 0.42, 'brier': 0.9393, 'ece15': 0.3956, 'nll': 3.4445},
    'probe:zh/en-prompt/qtype:choice': {'n': 410, 'acc': 0.5268, 'brier': 0.734, 'ece15': 0.2971, 'nll': 2.0607},
    'probe:zh/en-prompt/qtype:noul': {'n': 100, 'acc': 0.61, 'brier': 0.5814, 'ece15': 0.2854, 'nll': 1.0672},
    'probe:zh/en-prompt/qtype:score': {'n': 90, 'acc': 0.2667, 'brier': 1.2868, 'ece15': 0.5873, 'nll': 5.6058, 'score_mae': 1.5819},
    'probe:zh/zh-prompt': {'n': 600, 'acc': 0.4733, 'brier': 0.8131, 'ece15': 0.3405, 'nll': 2.0339, 'score_mae': 1.4027},
    'probe:zh/zh-prompt/family:nli': {'n': 90, 'acc': 0.6, 'brier': 0.6162, 'ece15': 0.2429, 'nll': 1.1669},
    'probe:zh/zh-prompt/family:paraphrase': {'n': 100, 'acc': 0.53, 'brier': 0.8232, 'ece15': 0.4244, 'nll': 1.8876},
    'probe:zh/zh-prompt/family:rating': {'n': 90, 'acc': 0.2778, 'brier': 1.1783, 'ece15': 0.5347, 'nll': 3.2109, 'score_mae': 1.4027},
    'probe:zh/zh-prompt/family:reading_mc': {'n': 100, 'acc': 0.34, 'brier': 0.905, 'ece15': 0.3396, 'nll': 1.8777},
    'probe:zh/zh-prompt/family:sentiment': {'n': 70, 'acc': 0.7143, 'brier': 0.397, 'ece15': 0.1335, 'nll': 0.643},
    'probe:zh/zh-prompt/family:topic': {'n': 150, 'acc': 0.4533, 'brier': 0.8381, 'ece15': 0.3501, 'nll': 2.6984},
    'probe:zh/zh-prompt/qtype:choice': {'n': 410, 'acc': 0.5024, 'brier': 0.7304, 'ece15': 0.2839, 'nll': 1.8111},
    'probe:zh/zh-prompt/qtype:noul': {'n': 100, 'acc': 0.53, 'brier': 0.8232, 'ece15': 0.4244, 'nll': 1.8876},
    'probe:zh/zh-prompt/qtype:score': {'n': 90, 'acc': 0.2778, 'brier': 1.1783, 'ece15': 0.5347, 'nll': 3.2109, 'score_mae': 1.4027},
}
