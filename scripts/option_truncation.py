"""How laya.common.build_sequence fits many-option questions into the question head (report only: changes nothing).

  uv run python scripts/option_truncation.py TOKENIZER [--head-max-len 256] [--corpus --config configs/train_base.yaml]
  uv run python scripts/option_truncation.py --lower-bound       # no tokenizer needed: 1 token per whitespace word

TOKENIZER: a tokenizer directory, e.g. an export's tokenizer/ (outputs/model/<name>/tokenizer from a Kaggle run: no
GPU, no model needed) or runs/<run>/tokenizer. build_sequence keeps at most 48 tokens per option; once the options
overflow head_max_len it keeps [MASK] + max(4, (head_max_len - 16) // K) - 1 tokens of each and cuts the instruction
(with its '<type> question:' prefix) to max(8, what is left). Options cut to the same tokens cannot be told apart.
--lower-bound counts one token per whitespace word: with a tokenizer whose tokens never span a space, every cut and
every collision found that way also happens with it (it can only find more).

Always: the label sets with the most options (banking77's 77 intents, MASSIVE's 60 intents in both languages, MInDS-14,
DBpedia-14, GoEmotions), each with the plain and a Laya-style instruction. --corpus: also zhjudge.records.option_fit on
the records `zhjudge eval` scores (the capped test sample, and its test:laya renderings), per source; `zhjudge train`
(train_stats.json) and `zhjudge eval` (eval.json meta.option_fit) record the same counts for their own items.
"""
from __future__ import annotations

import argparse
import sys

from zhjudge.data.views import DBPEDIA_ZH, GOEMO_ZH
from zhjudge.data.zh import MASSIVE_EN, MASSIVE_ZH, MINDS_ZH
from zhjudge.records import encode_records, option_fit, option_fit_lines

# the ClassLabel names of legacy-datasets/banking77, '_' -> ' ' as the converter writes them (zhjudge.data.en)
BANKING77 = [n.replace("_", " ") for n in """activate_my_card age_limit apple_pay_or_google_pay atm_support
automatic_top_up balance_not_updated_after_bank_transfer balance_not_updated_after_cheque_or_cash_deposit
beneficiary_not_allowed
cancel_transfer card_about_to_expire card_acceptance card_arrival card_delivery_estimate card_linking card_not_working
card_payment_fee_charged card_payment_not_recognised card_payment_wrong_exchange_rate card_swallowed
cash_withdrawal_charge cash_withdrawal_not_recognised change_pin compromised_card contactless_not_working
country_support declined_card_payment declined_cash_withdrawal declined_transfer direct_debit_payment_not_recognised
disposable_card_limits edit_personal_details exchange_charge exchange_rate exchange_via_app extra_charge_on_statement
failed_transfer fiat_currency_support get_disposable_virtual_card get_physical_card getting_spare_card
getting_virtual_card lost_or_stolen_card lost_or_stolen_phone order_physical_card passcode_forgotten
pending_card_payment pending_cash_withdrawal pending_top_up pending_transfer pin_blocked receiving_money
Refund_not_showing_up request_refund reverted_card_payment? supported_cards_and_currencies terminate_account
top_up_by_bank_transfer_charge top_up_by_card_charge top_up_by_cash_or_cheque top_up_failed top_up_limits
top_up_reverted topping_up_by_card transaction_charged_twice transfer_fee_charged transfer_into_account
transfer_not_received_by_recipient transfer_timing unable_to_verify_identity verify_my_identity verify_source_of_funds
verify_top_up virtual_card_not_working visa_or_mastercard why_verify_identity wrong_amount_of_cash_received
wrong_exchange_rate_for_cash_withdrawal""".split()]
# (name, options, plain instruction as the converter writes it, a Laya-style one from zhjudge.data.views)
CASES = [
    ("banking77_en", BANKING77, "Which banking intent best describes this customer message?",
     "Which banking intent best describes `customer_message`?"),
    ("massive_en intents", [MASSIVE_EN[k] for k in sorted(MASSIVE_EN)], "What does the user want the assistant to do?",
     "What does the user want the assistant to do in `user_message`?"),
    ("massive_zh intents", [MASSIVE_ZH[k] for k in sorted(MASSIVE_ZH)], "用户这句话的意图是什么？",
     "`用户消息`的意图是什么？"),
    ("minds14_zh", list(MINDS_ZH), "这位银行客户来电的意图是什么？", None),  # eval-only: no Laya-style rendering
    ("dbpedia14_en", list(DBPEDIA_ZH), "What kind of entity is this encyclopedia entry about?",
     "What kind of entity do `title` and `content` describe?"),
    ("goemotions_en", list(GOEMO_ZH), "Which emotion does this comment express most clearly?",
     "Which emotion does `comment` express most clearly?"),
]


class WordTok:
    """Lower bound on token counts: one token per whitespace word (an unspaced CJK label counts as one word)."""
    mask_token, mask_token_id, cls_token_id, sep_token_id = "[MASK]", 1, 2, 3

    def __init__(self):
        self.vocab = {}

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [self.vocab.setdefault(w, len(self.vocab) + 10) for w in text.split()]}


def case_lines(tok, head_max_len, max_len):
    L = []
    for name, opts, *instructions in CASES:
        for how, ins in zip(("plain", "Laya-style"), instructions):
            if ins is None:
                continue
            rec = {"id": name, "source": name, "lang": "en", "state": "x", "gold": 0, "target": [0.0] * len(opts),
                   "question": {"type": "choice", "instructions": ins, "criteria": opts}}
            items, _ = encode_records([rec], tok, max_len, head_max_len, log_every=0)
            f = option_fit(items, [rec], tok, head_max_len)[name]
            full = len(tok(f"choice question: {ins}", add_special_tokens=False)["input_ids"])
            kept = items[0]["markers"][0] - 2 if items else 0
            L.append(f"{name} ({how}): {len(opts)} options, {f['options_cut']} cut, {f['options_colliding']} rendered "
                     f"identically {f['examples']}, instruction kept {kept} of {full} tokens"
                     + (" (DROPPED: options beyond max_len)" if f["dropped"] else ""))
    return L


def corpus_lines(tok, head_max_len, max_len, config, sets):
    from zhjudge.config import load_config, paths
    from zhjudge.eval import eval_records, eval_seed, laya_records

    cfg = load_config(config, sets)
    P = paths(cfg)
    L = []
    plain = [r for r in eval_records(cfg, P, eval_seed(cfg)) if r["_slices"][0] == "test:ALL"]
    laya = laya_records({**cfg, "eval": {**cfg["eval"], "laya_test": cfg["eval"].get("laya_test") or 500}}, P,
                        eval_seed(cfg))
    for name, recs in (("test:* (plain)", plain), ("test:laya", laya)):
        items, _ = encode_records(recs, tok, max_len, head_max_len, log_every=0)
        fit = option_fit(items, recs, tok, head_max_len)
        L += [f"{name}: {len(recs)} records"] + option_fit_lines(fit, head_max_len)
    return L


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tokenizer", nargs="?", help="tokenizer directory (or --lower-bound)")
    ap.add_argument("--lower-bound", action="store_true", help="1 token per whitespace word instead of a tokenizer")
    ap.add_argument("--head-max-len", type=int, default=256, help="model.head_max_len (train_base.yaml: 256)")
    ap.add_argument("--max-len", type=int, default=1024, help="model.max_len (train_base.yaml: 1024)")
    ap.add_argument("--corpus", action="store_true", help="also the test records `zhjudge eval` scores (data/unified)")
    ap.add_argument("--config", default="configs/train_base.yaml")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    a = ap.parse_args(argv)
    if not a.tokenizer and not a.lower_bound:
        ap.error("give a tokenizer directory or --lower-bound")
    if a.lower_bound:
        tok = WordTok()
    else:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(a.tokenizer)
    print(f"head_max_len {a.head_max_len}, max_len {a.max_len}, "
          f"{'lower bound: 1 token per word' if a.lower_bound else a.tokenizer}")
    print("\n".join(case_lines(tok, a.head_max_len, a.max_len)))
    if a.corpus:
        print("\n".join(corpus_lines(tok, a.head_max_len, a.max_len, a.config, a.set)))


if __name__ == "__main__":
    sys.exit(main())
