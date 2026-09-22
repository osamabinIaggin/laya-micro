#!/usr/bin/env python3
"""Cut a Laya checkpoint's vocabulary down to the tokens a target domain uses.

The multilingual checkpoint (mmBERT-base) spends 196,608,000 of its 321,908,998
parameters on a 256,000 x 768 embedding table for 100+ languages. An
English-only robot touches a few thousand of those rows. Slicing the table and
rewriting the tokenizer to match leaves every transformer weight untouched, so
in-domain answers are bit-identical.

    python3 prune_vocab.py --out laya-multilingual-en --seed-top 4096

Writes a drop-in checkpoint directory: laya.Agent(<dir>) loads it unchanged.
"""

import argparse
import copy
import json
import os
import shutil
import string

REPO = "convaiinnovations/laya"


def corpus(cases_file):
    """Every string Zeus can put in front of the model, in its rendered form."""
    import bench as dp

    texts = []
    with open(cases_file) as f:
        cases = json.load(f)
    for c in cases:
        texts.append(c["utterance"])
        texts.append(json.dumps(dp.state_for(c["utterance"]), ensure_ascii=False))
        texts.append(c["vision"])
    questions = [dp.flat_questions(), dp.first_questions()]
    questions += [dp.second_questions(g) for g in dp.ACTION_GROUPS]
    for qs in questions:
        for q in qs.values():
            texts.append("%s question: %s" % (q["type"], q["instructions"]))
            crit = q.get("criteria")
            if q["type"] == "choice":
                texts += [" " + (k if not v else "%s: %s" % (k, v)) for k, v in crit.items()]
            else:
                texts += [" false: no, the statement does not hold", " true: yes, the statement holds"]
    # headroom: anything the recogniser can spell, plus small numbers
    texts += [" " + c for c in string.printable] + list(string.printable)
    texts += [" %d" % i for i in range(1001)] + [str(i) for i in range(1001)]
    return texts


def keep_set(tok_json, texts, seed_top):
    """Token ids to retain, closed over the BPE merges that build them.

    A kept token that some merge produces needs every parent of every such merge,
    or BPE can never assemble it and the word tokenizes differently than it did
    during training.
    """
    from tokenizers import Tokenizer

    d = json.load(open(tok_json))
    vocab = d["model"]["vocab"]
    inv = {i: t for t, i in vocab.items()}
    merges = [tuple(m) for m in d["model"]["merges"]]

    tok = Tokenizer.from_file(tok_json)
    ids = set()
    for t in texts:
        ids.update(tok.encode(t, add_special_tokens=False).ids)
    ids |= set(range(seed_top))                      # ids are frequency-ordered
    ids |= {t["id"] for t in d["added_tokens"]}
    # byte-fallback tokens make any UTF-8 input encodable without <unk>
    ids |= {i for t, i in vocab.items() if len(t) == 6 and t.startswith("<0x") and t.endswith(">")}

    child2parents = {}
    for a, b in merges:
        child2parents.setdefault(a + b, []).append((a, b))
    keep = {inv[i] for i in ids}
    stack = list(keep)
    while stack:
        for a, b in child2parents.get(stack.pop(), ()):
            for q in (a, b):
                if q not in keep:
                    keep.add(q)
                    stack.append(q)
    return d, vocab, keep


def write_tokenizer(d, vocab, keep, path):
    merges = [list(m) for m in d["model"]["merges"]
              if m[0] in keep and m[1] in keep and (m[0] + m[1]) in keep]
    order = sorted(keep, key=lambda t: vocab[t])     # keep ids monotone in the old ids
    new_id = {t: i for i, t in enumerate(order)}
    nd = copy.deepcopy(d)
    nd["model"]["vocab"] = new_id
    nd["model"]["merges"] = merges
    nd["added_tokens"] = [dict(t, id=new_id[t["content"]])
                          for t in d["added_tokens"] if t["content"] in new_id]
    pp = nd.get("post_processor") or {}
    for v in (pp.get("special_tokens") or {}).values():
        if "ids" in v:
            v["ids"] = [new_id[c] for c in v["tokens"]]
    json.dump(nd, open(path, "w"), ensure_ascii=False)
    return new_id, order


def verify(src_json, dst_json, new_id, vocab, texts):
    from tokenizers import Tokenizer

    old2new = {vocab[t]: i for t, i in new_id.items()}
    a, b = Tokenizer.from_file(src_json), Tokenizer.from_file(dst_json)
    bad = 0
    for s in texts:
        if [old2new.get(x, -1) for x in a.encode(s, add_special_tokens=False).ids] \
                != b.encode(s, add_special_tokens=False).ids:
            bad += 1
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subfolder", default="multilingual", help="multilingual | typed-decisions | '' for english")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed-top", type=int, default=4096,
                    help="also keep the N lowest token ids, which are the most frequent")
    ap.add_argument("--cases", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases", "zeus_cases.json"))
    args = ap.parse_args()

    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file, save_file

    root = snapshot_download(REPO, allow_patterns=[f"{args.subfolder}/*"] if args.subfolder else None)
    src = os.path.join(root, args.subfolder) if args.subfolder else root
    src_json = os.path.join(src, "tokenizer", "tokenizer.json")

    texts = corpus(args.cases)
    d, vocab, keep = keep_set(src_json, texts, args.seed_top)
    print("vocabulary %d -> %d (%.1f%%)" % (len(vocab), len(keep), 100 * len(keep) / len(vocab)))

    os.makedirs(os.path.join(args.out, "tokenizer"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "encoder"), exist_ok=True)
    dst_json = os.path.join(args.out, "tokenizer", "tokenizer.json")
    new_id, order = write_tokenizer(d, vocab, keep, dst_json)

    bad = verify(src_json, dst_json, new_id, vocab, texts)
    if bad:
        raise SystemExit("%d of %d strings tokenize differently after pruning" % (bad, len(texts)))
    print("tokenizer parity: %d/%d strings identical" % (len(texts), len(texts)))

    sd = load_file(os.path.join(src, "model.safetensors"))
    k = "encoder.embeddings.tok_embeddings.weight"
    idx = torch.tensor([vocab[t] for t in order], dtype=torch.long)
    print("%s %s -> %s" % (k, tuple(sd[k].shape), (len(order), sd[k].shape[1])))
    sd[k] = sd[k].index_select(0, idx).contiguous()
    save_file(sd, os.path.join(args.out, "model.safetensors"), metadata={"format": "pt"})

    cfg = json.load(open(os.path.join(src, "encoder", "config.json")))
    cfg["vocab_size"] = len(order)
    for field, token in (("pad_token_id", "<pad>"), ("cls_token_id", "<bos>"), ("bos_token_id", "<bos>"),
                         ("eos_token_id", "<eos>"), ("sep_token_id", "<eos>"), ("mask_token_id", "<mask>")):
        cfg[field] = new_id[token]
    json.dump(cfg, open(os.path.join(args.out, "encoder", "config.json"), "w"), indent=2)
    shutil.copy(os.path.join(src, "rl_agent_config.json"), os.path.join(args.out, "rl_agent_config.json"))
    shutil.copy(os.path.join(src, "tokenizer", "tokenizer_config.json"),
                os.path.join(args.out, "tokenizer", "tokenizer_config.json"))

    size = sum(os.path.getsize(os.path.join(p, f)) for p, _, fs in os.walk(args.out) for f in fs)
    print("wrote %s  (%.0f MB)" % (args.out, size / 1e6))


if __name__ == "__main__":
    main()
