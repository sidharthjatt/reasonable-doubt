---
license: cc-by-4.0
base_model: microsoft/deberta-v3-base
datasets:
  - coastalcph/lex_glue
language:
  - en
pipeline_tag: text-classification
library_name: onnx
tags:
  - legal
  - contracts
  - ledgar
  - lexglue
  - onnx
  - deberta-v3
---

# reasonable-doubt-deberta-ledgar

DeBERTa-v3-base fine-tuned on **LEDGAR** (LexGLUE), exported to **ONNX FP32**, for
single-label classification of contract provisions into **100 classes**.

This is the artefact actually served by the
[reasonable-doubt](https://huggingface.co/spaces/sidharthjatt/reasonable-doubt) Space —
the same bytes, not a re-export.

## Numbers

| | |
|---|---|
| **macro-F1, `test_3000`** | **0.8091** |
| accuracy, `test_3000` | 0.8733 |
| measured on | ONNX FP32, arm64 (Apple Silicon), onnxruntime 1.29.0 |
| classes | 100 (99 predicted on `test_3000`) |
| seeds | **1** — see Limitations |

**Macro-F1 is the metric to read, not accuracy.** LEDGAR's class distribution is heavily
long-tailed, so accuracy flatters any model that gets the head classes right.

`test_3000` is a fixed 3,000-row sample of the LexGLUE LEDGAR **test** split, which is
**chronological** (train 2016–2017, dev 2018, test 2019) — not a random split. Numbers
here describe 2019 filings.

## Files

| file | sha256 |
|---|---|
| `model.onnx` | `e70fb095ee8ddd82c9449d6da0006b4fa1c7a92bb8547d5538f869c445010200` |
| `config.json` | `0b84b3be80278cd6e622ebeb0302fe186997225a605b98b6709b48cd4cb4f64e` |
| `tokenizer.json` | `133b1071a455bb96ec0d7aa1242e616cc42ec29652fded524fb6883832afa903` |
| `spm.model` | `c679fbf93643d19aab7ee10c0b99e460bdbc02fedf34b92b05af343b4af586fd` |
| `tokenizer_config.json` | `f3fe8bea48e9f8d6884a74d7aecafb3ae9ed869bd5f6a8561d63cc5f45d15a73` |
| `special_tokens_map.json` | `b2f1b2f15f29a6b6d9d6ea4eca1675d2c231a71477f151d48f79cc83a625ba21` |
| `added_tokens.json` | `dc046d04c9b0ada7ae6f1dc89c465801799acdf0c9a6aab8c15a1b2d5ca4e91f` |

`router_threshold_fp32.json` is the **confidence threshold** used by the demo, calibrated
on a held-out dev split (never on test). It flags the lowest-confidence ~4% of inputs.

## Intended use

Triage and routing of contract clauses — suggesting a provision type, and **saying when it
is unsure**. It is a research artefact.

**Not legal advice, and not a substitute for a lawyer reading the contract.**

## Limitations — read these

- **SINGLE SEED.** One training run. No mean ± std, so the figure above carries no
  variance estimate and should not be compared to multi-seed numbers as if it did.
- **`needs_review` means low confidence.** The demo flags inputs whose top-two-class
  margin falls below the calibrated threshold. On exactly those rows the model is
  measured at **~0.35 accuracy** against ~0.87 overall — the flag is reliable as a signal
  that the answer is doubtful, and it is not a second opinion.
- **Escalating flagged rows to a frontier LLM was measured and did not help**
  (+0.0008 macro-F1, 95% CI [−0.0060, +0.0072]). The demo therefore flags rather than
  escalating.
- **Domain.** US SEC EDGAR Exhibit-10 filings. Behaviour on other jurisdictions, contract
  types, or non-English text is untested.
- **Chronological drift.** Trained on 2016–2017, evaluated on 2019. Performance on newer
  filings is unmeasured.
- **Long clauses are truncated** at 512 tokens.
- **Class imbalance.** Rare classes are far weaker than the aggregate suggests.
- **INT8 is not served.** An INT8 quantisation of these weights collapses to chance on
  CPUs without AVX-512 VNNI, silently and with no error raised. FP32 is served for that
  reason.

## Attribution

This model is a **derivative work** of the LEDGAR task as distributed in **LexGLUE**,
which is licensed **CC-BY-4.0**, and of **microsoft/deberta-v3-base**, licensed **MIT**.
It is released under **CC-BY-4.0** to carry the data licence forward.

> **A licensing caveat, stated rather than buried.** The `coastalcph/lex_glue` dataset
> card carries the machine-readable tag `license: cc-by-4.0`, but its prose *"Licensing
> Information"* section is unfilled (`[More Information Needed]`). This model relies on
> the `cc-by-4.0` tag as the licence of record. Anyone putting this to a use where the
> distinction matters should confirm terms with the LexGLUE authors.

**LexGLUE** — Chalkidis, Jana, Hartung, Bommarito, Androutsopoulos, Katz, Aletras.
*LexGLUE: A Benchmark Dataset for Legal Language Understanding in English.* ACL 2022.
<https://arxiv.org/abs/2110.00976>

```bibtex
@inproceedings{chalkidis-etal-2021-lexglue,
        title={LexGLUE: A Benchmark Dataset for Legal Language Understanding in English},
        author={Chalkidis, Ilias and Jana, Abhik and Hartung, Dirk and
        Bommarito, Michael and Androutsopoulos, Ion and Katz, Daniel Martin and
        Aletras, Nikolaos},
        year={2022},
        booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics},
        address={Dubln, Ireland},
}
```

**LEDGAR** (the underlying corpus) — Tuggener, von Däniken, Peetz, Cieliebak.
*LEDGAR: A Large-Scale Multi-label Corpus for Text Classification of Legal Provisions in
Contracts.* LREC 2020. <https://aclanthology.org/2020.lrec-1.155/>

**DeBERTa-v3** — He, Liu, Gao, Chen. *DeBERTa: Decoding-enhanced BERT with Disentangled
Attention.* <https://arxiv.org/abs/2006.03654> ·
*DeBERTaV3.* <https://arxiv.org/abs/2111.09543>

Base model `microsoft/deberta-v3-base` is MIT licensed:

```
MIT License

Copyright (c) Microsoft Corporation.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
