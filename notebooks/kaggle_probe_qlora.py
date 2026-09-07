# =============================================================================
# QLoRA SMOKE PROBE — run this BEFORE kaggle_tier1.py. ~2 minutes, GPU on.
#
# Exercises the exact path that has now broken twice and that CANNOT be verified
# off-CUDA: 4-bit load -> LoRA wrap -> one real training step -> one generation.
#
# Run CELL 1 of kaggle_tier1.py first, RESTART THE KERNEL, then run this.
# If this passes, the dependency set works and Tier 1 can start.
# If it fails, it fails in ~2 minutes instead of after hours.
# =============================================================================
# ONE GPU, PINNED BEFORE TORCH IS IMPORTED — must be the first executable line.
# torch reads CUDA_VISIBLE_DEVICES at CUDA init; setting it after `import torch` does
# nothing and says nothing. On Kaggle's "GPU T4 x2", two visible devices make
# transformers 5.0.0 wrap the 4-bit model in DataParallel (its guard at
# Trainer._wrap_model ~line 1665 tests is_loaded_in_8bit only, never is_loaded_in_4bit)
# and silently double the effective batch, since
# train_batch_size = per_device_train_batch_size * max(1, n_gpu).
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import inspect, sys, time, traceback

# Precision, stated once. Nothing is left to a library default: transformers 5.0.0
# changed from_pretrained's dtype default to "auto", which reads Qwen2.5's config
# torch_dtype (bfloat16) — and T4 is sm_75 with no bf16 support.
MODEL_DTYPE_NAME, COMPUTE_DTYPE_NAME = "float16", "float16"
USE_FP16, USE_BF16 = True, False

def step(n, what):
    print(f"\n[{n}] {what}", flush=True)

ok = True
try:
    step(1, "versions")
    import torch, transformers, trl, peft, bitsandbytes
    v = {"transformers": transformers.__version__, "trl": trl.__version__,
         "peft": peft.__version__, "bitsandbytes": bitsandbytes.__version__,
         "torch": torch.__version__}
    for k, x in v.items(): print(f"    {k:<14} {x}")
    assert v["trl"] == "1.12.0", f"trl {v['trl']} != 1.12.0 — did you restart the kernel?"
    assert v["peft"] == "0.20.0", f"peft {v['peft']} != 0.20.0 — did you restart the kernel?"
    print(f"    CUDA available: {torch.cuda.is_available()} | "
          f"devices: {torch.cuda.device_count()} | "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a'}")
    assert torch.cuda.is_available(), "no GPU — set Accelerator to GPU"

    step(2, "peft internal consistency at the crash site")
    from peft import LoraConfig
    import pathlib
    bnb_src = (pathlib.Path(peft.__file__).parent / "tuners" / "lora" / "bnb.py").read_text()
    uses_velora = "velora_config" in bnb_src
    has_velora = "velora_config" in LoraConfig.__dataclass_fields__
    print(f"    bnb.py references velora_config : {uses_velora}")
    print(f"    LoraConfig defines velora_config: {has_velora}")
    assert uses_velora == has_velora, (
        "MIXED PEFT INSTALL — bnb.py and LoraConfig come from different versions. "
        "This is the exact 'velora_config' crash. Re-run CELL 1 and RESTART the kernel.")
    print("    consistent")

    step("2b", "precision and device gate — the check that turns a step-5 crash "
               "into a step-2 message")
    # Every T4 failure so far has been a disagreement between values knowable in the
    # first ten seconds. Resolve them all here and compare, rather than discovering the
    # mismatch after a backward pass.
    from transformers import AutoConfig
    MODEL_DTYPE = getattr(torch, MODEL_DTYPE_NAME)
    COMPUTE_DTYPE = getattr(torch, COMPUTE_DTYPE_NAME)
    cap = torch.cuda.get_device_capability(0)
    bf16_ok = torch.cuda.is_bf16_supported()
    cfg_dtype = AutoConfig.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct").dtype
    print(f"    visible devices     : {torch.cuda.device_count()} "
          f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r})")
    print(f"    gpu                 : {torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]}")
    print(f"    bf16 supported here : {bf16_ok}")
    print(f"    model config dtype  : {cfg_dtype}   <- what dtype='auto' WOULD pick")
    print(f"    MODEL_DTYPE (forced): {MODEL_DTYPE}")
    print(f"    COMPUTE_DTYPE       : {COMPUTE_DTYPE}")
    print(f"    trainer precision   : fp16={USE_FP16} bf16={USE_BF16}")
    assert torch.cuda.device_count() == 1, (
        f"{torch.cuda.device_count()} GPUs visible — CUDA_VISIBLE_DEVICES did not take. "
        f"It must be set BEFORE torch is imported.")
    assert MODEL_DTYPE == COMPUTE_DTYPE, (
        f"MODEL_DTYPE {MODEL_DTYPE} != COMPUTE_DTYPE {COMPUTE_DTYPE} — bnb dequantises "
        f"into the compute dtype, and a mismatch is silent drift, not an error")
    assert USE_FP16 != USE_BF16, "exactly one of fp16/bf16"
    assert not (USE_BF16 and not bf16_ok), \
        f"bf16 requested on sm_{cap[0]}{cap[1]}, which has no bf16 support"
    assert not (MODEL_DTYPE is torch.bfloat16 and not bf16_ok), \
        f"MODEL_DTYPE is bf16 on sm_{cap[0]}{cap[1]} — this is the dtype='auto' trap"
    if cfg_dtype is torch.bfloat16 and not bf16_ok:
        print(f"    NOTE: the config asks for bf16 and this GPU cannot do it. This is "
              f"exactly why dtype is passed explicitly and never left at 'auto'.")
    print("    precision and device agree")

    step(3, "4-bit load")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    M = "Qwen/Qwen2.5-1.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(M); tok.pad_token = tok.pad_token or tok.eos_token
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=COMPUTE_DTYPE, bnb_4bit_use_double_quant=True)
    # dtype is passed EXPLICITLY. transformers 5.0.0 defaults it to "auto", which reads
    # this model's config torch_dtype (bfloat16) — unusable on sm_75.
    base = AutoModelForCausalLM.from_pretrained(M, device_map={"": 0},
                                                quantization_config=quant,
                                                dtype=MODEL_DTYPE)
    print(f"    loaded on {next(base.parameters()).device}, "
          f"{sum(p.numel() for p in base.parameters())/1e9:.2f}B params")

    step(4, "LoRA wrap on the 4-bit model (the path that crashed)")
    from peft import prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig
    base = prepare_model_for_kbit_training(base)
    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])

    step(5, "one real training step on 10 rows")
    from datasets import Dataset
    rows = [{"text": tok.apply_chat_template(
                [{"role":"system","content":"Classify the clause."},
                 {"role":"user","content":f"Provision number {i} concerning notices."}],
                tokenize=False, add_generation_prompt=True) + "Notices" + tok.eos_token}
            for i in range(10)]
    SEQ = "max_length" if "max_length" in SFTConfig.__dataclass_fields__ else "max_seq_length"
    print(f"    sequence-length kwarg resolved to {SEQ!r}")
    cfg = SFTConfig(output_dir="/kaggle/working/_probe", seed=1, max_steps=1,
                    per_device_train_batch_size=2, gradient_accumulation_steps=1,
                    learning_rate=2e-4, logging_steps=1, report_to=[],
                    save_strategy="no", gradient_checkpointing=True,
                    fp16=USE_FP16, bf16=USE_BF16,  # bf16's default is None, not False
                    **{SEQ: 512})
    tok.padding_side = "right"
    tr = SFTTrainer(model=base, train_dataset=Dataset.from_list(rows),
                    peft_config=lcfg, args=cfg)

    # UNDO TRL's UNCONDITIONAL bf16 CAST. trl 1.12.0 SFTTrainer.__init__, ~line 1154:
    #     if _is_quantized_model:
    #         for param in model.parameters():
    #             if param.requires_grad:
    #                 param.data = param.data.to(torch.bfloat16)
    # No GPU capability check, no reference to args.fp16. On sm_75 fp16 AMP's GradScaler
    # then raises NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda"
    # not implemented for 'BFloat16'. Passing dtype= to from_pretrained does NOT fix it:
    # prepare_model_for_kbit_training casts to fp32 and TRL's cast runs last. fp32 is the
    # right target — AMP keeps master weights in fp32. See PREREGISTRATION 3r.
    # THIS BLOCK IS DUPLICATED VERBATIM IN kaggle_tier1.py. Change both or neither.
    _pre = {str(p_.dtype) for p_ in tr.model.parameters() if p_.requires_grad}
    _recast = 0
    for p_ in tr.model.parameters():
        if p_.requires_grad and p_.dtype != torch.float32:
            p_.data = p_.data.to(torch.float32); _recast += 1
    _post = {str(p_.dtype) for p_ in tr.model.parameters() if p_.requires_grad}
    print(f"    trainable dtypes: TRL left {_pre} -> re-cast to {_post} "
          f"({_recast} tensors)")
    assert _post == {"torch.float32"}, (
        f"trainable params are {_post}, not fp32 — the GradScaler will fail")
    if "torch.bfloat16" not in _pre:
        print(f"    NOTE: TRL did NOT leave bf16 this time ({_pre}). The cast may have "
              f"changed; re-read SFTTrainer.__init__ before assuming the fix is still "
              f"needed, and do not silently keep a no-op.")

    _t_train = time.time()
    out = tr.train()
    _train_s = time.time() - _t_train
    print(f"    training loss after 1 step: {out.training_loss:.4f}")
    assert out.training_loss == out.training_loss, "loss is NaN"

    # THROUGHPUT, MEASURED. The 3-5h/seed figure in RUNNING.md was an estimate made
    # for two devices; one T4 is the real configuration and the quota decision needs a
    # number, not an extrapolation. This is one warm-up-inclusive step on 10 short rows,
    # so it OVERSTATES per-step cost (fixed setup amortised over one step) while
    # UNDERSTATING per-token cost (these rows are far shorter than the 607-token fit
    # mean). It is an order-of-magnitude check for the quota, not a benchmark — it is
    # printed as an estimate and named as one.
    _steps = max(1, out.global_step)
    _s_per_step = _train_s / _steps
    _fit_rows, _eff_batch = 57000, 16
    _proj_h_ESTIMATE = (_fit_rows / _eff_batch) * _s_per_step / 3600
    print(f"    {_train_s:.1f}s for {_steps} step(s) = {_s_per_step:.2f}s/step "
          f"at effective batch {2*1}")
    print(f"    ROUGH per-seed projection: {_proj_h_ESTIMATE:.1f}h "
          f"({_fit_rows:,} rows / effective batch {_eff_batch} x {_s_per_step:.2f}s)")
    print(f"    -> 3 seeds ~ {3*_proj_h_ESTIMATE:.0f}h of a 30h weekly quota. "
          f"ESTIMATE ONLY — short rows, one step, warm-up included.")

    # gradient checkpointing + prepare_model_for_kbit_training is the documented
    # sticking point: with reentrant checkpointing and nothing upstream of the LoRA
    # layers requiring grad, the backward pass raises "element 0 of tensors does not
    # require grad". Measured under transformers 5.0.0 / trl 1.12.0 / peft 0.20.0 (see
    # PREREGISTRATION 3q), the kwarg is NOT needed and is NOT passed. Both facts that
    # make it unnecessary are properties of the installed libraries, so they are
    # verified here rather than assumed.
    import functools as _ft
    _bound = {repr(f.keywords) for f in
              (getattr(m_, "_gradient_checkpointing_func", None) for m_ in tr.model.modules())
              if isinstance(f, _ft.partial)}
    print(f"    checkpoint kwargs actually bound: {_bound or '{none — checkpointing off}'}")
    assert _bound, ("gradient checkpointing is enabled in SFTConfig but no module holds a "
                    "checkpoint function — the setting did not take")
    # A step that runs but moves nothing is a failure presenting as success. grad_norm
    # is the direct evidence that gradients reached the adapter.
    _gn = [r["grad_norm"] for r in tr.state.log_history if r.get("grad_norm") is not None]
    print(f"    grad_norm: {_gn}")
    assert _gn and all(g == g for g in _gn) and max(_gn) > 0, (
        f"no positive finite grad_norm logged ({_gn}) — the step ran but no gradient "
        f"reached the LoRA parameters. If this fires, pass "
        f"gradient_checkpointing_kwargs={{'use_reentrant': False}} in SFTConfig here AND "
        f"in kaggle_tier1.py, and record why the measurement in 3q did not hold on CUDA.")

    step(6, "generation with left padding")
    m = tr.model
    m.config.use_cache = True
    if hasattr(m, "generation_config"): m.generation_config.use_cache = True
    m.eval(); tok.padding_side = "left"
    enc = tok([rows[0]["text"], rows[1]["text"]], return_tensors="pt",
              padding=True).to(m.device)
    with torch.no_grad():
        g = m.generate(**enc, max_new_tokens=8, do_sample=False,
                       pad_token_id=tok.pad_token_id)
    for j in range(2):
        print(f"    -> {tok.decode(g[j][enc['input_ids'].shape[1]:], skip_special_tokens=True)!r}")

except Exception:
    ok = False
    traceback.print_exc()

print("\n" + "=" * 70)
print("PROBE PASSED — the dependency set runs QLoRA end to end. Start Tier 1."
      if ok else
      "PROBE FAILED — read the traceback above. Do NOT start Tier 1.")
print("=" * 70)
