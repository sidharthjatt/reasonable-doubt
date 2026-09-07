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
import inspect, sys, traceback

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

    step(3, "4-bit load")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    M = "Qwen/Qwen2.5-1.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(M); tok.pad_token = tok.pad_token or tok.eos_token
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(M, device_map={"": 0},
                                                quantization_config=quant)
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
                    learning_rate=2e-4, fp16=True, logging_steps=1, report_to=[],
                    save_strategy="no", gradient_checkpointing=True, **{SEQ: 512})
    tok.padding_side = "right"
    tr = SFTTrainer(model=base, train_dataset=Dataset.from_list(rows),
                    peft_config=lcfg, args=cfg)
    out = tr.train()
    print(f"    training loss after 1 step: {out.training_loss:.4f}")
    assert out.training_loss == out.training_loss, "loss is NaN"

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
