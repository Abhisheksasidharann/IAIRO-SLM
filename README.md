Biomedical SLM — 252M Parameter Language Model Pretrained from Scratch
________________________________________________________________________
A 252M-parameter decoder-only language model pretrained from scratch on biomedical scientific text, built as a team project. This repository documents the architecture, training process, and evaluation, including honest documentation of where the model fails.

What This Is ?

This is a base language model, not a question-answering system or a clinical tool. It was trained with a single objective: predict the next token in biomedical text. It has not undergone instruction tuning, fine-tuning, or any alignment process. It should not be used, as-is, to answer medical questions, and this repository includes direct evidence of why.

<img width="910" height="732" alt="image" src="https://github.com/user-attachments/assets/283d5143-68a9-49be-9571-dcaa3d06f457" />


Architecture confirmed directly by loading the checkpoint and inspecting model.named_parameters() — not taken from documentation alone.

<img width="915" height="652" alt="image" src="https://github.com/user-attachments/assets/a4acf4e0-d375-484a-b552-1f603398aaf8" />


Quality filtering applied during data extraction (confirmed from source notebook):

English only, language score ≥ 0.95
Domain = biomedical, biomedical score ≥ 0.90
Educational score ≥ 3.5
Minimum text length 200 characters

Train/test split: deterministic, article-level, via SHA-256 hash of article ID (3% test fraction). This guarantees no paragraph-level leakage between splits by construction. The split mechanism is fully recovered and auditable from source code; the literal resulting list of which article IDs landed in which split was not preserved and cannot currently be independently re-verified.

A known embedding initialization bug was found and fixed during earlier experiments: the underlying library's default embedding initialization (std=1.0) is incompatible with weight tying and produces saturated initial loss (~440 cross-entropy). The production script explicitly reinitializes the tied embedding to N(0, 0.02) before training, with this fix documented inline in the training code.

Checkpoint used in this repository: best_model, saved at step 8,500 (the point the training script's own validation loop identified as the best checkpoint by fast-validation loss). This is distinct from the final step-8,870 weights, which were also saved but are not what this repository evaluates.

<img width="915" height="235" alt="image" src="https://github.com/user-attachments/assets/c0bd1743-4a21-4450-ba8e-0d455e16bd72" />


On equal token budgets (1.8B), the 250M model outperforms the 100M model, consistent with expected parameter-scaling behavior. Extending the 250M model's training to 4.65B tokens produced further improvement, consistent with the project's Chinchilla-style token budget assumption (~20 tokens per parameter).

Evaluation

Fast validation (5M tokens, used during training): final logged value at step 8,500 — loss 2.5469, perplexity 12.767.

Independent evaluation (5M tokens, run separately after training): loss 2.6232, perplexity 13.780, measured directly against the actual model.pt checkpoint used throughout this repository.

These two numbers come from different 5M-token samples of the same source dataset and distribution; the small gap between them (≈0.08 loss) is consistent with ordinary sampling variation rather than a discrepancy requiring explanation.

Not yet done: the training script explicitly notes "Run full 200M-token validation separately after training." This has not been completed. All loss and perplexity figures in this repository are on 5M-token samples, not the full planned validation set. Readers should treat these numbers as indicative, not definitive.

Generation Testing

Direct generation was run on 10 biomedical prompts using both greedy decoding and temperature/top-k sampling (T=0.8, top-k=50). Full outputs are in smoke_test_results.txt.

What works: the model produces fluent, grammatically correct biomedical English and reliably reproduces extremely common associations seen often in training data (e.g., metformin → type 2 diabetes, insulin resistance → elevated diabetes/hypertension risk).

What fails, and why it matters:

Greedy decoding frequently collapses into repetition loops. Example: a prompt about myocardial infarction degenerated into a repeating sentence structure after ~30 tokens.
Sampling reduces but does not eliminate repetition. One sampled output looped on the single word "computerised" for over a dozen repetitions.
The model confidently fabricates specific facts, and does so differently each time. Run twice with sampling, the same acetaminophen dosing prompt produced two different wrong answers in two different unit ranges. The same warfarin mechanism-of-action prompt produced two different, both incorrect, invented mechanisms (one involving cell membranes and cancer cells, one involving renal filtration and eGFR — warfarin's actual mechanism is vitamin K epoxide reductase inhibition, named in neither).

This is the central finding of this project: the model cannot reliably be trusted to answer specific, checkable biomedical facts from memory, even though it produces fluent, domain-appropriate language. This is expected behavior for a model of this scale trained this way — common patterns seen thousands of times are learned reliably; specific facts stated once in passing in training text are not memorized, and the model has no internal signal distinguishing a recalled fact from a plausible-sounding invention.

What Cannot Be Honestly Claimed
The complete source composition of the 4.65B-token training corpus beyond the recovered Biomed-Enriched extraction pipeline — no other recovered code path shows additional sources being mixed in, but this is based on the artifacts that survived, not a guarantee no other process touched the data
Whether the 5M-token "independent" evaluation set has zero overlap with training data — the exact token-level contamination check has not been run
The literal list of article IDs assigned to train vs. test (the split mechanism is known; its output was not preserved)
Results from the QK-Norm and 4e-4 learning rate ablation experiments beyond early, short-run validation losses (954 and lower step counts) — neither ablation ran long enough to demonstrate a conclusion, and the project's own results note this explicitly
Repository Contents
train_250m_final_full_4p65B_v1.py — the actual production training script
training.log, train_log.json — full step-by-step training logs
save_results.py — generation smoke-test script used to produce smoke_test_results.txt
perplexity.py — independent evaluation script.
smoke_test_results.txt — raw generation outputs, unedited, including failures
Data pipeline notebooks: extraction, audit, and train/test split
Status

This is a project demonstrating end-to-end small language model pretraining: data pipeline construction, scale-up experimentation, production training at scale, and honest evaluation. It is not a finished product and is not intended for clinical, medical, or production use in its current form.
