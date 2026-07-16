# Presentation figures — Safe Imitation Learning for Spacecraft KOZ Avoidance

Regenerate anytime: `python scripts/make_presentation_charts.py`
(sources TensorBoard data from `runs/lqr/paper_final*`; this folder is gitignored)

| # | Figure | Claim | Source run | Headline numbers |
|---|--------|-------|-----------|------------------|
| 01 | filter_reliance_decline | The policy stops needing the safety filter | paper_final (γ=0.5) | interventions 8.6% → **0.00%**, severity → 4% of untrained |
| 02 | koz_violations_decline | The raw network itself becomes safe | paper_final | violations/ep 39 → **0.7–0.8** (rounds 18–19); worst margins −19° → −3° |
| 03 | deployment_safety | With the filter: zero violations, even untrained | paper_final2 (γ=0.2) | **0 violations in 210 filtered episodes**, success rate 100% at all rounds |
| 04 | task_performance | Safety costs no task performance | paper_final | attitude error 125° → **1°**; reward −914 → **+601** |
| 05 | gamma_ablation | γ trade-off: internalisation vs early warning | paper_final vs paper_final2 | 0.5 → 0.00% endpoint; 0.2 → ~2% floor, binds 2.5× earlier |
| 06 | supplementary_full_timeline | Full training arc (phases, margins, rewards) | paper_final | filter ON from step 5000; DAGGER rounds 1–20 |

Talking-point notes
- Fig 02: the round-20 bar (15.9) is a single tail episode of 40 (annotated); quote rounds 18–19 and say so.
- Fig 03: same policy evaluated with and without the QP filter at every round — the difference is attributable to the filter (success-rate metric).
- Fig 05: γ is the paper's class-K design knob (Sec. III-A). Both runs share the pipeline, scenario, seeds and normalisation (frozen norms from baseline_27).
- Expert reference: episode reward 179–5110 (median 1205), final attitude error ≤0.3°, zero KOZ violations in training.
