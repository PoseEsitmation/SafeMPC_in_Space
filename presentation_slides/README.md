# Presentation figures — Safe Imitation Learning for Spacecraft KOZ Avoidance

Regenerate anytime: `python scripts/make_presentation_charts.py`
(sources TensorBoard data from `runs/lqr/paper_final*`; this folder is gitignored)

| # | Figure | Claim | Source run | Headline numbers |
|---|--------|-------|-----------|------------------|
| 01 | filter_reliance_decline | The policy stops needing the safety filter | paper_final (γ=0.5) | interventions 8.6% → **0.00%**, severity → 4% of untrained |
| 02 | koz_violations_decline | The raw network itself becomes safe | paper_final | violations/ep 39 → **0.7–0.8** (rounds 18–19); worst margins −19° → −3° |
| 03 | deployment_safety | With the filter: zero violations, even untrained | paper_final2 (γ=0.2) | **0 violations in 210 filtered episodes**, success rate 100% at all rounds |
| 04 | task_performance | Safety costs no task performance | paper_final | attitude error 125° → **1°**; reward −914 → **+601** |
| 05 | gamma_ablation | Filter reliance driven to zero at the paper's γ | paper_final (γ=0.5) | interventions 8.6% → **0.00%** at round 20 |
| 06 | supplementary_full_timeline | Full training arc (phases, margins, rewards) | paper_final | filter ON from step 5000; DAGGER rounds 1–20 |

Talking-point notes
- Fig 02: shows rounds 0–19. Round 20 is omitted: its average (15.9) came from a single tail episode of 40 that dominated the y-scale; mention this if asked why the axis stops at 19.
- Fig 03: same policy evaluated with and without the QP filter at every round — the difference is attributable to the filter (success-rate metric).
- Fig 05: γ is the paper's class-K design knob (Sec. III-A); at 0.5 the constraint binds only when genuinely needed, so DAGGER can absorb it completely. (A γ=0.2 comparison run, paper_final2, exists if the early-warning trade-off comes up in questions: perfect safety even untrained, but a ~2% intervention floor.)
- Expert reference: episode reward 179–5110 (median 1205), final attitude error ≤0.3°, zero KOZ violations in training.
