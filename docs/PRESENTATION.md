# Outline

## 1. Motivation (2 minutes - 2/3 slides)

Applications: 
- Character animation
- Embodied AI
- AR environments
- Synthetic training data for embodied AI.

< nice visuals to represent applications >

Why generate human motion conditioned on text + scene + goal *together*?
Text says *what*, scene says *what's physically possible*, goal says *where it ends*.

"Research" question: 
Does joint text–scene–goal conditioning work on real-world dataset, and how does each condition affect the generated motion?

## 2. Landscape of approaches (2 minutes - 2/3 slides)

- Procedural/kinematic — rigid, no learning.
- Mocap retrieval/blending — limited to library, no novel combinations.
- Autoregressive deep models (RNN/GAN) — fast, but drift/mode collapse.
  One line only; nobody defends these now.
- Discrete-token / masked generative (T2M-GPT, MoMask) — currently lead
  text-only benchmarks, but VQ discards information and works poorly as a
  prior or with guidance.
- Diffusion-based motion generation — continuous, flexible conditioning,
  supports inpainting and per-condition CFG.

Prior work either conditions on one or two modalities, or trains only on synthetic or staged datasets:

| Method | Family | Text | Scene | Goal |
|---|---|---|---|---|
| MDM (2023) | diffusion, continuous | ✓ | — | — |
| MoMask (2024) | masked discrete tokens | ✓ | — | — |
| HUMANISE (2022) | cVAE | ✓ | ✓ | — |
| TRUMANS (2024) | autoregressive diffusion | — | ✓ | ✓ |
| LINGO (2024) | autoregressive diffusion | ✓ | ✓ | ✓ |
| ARDY (2026, NVIDIA) | autoregressive diffusion | ✓ | — | ✓ |
| **Ours** | diffusion, continuous | ✓ | ✓ | ✓ |

## 3. Justification (2 minutes - 2 slides)

The idea here is to justify some archtecture / design choices based on previous research, no ablation.

The claim is **one model conditioned on text + scene + goal simultaneously, trained at real-world scale dataset**.

- *Why now*: multi model conditioning needs all three channels aligned in one dataset preferably, which becomes posible with NymeriaPlus.

- *Why one model*: in a single model the conditions can attend to each other rather than being applied in sequence. The scene can veto a goal behind a wall, the goal can bias which part of the scene matters, and the text can change *how* the goal is reached. 

- *Why diffusion*: MDM proved that lightweight transformer denoiser already reaches good text-to-motion quality. Diffusion models are also well suited to conditioning signals, which makes classifier-free guidance, per-condition guidance scales, and inpainting standard tools here.

## 4. Method overview (4 minutes - 2 slides)

Data → Architecture

NymeriaPlus in context:

| Dataset | Hours | Sequences | Language | Scene | Capture |
|---|---|---|---|---|---|
| HumanML3D (2022) | ~28.6 h | 14,616 | ~45k descriptions | none | mocap (AMASS) |
| HUMANISE (2022) | ~11 h (1.2M frames) | 19,648 | templated | 643 ScanNet scans | synthetic pairing |
| TRUMANS (2024) | ~15 h (1.6M frames) | not reported | action labels | 100 lab indoor scenes | mocap in built scenes |
| NymeriaPlus (2026) | 300 h | 1,200 | 310.5K sentences | point clouds, ShapeR meshes, bounding boxes | egocentric, in the wild |
| **NymeriaPlus (filtered)** | ~182 h | 707 | 144k annotations | | |

Architecture: 

< nice archtecture diagram here >

## 5. Results / Ablations (8 minutes - 8/10 slides)

The idea is it to make it highly visual, some clips of generated motion vs ground truth.

- Qualitative: Clips of generated motion over real scene geometry, including failures. 

- Quantitative (?): MPJPE vs. ground truth, goal error, scene colision, ...

- Show capabilities of our model compared to existent solutions.

- **Ablation A** Condition dropout at test time (CFG) showing how each modality contributes to final motion.

- **Ablation B**: Compare text encoders and/or number of tokens. (Probably no time for it)

- **Ablation C**: Compare scene encoders and/or number of tokens. (Probably no time for it)

## 6. Limitations and Conclusion (2 minutes - 2 slides)

- *Data*. What makes this work possible is also what makes it hard. Real "unscripted" recordings come with errors that staged and synthetic datasets don't have: meshes are reconstructed rather than authored, annotations are incomplete and inconsistent even within the same room, and most of the time the person isn't interacting with the scene at all. The model has to absorb noise that staged and synthetic setups never expose it to.

- *Model*. The scene representation is coarse, so small objects and fine geometric variation are invisible to the model and precise contact isn't guaranteed.

- *Usability*. Conditions go out of distribution easily. During training, text, scene, and goal are sampled jointly from real recordings, so their combinations are correlated — a goal far from any observed trajectory, or an instruction that doesn't match the geometry, is a combination the model has never seen.

- *Conclusion*. A single lightweight diffusion model generates human motion conditioned on text, scene, and goal together, trained on real unscripted recordings rather than staged or synthetic capture, under the limitations discussed above. Follow ups could be better ways to handle the inconsistencies from dataset and properly encoding scene.

