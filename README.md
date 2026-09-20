# Beyond the Final Sketch

**A Process-Enriched Dataset and Exploratory Study of Human Drawing Supervision for Face Sketch Synthesis**

Authors: Lifen Weng*, Yiyang Shi

> Use the drawing process as supervision, rather than as an inference chain.

## Overview

This project studies face sketch synthesis with genuine human drawing trajectories. Its central idea is to use the drawing process as supervision during training, rather than as an inference chain. No SOTA claim is made.

## Dataset

CUFS provides the original face photographs; this project contributes the newly collected B01–B25 drawing trajectories, metadata, subject mappings, and split definitions. Users may need to obtain CUFS photographs separately under the original terms. The B01–B25 trajectories were created and curated by the School of Design Art, Xiamen University of Technology.

Dataset archival download: **Pending institutional release approval.**

**Release status.** This local directory is staging material only. The public GitHub repository URL and the dataset archival URL are pending authorized creation; see `dataset/release_notes/PUBLIC_RELEASE_ACTIONS.md` before release.

Each subject has one real human-drawn B01–B25 trajectory. The intermediate states were manually saved during actual digital drawing; they are neither interpolated pseudo-stages nor fixed semantic boundaries. See `dataset/` for split, subject-mapping, and release-note documentation.

## Repository structure

```text
dataset/                release metadata, mappings, and split definitions
process_supervision/    Direct, Sparse B7/B19, Full25, Ordered, Shuffled
cascade_exploration/    preliminary Face → B7 → B19 → B25 flow
diffusion_exploration/  documentation for the reported reduced MLGLDM-inspired exploration
pix2pix_reference/      project adapters for the official endpoint-only reference
evaluation/             reported metrics and evaluation material
configs/                frozen protocol configuration documentation
figures_examples/       frozen paper figures
```

## Installation

Create the project environment with the pinned packages in `requirements.txt`, then configure local dataset locations and the released split files. No dataset or checkpoint download URL is invented in this staging repository; the public archival link will be added after release approval.

## Process supervision

The core comparison studies Direct, Sparse B7/B19, Full25, Ordered, and Shuffled process supervision. Architecture and loss material is in `process_supervision/common/`; condition-specific entry points are in the five condition directories. These experiments use the drawing process as training supervision and retain direct Face → B25 inference.

## Cascade exploration

`cascade_exploration/` provides minimal code for the preliminary Face → B7 → B19 → B25 design exploration. It illustrates why prediction errors can cascade and is not a formal benchmark.

## Diffusion exploration

`diffusion_exploration/` documents the reported reduced direct, Full25-process, and weak-process exploration. Its releasable implementation will be added only after source-level approval; it is a paper-inspired exploratory backbone, **not** an official reproduction of MLGLDM. Stable Diffusion 1.5 development scripts are not included.

## pix2pix reference

pix2pix is an external endpoint-only Face → B25 reference. Obtain the upstream implementation at the pinned official commit, then use the pairing adapter, frozen seed list, and evaluator in `pix2pix_reference/`. B01–B24 are not used by this reference.

## Reproduction

`process_supervision/` contains Direct, Sparse B7/B19, Full25, Ordered, and Shuffled study material. `cascade_exploration/` contains the minimal Face -> B7 -> B19 -> B25 exploration. `diffusion_exploration/` contains only reported reduced paper-inspired variants; it is **NOT an official reproduction of MLGLDM**. `pix2pix_reference/` contains our adapter and evaluator for the official endpoint-only reference, not a copy of its source repository.

Use the official pix2pix repository https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix at commit `2a7afba2895d52556dd5dfe07e8555ef657ced6f`. The frozen external-reference seeds are 3407, 2026, and 9317.

## Evaluation

Metrics are F1, IoU, L1, and EdgeLoss. F1/IoU emphasize structural overlap; L1/EdgeLoss emphasize pixel-level and local-edge discrepancy.

## Dataset access and CUFS attribution

The B01-B25 drawing trajectories are planned for separate archival release after institutional approval. CUFS provides the source face photographs; CUFS photographs are not redistributed in this repository or the planned dataset archive.

## Citation

See `CITATION.cff` for citation metadata.

## License

Code license: MIT. Dataset license: Pending institutional approval. The MIT license applies to code only and does not apply to the B01-B25 trajectories or CUFS photographs.

## Contact

Lifen Weng: 2009990509@xmut.edu.cn.
