# GatedConv for nanopore non-canonical base classification

GatedConv classifies bases at known candidate sites in Oxford Nanopore reads. It uses target-centered ionic current, masked six-base sequence context, signal summaries, base qualities and basecalling error features. The model classifies the fourth position of each six-base window. `X` and `Y` are labels for the two non-canonical bases in the study.

This repository retains its original GitHub name, `LiteMamba-NCB-Detection`. This README describes the revised GatedConv workflow. See the [first-submission documentation](docs/first-submission-workflow.md) for the original pipeline in `scripts/`.

| Path | Contents |
| --- | --- |
| [`paper_figures/`](paper_figures/) | Code and frozen inputs for Figures 2E–F, 3B, 3D–F, 4A–D, S6 and S7, and Supplementary Table S1. |
| [`workflow/`](workflow/) | Revised preprocessing, dataset splits, training and evaluation code. [Workflow instructions](workflow/README.md) |
| [`workflow/same_source/`](workflow/same_source/) | Code and input requirements for the same-source controls in Supplementary Table S5. [Control workflow](workflow/same_source/README.md) |
| [`release_assets/`](release_assets/) | Checkpoint and split manifest index with SHA-256 digests. |

## Reproduce the figures and Supplementary Table S1

Run these commands from the repository root in a Python 3.12 environment. These six CPU workflows use the frozen inputs in the repository.

```bash
python3 -m pip install -r paper_figures/s6/requirements.txt
python3 paper_figures/figure2ef/plot.py --output-dir reproduced/figure2ef
python3 paper_figures/figure3b/plot.py --output-dir reproduced/figure3b
python3 paper_figures/figure3df/plot.py --output-dir reproduced/figure3df
bash paper_figures/figure4/run_plot.sh --output-dir reproduced/figure4
python3 paper_figures/s6/reproduce_s6.py --output-dir reproduced/s6
python3 paper_figures/s7/plot.py --output-dir reproduced/s7
```

| Figure or table | Output under `reproduced/` |
| --- | --- |
| [2E–F](paper_figures/figure2ef/) | `figure2ef/Figure2E_zoom.png` and `figure2ef/Figure2F_zoom.png` |
| [3B and Supplementary Table S1](paper_figures/figure3b/) | `figure3b/Figure3B_accuracy.png`, `figure3b/Supplementary_Table_S1.csv` and `figure3b/Figure3B_per_seed.csv` |
| [3D–F](paper_figures/figure3df/) | `figure3df/figures/Figure3D_GatedConv_seed42.png`, `Figure3E_paired_GatedConv_5X.png` and `Figure3F_paired_GatedConv_5Y.png` |
| [4A–D](paper_figures/figure4/) | `figure4/figures/Figure4_overview.png` and individual panels |
| [S6](paper_figures/s6/) | `s6/figures/Supplementary_Figure_S6_GatedConv.png` |
| [S7](paper_figures/s7/) | `s7/figures/Supplementary_Figure_S7_GatedConv.png` |

Figure 3B and Supplementary Table S1 are recalculated from frozen test-set confusion counts for the three seeds. The UMAP plots use the saved coordinates of held-out test embeddings. Figure 4 uses frozen counts to recompute the metrics. The [manuscript Figure 4 PNG](paper_figures/figure4/Figure4_manuscript.png) is provided alongside the code and provenance for the regenerated overview. Supplementary Figure S6 supports `--verify` for byte-for-byte comparison with the reference PNGs in the documented rendering environment.

## Train or evaluate the models

See the [workflow instructions](workflow/README.md) for task-specific dependencies, input formats and commands. The revised binary and main multiclass tasks target a 7:2:1 train, validation and test split. The binary tasks use a shared read-grouped manifest. The main ATCGXY, ATCGX and ATCGY manifests use sequencing read IDs where available and source-row identifiers for historical positive examples under the one-row-per-read design. The partition audits check disjointness for the recorded identities. The three same-source controls use separate manifests.

Fresh training requires feature tables and corresponding metadata. Historical evaluation requires the original data and run records documented for each launcher. The figure and table commands above are the reproducible entry points for the packaged aggregate results. The model and read-split archives supply checkpoints and assignments only.

## Checkpoints and split manifests

See [GitHub Releases](https://github.com/SailingBeyond/LiteMamba-NCB-Detection/releases) for the matching checkpoint and split manifest archives when the revision release is available.

The [asset index](release_assets/asset_index.json) maps 72 checkpoints to their primary split manifests, architecture, run and seed. The six archives cover the binary tasks, ATCGXY, ATCGX, ATCGY, ablations and same-source controls. The ablation runs use the ATCGXY split manifest. The index records 75 files verified against their original frozen SHA-256 digests and four recovered split manifests verified by split-count and read-assignment audits.

After downloading the six `gatedconv-*.tar.gz` files, `asset_index.json`, `SHA256SUMS` and `SHA256SUMS.archives` into `gatedconv_release_attachments/`, run these commands from the repository root on Linux. The files are unpacked together so the checkpoint and split paths match the index.

```bash
(
  set -eu
  cd gatedconv_release_attachments
  sha256sum --check --strict SHA256SUMS.archives
  archive_dir="$PWD"
  unpack_dir="$(mktemp -d "$PWD/unpacked.XXXXXXXX")"
  for archive_name in gatedconv-ablation-weights.tar.gz gatedconv-binary.tar.gz \
    gatedconv-main-ATCGX.tar.gz gatedconv-main-ATCGXY.tar.gz \
    gatedconv-main-ATCGY.tar.gz gatedconv-same-source-controls.tar.gz; do
    tar -xzf "$archive_dir/$archive_name" -C "$unpack_dir"
  done
  cd "$unpack_dir"
  sha256sum --check --strict "$archive_dir/SHA256SUMS"
  printf 'Verified weights and splits in %s\n' "$unpack_dir"
)
```

## License

The source code is available under the [MIT License](LICENSE).
