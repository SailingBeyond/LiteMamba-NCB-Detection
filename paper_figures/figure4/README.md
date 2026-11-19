# Figure 4. Evaluation on external templates

`Figure4_manuscript.png` is the author-supplied Figure 4 selected for the current manuscript. The file is preserved byte for byte. `provenance.json` records its SHA-256 and the provenance of the inputs derived from `figure4_gatedconv_panels_v1_3_3.zip`.

From the staging directory, install the plotting packages and run the CPU renderer.

```bash
python3 -m pip install -r paper_figures/figure4/requirements.txt
bash paper_figures/figure4/run_plot.sh --output-dir reproduced/figure4
```

`reproduced/figure4/figures/Figure4_overview.png` is the newly rendered overview. Independent panels and subpanels are exported as PNG, PDF and SVG. Plotting data are written to `reproduced/figure4/source_data/`. The overview and the manuscript PNG are separate exports and may have different dimensions or pixel values.

- **A** shows X and Y recall for each of the three external test groups. Error bars show sample standard deviations across three training seeds.
- **B** shows the percentage of calls assigned to any canonical base on each DNA control template. Each point is one template.
- **C** shows X and Y recall by reference, strand and masked six-base context. From left to right, the subplots show XNA1–12, XNA13–16 and XNA17–20.
- **D** shows the percentage of true X or true Y observations assigned to each incorrect base. The denominator includes all observations of that true class in the relevant test group.

The renderer uses the frozen six-class GatedConv counts for seeds 42, 2024 and 2025. It checks the original argmax counts against the zero-threshold entries and recomputes the metrics before plotting. The retained count and metric tables match the original plotting inputs. The smaller archives retain only the input members required for this replay, with machine-specific provenance removed from the collection metadata.
