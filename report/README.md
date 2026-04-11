# CMPE 492 Midterm Report

## Structure

```
report/
├── report.tex              # Main LaTeX source
├── report.pdf              # Compiled PDF (52 pages)
├── references.bib          # BibTeX bibliography
├── styles/
│   └── fbe_tez_v11.bst     # Bogazici FBE bibliography style
├── figures/                 # Diagrams (PlantUML sources + rendered PNGs)
│   ├── use_case_diagram.*   # Use case diagram (Ch. 5)
│   ├── chat_sequence.*      # Chat request sequence diagram (Ch. 6)
│   └── eval_pipeline.*      # Evaluation pipeline activity diagram (Ch. 6)
└── results/                 # Experiment result figures (Ch. 8)
    ├── chunking/            # Retrieval relevance by chunking strategy
    ├── reranker/            # nDCG with/without reranking
    ├── concept_normalization/ # Baseline vs. custom embedder comparison
    ├── obfuscation/         # Obfuscation blocked rate by substitution level
    └── turboquant/          # Memory, nDCG, and speed at different bit-widths
```

## How to Compile

Requires a TeX distribution (e.g., MacTeX, TeX Live).

```bash
cd report/
pdflatex report.tex
bibtex report
pdflatex report.tex
pdflatex report.tex
```

Three `pdflatex` passes are needed to resolve cross-references and the table of contents. `bibtex` resolves citations from `references.bib`.

## Regenerating PlantUML Diagrams

If you edit the `.puml` files, regenerate the PNGs:

```bash
# Requires Java and plantuml.jar
java -jar plantuml.jar figures/*.puml
```

## Missing Figures

Some figures referenced in the methodology chapters are TODO placeholders (e.g., `rag_pipeline.png`, `chunking_comparison.png`, `turboquant_pipeline.png`, `chat_ui.png`). These produce warnings during compilation but do not prevent the PDF from building. Add the corresponding images to `figures/` to resolve.
