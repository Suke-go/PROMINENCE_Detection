# Visceral Resonance: Demo Paper

This directory contains the LaTeX source for the demo paper submission.

## Files

- `visceral_resonance.tex` - Main LaTeX source (ACM SIGCONF format)
- `Makefile` - Build script

## Building

```bash
# Using make
make

# Or manually with pdflatex
pdflatex visceral_resonance.tex
pdflatex visceral_resonance.tex  # Run twice for references
```

## Requirements

- LaTeX distribution (TeX Live, MiKTeX, etc.)
- `acmart` document class
- Standard packages: `amsmath`, `graphicx`, `tikz`, `booktabs`

## Key Improvements from Review

1. **Hypothesis clarified**: EMS-induced muscle contraction → intra-abdominal pressure → subthreshold interoceptive signals
2. **Stimulation site rationale**: Focus on abdomen with cultural, anatomical, and neuroscientific justification
3. **Evaluation plan added**: 7-point Likert scales for subjective experience
4. **References improved**: Proper academic citations with complete metadata
