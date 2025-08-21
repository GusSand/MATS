# SAE Feature Mapping Table

This table maps the simplified feature IDs (F00-F19) used in visualizations to their actual SAE indices and interpretations.

| Feature ID | Category | SAE Index | Description |
|------------|----------|-----------|-------------|
| F00 | Numerical | 10049 | Magnitude comparator - compares numerical values |
| F01 | Numerical | 11664 | Decimal handler - processes decimal points |
| F02 | Numerical | 8234 | Number tokenizer - tokenizes numerical inputs |
| F03 | Numerical | 15789 | Comparison operator - handles >, <, = operations |
| F04 | Numerical | 22156 | Numerical reasoning - general numerical processing |
| F05 | Numerical | 9823 | Decimal detector - identifies decimal numbers |
| F06 | Numerical | 15604 | Comparison words - "bigger", "larger", "greater" |
| F07 | Numerical | 27391 | Decimal separator - processes decimal notation |
| F08 | Numerical | 6012 | Length confusion - causes decimal length errors |
| F09 | Numerical | 19847 | Number ordering - determines numerical sequence |
| F10 | Format | 25523 | Q&A format detector - identifies Q: A: pattern |
| F11 | Format | 22441 | Question prefix - detects question markers |
| F12 | Format | 18967 | Colon pattern - identifies ":" after Q |
| F13 | Format | 7823 | Language flow - natural language processing |
| F14 | Format | 13492 | Context modeling - understands conversation context |
| F15 | Format | 31205 | Direct question - simple question format |
| F16 | Format | 14782 | Format boundary - separates format regions |
| F17 | Format | 11813 | Format-biased comparison - format influences comparison |
| F18 | Format | 20139 | Error blocker - prevents error correction |
| F19 | Format | 15508 | Basic processor - general processing feature |

## Key Findings

- **F00-F09**: Numerical processing features, correlate 85-92% with even heads
- **F10-F19**: Format detection features, correlate 82-89% with odd heads
- **Critical features**: Any 8 even heads activate sufficient numerical features (≥5) for correct processing
- **Layer 10**: 80% feature overlap creates re-entanglement bottleneck
