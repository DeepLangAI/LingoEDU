# Evaluation Component Usage Guide

## Environment Setup

- Python >= 3.10 is recommended
- Install using pip or uv:
  ```bash
  pip install -e evaluation
  # or
  uv pip install -e evaluation
  ```

## Quick Start

- Ground truth data format

  Ground truths are in markdown format.

- Refer to `evaluate.py` for evaluation usage

   ```
   from edu_evaluator.metrics.ted import TEDMetric
   # ...see evaluate.py for more code
   ```

  Both EDU predict outputs and ground truths are converted to trees. For ground truth markdowns, we use the maximum-score path algorithm to perform 1-to-N matching between a title and sentences.
  
  TED and DLA scores are cacluated on tree pairs and printed in the terminal.
