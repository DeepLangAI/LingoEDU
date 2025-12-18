# Inference Component Usage Guide

## Environment Setup

- Python >= 3.10 is recommended
- Install using pip or uv:
  ```bash
  pip install -e inference
  # or
  uv pip install -e inference
  ```

## Quick Start

- EDU input data format

  ```jsonc
  {
    "type": "string",
    "infos": [
      {
        "txt": "string",      // text of sentence part
        "position": {},       // position dict, can be empty
        "tags": [],           // tags list, can be empty
        "label": ""           // label string, can be empty
      }
    ]
  }
  ```

- Refer to `infer.py` for inference usage

   ```python
   from edu_core.edu_func import TitleEduFunction
   # ...see infer.py for more code
   ```
