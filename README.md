<div align="center">
<h1>LingoEDU</h1>
<p align="center">
🤗 <a href="https://huggingface.co/deeplang-ai/LingoEDU-4B">Hugging Face</a>
📜 <a href="https://arxiv.org/abs/2512.14244" target="_blank">Paper</a>
🏁 <a href="https://huggingface.co/datasets/deeplang-ai/StructBench" target="_blank">Benchmark</a>
</p>
</div>

## Introduction

This repository contains the code of paper [From Context to EDUs](https://arxiv.org/abs/2512.14244), which introduces the EDU-based Context Compressor, a novel explicit compression framework designed to preserve both global structure and fine-grained details. Empirical results demonstrate that our method achieves state-of-the-art structural prediction accuracy and significantly outperforms frontier LLMs while reducing costs. Furthermore, our structure-aware compression substantially enhances performance across downstream tasks ranging from long-context tasks to complex Deep Search scenarios.

## Performance

We evaluated our method on [StructBench](https://huggingface.co/datasets/deeplang-ai/StructBench) along with frontier LLMs and commercial parsing APIs. Our
method achieves SOTA structural accuracy with significantly lower costs.

<table>
  <thead>
    <tr>
      <th>Method</th>
      <th>Type</th>
      <th>TED (Structure) ↓</th>
      <th>DLA (Accuracy) ↑</th>
      <th>Cost ($/doc) ↓</th>
      <th>Latency (s) ↓</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>GPT-4o</td>
      <td rowspan="12">General LLM*</td>
      <td>6.22</td>
      <td>29.03%</td>
      <td>0.0210</td>
      <td>-</td>
    </tr>
    <tr>
      <td>GPT-4.1</td>
      <td>6.35</td>
      <td>37.90%</td>
      <td>0.0168</td>
      <td>-</td>
    </tr>
    <tr>
      <td>OpenAI o3</td>
      <td>5.51</td>
      <td>28.63%</td>
      <td>0.0168</td>
      <td>-</td>
    </tr>
    <tr>
      <td>OpenAI o4-mini</td>
      <td>5.87</td>
      <td>32.66%</td>
      <td>0.0092</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Claude-3.7-Sonnet</td>
      <td>6.65</td>
      <td>35.08%</td>
      <td>0.0286</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Claude-4</td>
      <td><u>5.08</u></td>
      <td><u>43.15%</u></td>
      <td>0.0286</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Gemini-2.5-flash</td>
      <td>5.82</td>
      <td>27.82%</td>
      <td>0.0040</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Gemini-2.5-pro</td>
      <td>5.61</td>
      <td>32.66%</td>
      <td>0.0162</td>
      <td>-</td>
    </tr>
    <tr>
      <td>DeepSeek-V3</td>
      <td>6.32</td>
      <td>33.47%</td>
      <td>0.0012</td>
      <td>-</td>
    </tr>
    <tr>
      <td>DeepSeek-R1</td>
      <td>6.26</td>
      <td>30.65%</td>
      <td>0.0046</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Qwen3-32B</td>
      <td>6.52</td>
      <td>26.21%</td>
      <td>0.0012</td>
      <td>10.17<sup>†</sup></td>
    </tr>
    <tr>
      <td>Qwen3-235B</td>
      <td>7.67</td>
      <td>19.10%</td>
      <td>0.0012</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Jina-Reader</td>
      <td rowspan="2">Parser API</td>
      <td>17.04</td>
      <td>-</td>
      <td><b>0.0004</b></td>
      <td>-</td>
    </tr>
    <tr>
      <td>Firecrawl</td>
      <td>16.81</td>
      <td>-</td>
      <td><u>0.0007</u></td>
      <td>-</td>
    </tr>
    <tr>
      <td><b>Our Method (LingoEDU)</b></td>
      <td><b>Specialized</b></td>
      <td><b>4.77</b></td>
      <td><b>49.60%</b></td>
      <td><u>0.0007</u></td>
      <td><b>1.20<sup>†</sup></b></td>
    </tr>
  </tbody>
</table>

## Experiments

To address whether structure-aware compression tangibly enhance performance for downstream tasks, we conducted experiments on several long-context task benchmarks, including LongBench, HLE and Browse-Comp-ZH, across senarios of standard long-context benchmarks and complex Deep Search pipelines. Our method achieved SOTA in all these benchmarks. See the details of implemention and evaluation in directory [experiments/](https://github.com/DeepLangAI/LingoEDU/tree/main/experiments).

- General Long-Context Understanding

<table>
  <thead>
    <tr>
      <th rowspan="2">Task Type</th>
      <th rowspan="2">Dataset</th>
      <th rowspan="2">Glyph</th>
      <th colspan="4">Gemini-2.5-Pro</th>
      <th colspan="4">GPT-4.1</th>
    </tr>
    <tr>
      <th>Standard</th>
      <th>Self-Sum</th>
      <th>Ours (LingoEDU)</th>
      <th>Δ</th>
      <th>Standard</th>
      <th>Self-Sum</th>
      <th>Ours (LingoEDU)</th>
      <th>Δ</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="4">Multi-Doc QA</td>
      <td>HotpotQA</td>
      <td>66.42</td>
      <td>35.20</td>
      <td>37.78</td>
      <td>40.46</td>
      <td>+14.94%</td>
      <td>65.83</td>
      <td><u>67.89</u></td>
      <td><b>70.11</b></td>
      <td>+6.50%</td>
    </tr>
    <tr>
      <td>2WikiMQA</td>
      <td>72.98</td>
      <td>38.10</td>
      <td>39.90</td>
      <td>40.91</td>
      <td>+7.38%</td>
      <td>72.98</td>
      <td><u>74.39</u></td>
      <td><b>74.68</b></td>
      <td>+2.33%</td>
    </tr>
    <tr>
      <td>Musique</td>
      <td>-</td>
      <td>28.55</td>
      <td>30.77</td>
      <td>31.22</td>
      <td>+9.35%</td>
      <td>51.90</td>
      <td><u>53.48</u></td>
      <td><b>54.86</b></td>
      <td>+5.70%</td>
    </tr>
    <tr>
      <td>DuReader</td>
      <td>-</td>
      <td>7.15</td>
      <td>7.79</td>
      <td>8.12</td>
      <td>+7.69%</td>
      <td>21.80</td>
      <td><u>23.51</u></td>
      <td><b>25.34</b></td>
      <td>+16.24%</td>
    </tr>
    <tr>
      <td rowspan="4">Summarization</td>
      <td>GovReport</td>
      <td>25.53</td>
      <td>4.10</td>
      <td>4.34</td>
      <td>4.25</td>
      <td>+2.44%</td>
      <td>29.97</td>
      <td><u>30.98</u></td>
      <td><b>31.56</b></td>
      <td>+2.94%</td>
    </tr>
    <tr>
      <td>QMSum</td>
      <td>19.78</td>
      <td>15.80</td>
      <td>16.53</td>
      <td>16.17</td>
      <td>+2.34%</td>
      <td><u>22.84</u></td>
      <td>22.53</td>
      <td><b>23.30</b></td>
      <td>+0.61%</td>
    </tr>
    <tr>
      <td>MultiNews</td>
      <td>-</td>
      <td>4.05</td>
      <td>4.44</td>
      <td>4.85</td>
      <td>+19.75%</td>
      <td><u>20.85</u></td>
      <td>22.06</td>
      <td><b>23.50</b></td>
      <td>+5.80%</td>
    </tr>
    <tr>
      <td>VCSum</td>
      <td>-</td>
      <td>5.80</td>
      <td>6.17</td>
      <td>6.36</td>
      <td>+9.66%</td>
      <td>12.50</td>
      <td><u>13.71</u></td>
      <td><b>14.62</b></td>
      <td>+8.96%</td>
    </tr>
    <tr>
      <td rowspan="4">Few-shot</td>
      <td>TREC</td>
      <td><b>82.62</b></td>
      <td>46.50</td>
      <td>49.00</td>
      <td>57.50</td>
      <td>+23.66%</td>
      <td>77.00</td>
      <td><u>80.50</u></td>
      <td>80.00</td>
      <td>+3.90%</td>
    </tr>
    <tr>
      <td>TriviaQA</td>
      <td>88.54</td>
      <td>59.85</td>
      <td>62.31</td>
      <td>63.25</td>
      <td>+1.25%</td>
      <td>90.07</td>
      <td><u>93.69</u></td>
      <td><b>93.76</b></td>
      <td>+4.10%</td>
    </tr>
    <tr>
      <td>SAMSum</td>
      <td>-</td>
      <td>20.45</td>
      <td>21.89</td>
      <td>23.80</td>
      <td>+11.39%</td>
      <td>39.20</td>
      <td><u>40.79</u></td>
      <td><b>41.68</b></td>
      <td>+6.33%</td>
    </tr>
    <tr>
      <td>LSHT</td>
      <td>-</td>
      <td>26.10</td>
      <td>29.50</td>
      <td>35.48</td>
      <td>+3.45%</td>
      <td>48.60</td>
      <td><u>50.50</u></td>
      <td><b>52.50</b></td>
      <td>+8.02%</td>
    </tr>
  </tbody>
</table>

- Deep Search

<table>
  <thead>
    <tr>
      <th rowspan="2">Model Backbone</th>
      <th colspan="4">HLE (Academic Reasoning)</th>
      <th colspan="4">BrowseComp-ZH (Noisy Web)</th>
    </tr>
    <tr>
      <th>Base</th>
      <th>Self-Sum</th>
      <th>Ours (LingoEDU)</th>
      <th>Δ</th>
      <th>Base</th>
      <th>Self-Sum</th>
      <th>Ours (LingoEDU)</th>
      <th>Δ</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>DeepSeek-R1</td>
      <td>9.0</td>
      <td>9.5</td>
      <td>13.6</td>
      <td>+51.11%</td>
      <td>18.8</td>
      <td>19.4</td>
      <td>20.4</td>
      <td>+8.51%</td>
    </tr>
    <tr>
      <td>Qwen3-235B-Thinking</td>
      <td>14.2</td>
      <td>14.7</td>
      <td>15.5</td>
      <td>+9.15%</td>
      <td>8.5</td>
      <td>9.0</td>
      <td>12.8</td>
      <td>+50.59%</td>
    </tr>
    <tr>
      <td>DeepSeek-V3.1</td>
      <td>14.5</td>
      <td>14.8</td>
      <td>15.6</td>
      <td>+7.59%</td>
      <td>29.2</td>
      <td>29.7</td>
      <td>38.7</td>
      <td>+32.53%</td>
    </tr>
    <tr>
      <td colspan="9"><em>Closed-Source Models</em></td>
    </tr>
    <tr>
      <td>GPT-5</td>
      <td>25.0</td>
      <td>25.9</td>
      <td>27.1</td>
      <td>+8.40%</td>
      <td>29.0</td>
      <td>29.8</td>
      <td>31.8</td>
      <td>+9.66%</td>
    </tr>
    <tr>
      <td>Claude Opus 4.1</td>
      <td>14.0</td>
      <td>14.8</td>
      <td>15.5</td>
      <td>+10.71%</td>
      <td>20.8</td>
      <td>21.5</td>
      <td>23.2</td>
      <td>+11.54%</td>
    </tr>
    <tr>
      <td>Gemini 3 Pro</td>
      <td>26.1</td>
      <td>26.7</td>
      <td>30.1</td>
      <td>+15.33%</td>
      <td>47.5</td>
      <td>48.0</td>
      <td>49.0</td>
      <td>+3.16%</td>
    </tr>
  </tbody>
</table>

## Usage

### Prepare

Preprocess the input article into sentences, with a data structure below:

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

You may do this via:

- Parse the document and extract text content using
  [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
  (for web pages),
  [poppler-utils](https://poppler.freedesktop.org/)
  (for PDF files),
  or OCR tools.
  We recommend keeping paragraphs segmented at this step.

- Split the text into sentences.
  A simple regex-based splitter is often sufficient;
  NLP toolkits such as
  [spaCy](https://spacy.io/)
  and
  [Stanza](https://stanfordnlp.github.io/stanza/)
  are also good options.

  Example regex splitter:

  ```python
  import re

  raw_sents = re.split(
      r'(?<=[;:,.!?；：，。！？…])\s*',
      text.strip()
  )

  sentences = [s for s in raw_sents if len(s.strip()) > 0]
  ```

💖 For web pages, we gladly present our wonderful tool WCD(Web Content Distill) which can convert your web pages as urls directly to inputs of the format above, open-sourced at: [WCD](https://github.com/DeepLangAI/wcd).

### Infer

```bash
pip install -e inference
python inference/infer.py --data_dir deeplang-ai/StructBench --inference_dir edu_output
```

Inference outputs will be generated under directory `edu_output`, each as a list of (level, start_sentence_index, end_sentence_index) tuples.

### Evaluate

```bash
pip install -e evaluation
python inference/evaluate.py --data_dir deeplang-ai/StructBench --inference_dir edu_output
```

TED and DLA scores will be printed in the terminal.

## Citation

If you find our work helpful, feel free to give us a cite.

```bibtex
@misc{zhou2025contextedusfaithfulstructured,
      title={From Context to EDUs: Faithful and Structured Context Compression via Elementary Discourse Unit Decomposition}, 
      author={Yiqing Zhou and Yu Lei and Shuzheng Si and Qingyan Sun and Wei Wang and Yifei Wu and Hao Wen and Gang Chen and Fanchao Qi and Maosong Sun},
      year={2025},
      eprint={2512.14244},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2512.14244}, 
}
```
