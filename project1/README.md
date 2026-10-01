# COMP5423 Project 1 · Tang 300

**Individual · 5% of the subject grade**
**Released:** Friday 25 September · **Due:** Friday 16 October, 23:00 (Hong Kong time)

## What this project is about

You will build a small GPT and train it on poems. Your model then writes short poems. After the deadline, the whole class reads the poems written by everyone — anonymous.

Your work is to **complete the six functions in `student.py`**, pick sensible settings in `config.json`, train, generate, and write a one-page report. The full requirements — what you may change, how the report is marked, and how the class rating works — are on the course site.

## Environment setup

- **Python 3.9 or newer**, with three packages:

  ```
  pip install torch safetensors matplotlib
  ```

  The default `torch` wheel runs on the CPU. If your computer has an NVIDIA GPU and you want the faster CUDA build, install the version matching your CUDA setup from [pytorch.org/get-started](https://pytorch.org/get-started/locally/). On an Apple-silicon Mac, no extra step is needed: the standard wheel already includes Metal (MPS) support.

- **GPU is Not a Must.** Rough training time for the default 5,000 steps: Mac with an M chip 10–20 minutes, ordinary laptop 1–3 hours (start it before you sleep), NVIDIA GPU a few minutes. `train.py` picks the fastest device automatically (CUDA → Metal → CPU). Google Colab is a reasonable alternative if your machine is slow.
