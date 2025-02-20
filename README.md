# ruGPT3XL Inference

This is a fork of the original [ruGPT3XL](https://huggingface.co/sberbank-ai/rugpt3xl) repository, for fast inference on any device.

The source code and weights are exactly the same as in the original.

## Quick Start

- Fast inference on any device.
- No `sparse_attention`
- < 7GB VRAM/RAM.
- `FORCE_CPU` flag for CPU inference.
- Latest versions of transformers and deepspeed.

## Setup

Use either:

### Poetry

```bash
poetry install
```

### Pip

```bash
pip install transformers deepspeed
```

## Usage

Run the model with:

```bash
python run.py
```

To use only the CPU:

```bash
FORCE_CPU=1 python run.py
```

```plaintext
[2025-02-04 01:44:03,376] [INFO] ...
Load checkpoint from ...
Model Loaded
Function generate Took 17.5580 seconds
Token speed: 2.75 tokens/sec
['Кто был президентом США в 2020? \nВ этом году выборы президента Соединенных Штатов Америки пройдут уже через несколько дней. И, как и всегда на протяжении ...']
```

## Speed Test Results

| Device                                | Token Speed         |
|---------------------------------------|---------------------|
| RTX 3080 10GB                 | 26.78 tokens/sec    |
| RTX 3070 8GB                  | 15.20 tokens/sec    |
| M1 Pro MPS/GPU                | 2.75 tokens/sec     |
| M1 Pro CPU                    | 0.66 tokens/sec     |

## License

Licensed under the Apache License, Version 2.0.
