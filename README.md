# In-Context Autonomous Network Incident Response: An End-to-End Large Language Model Agent Approach

This repository contains the artifacts related to the paper "In-Context Autonomous Network Incident Response: An End-to-End Large Language Model Agent Approach", which is accepted to AAAI 2026 Summer Symposium Series. we propose to leverage large language models’ (LLM) pre-trained security knowledge and in-context learning to create an end-to-end agentic solution for incident response planning. Specifically, our agent integrates four functionalities, perception, reasoning, planning, and action, into one lightweight LLM(14b model). Through fine-tuning and chain-of-thought reasoning, our LLM agent is capable of processing system logs and inferring the underlying network state (perception), updating its conjecture of attack models (reasoning), simulating consequences under different response strategies (planning), and generating an effective response (action).

<img width="990" height="330" alt="image" src="https://github.com/user-attachments/assets/b9b03469-6870-4c12-90fd-2fbb87486c82" />



## Requirements

- Python 3.8+
- `torch`
- `transformers`
- `peft`
- `bitsandbytes`
- `accelerate`

## Installation

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/TaoLi-NYU/llmagent4incidense-response-aaai26summer.git
cd llmagent4incidense-response-aaai26summer
pip install -e .
```

To install or upgrade the required dependencies:

```bash
pip install -e . --upgrade
```

## Development

Install the development dependencies:

```bash
pip install -r requirements_dev.txt
```

Run code checks and tests:

```bash
flake8 .
mypy .
pytest
```


## Author & Maintainer
Yiran Gao gaoyiran525@gmail.com

Kim Hammar <kimham@kth.se>

Tao Li li.tao@cityu.edu.hk

## Copyright and license

[LICENSE](LICENSE.md)

Creative Commons

Copyright (c) 2026 Yiran Gao, Kim Hammar, Tao Li.

