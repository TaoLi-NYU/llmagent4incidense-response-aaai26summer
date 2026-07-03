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

## Artifacts

- The incident-response fine-tuning dataset used in our experiments is available on
  [Hugging Face](https://huggingface.co/datasets/kimhammar/CSLE-IncidentResponse-V1/tree/main).

- The LoRA adapter weights of our fine-tuned model are available on
  [Hugging Face](https://huggingface.co/GYR1-determine/llmagent4incident-response).

## Fine-tuning DeepSeek-R1-Distill-Qwen-14B on our action generation dataset

Command:

```bash
python examples/fine_tune_action_generation.py
```

Expected output:
```text
Fetching 4 files: 100% 4/4 [01:16<00:00, 19.04s/it]
Loading checkpoint shards: 100% 4/4 [00:33<00:00,  8.25s/it]
generation_config.json: 100% 181/181 [00:00<00:00, 2.02MB/s]
README.md: 100% 33.0/33.0 [00:00<00:00, 363kB/s]
action_examples.json: 100% 694M/694M [00:05<00:00, 136MB/s]
Generating train split: 1 examples [00:09,  9.84s/example]
Trainable parameters: 50331648

...

Step: 299, Epoch: 0.1086, Progress: 10.86%, Avg_loss=0.9460, LR=0.00084720, Grad_norm=0.3544, minutes: 315.9855
prediction:
I note that the attacker is actively communicating with our internal and external resources, so I choose to immediately isolate the affected hosts and block all traffic to and from the attacker IPs to stop further spread and data exfiltration.</think>
{
    "Action": "Isolate WikiServer, GitServer, and DevWorkstation; block all traffic to and from 185.140.53.11, 185.140.53.12, and 185.140.53.13 at firewalls and proxies.",
    "Explanation": "Immediate isolation and blocking halt attacker communication and lateral movement."
}
label:
I note that the attacker IPs are actively communicating with internal systems and facilitating lateral movement, so to immediately stop further spread and communication, I choose to block their IPs at the perimeter and isolate the most affected hosts to contain the attack.</think>
{
    "Action": "Block all traffic to attacker IPs 185.140.53.11, 185.140.53.12, and 185.140.53.13 at perimeter firewalls and immediately isolate WikiServer (203.0.113.120) and DevWorkstation (10.66.22.41) from the network.",
    "Explanation": "Cutting external and internal communication halts spread and C2, achieving immediate containment."
}<｜end▁of▁sentence｜>




## Author & Maintainer
Yiran Gao gaoyiran525@gmail.com

Kim Hammar <kimham@kth.se>

Tao Li li.tao@cityu.edu.hk

## Copyright and license

[LICENSE](LICENSE.md)

Creative Commons

Copyright (c) 2026 Yiran Gao, Kim Hammar, Tao Li.

