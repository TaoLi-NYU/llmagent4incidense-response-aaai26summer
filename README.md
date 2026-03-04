# In-Context Autonomous Network Incident Response: An End-to-End Large Language Model Agent Approach

This repository contains the artifacts related to the paper "In-Context Autonomous Network Incident Response: An End-to-End Large Language Model Agent Approach", which is accepted to AAAI 2026 Summer Symposium Series. we propose to leverage large language models’ (LLM) pre-trained security knowledge and in-context learning to create an end-to-end agentic solution for incident response planning. Specifically, our agent integrates four functionalities, perception,reasoning, planning, and action, into one lightweight LLM(14b model). Through fine-tuning and chain-of-thought reasoning, our LLM agent is capable of processing system logs and inferring the underlying network state (perception), updating its conjecture of attack models (reasoning), simulating consequences under different response strategies (planning), and generating an effective response (action).

<img width="990" height="330" alt="image" src="https://github.com/user-attachments/assets/b9b03469-6870-4c12-90fd-2fbb87486c82" />



## Requirements

- Python 3.8+
- `torch`
- `transformers`
- `peft`
- `bitsandbytes`
- `accelerate`

## Development Requirements

- Python 3.8+
- `flake8` (for linting)
- `flake8-rst-docstrings` (for linting docstrings)
- `tox` (for automated testing)
- `pytest` (for unit tests)
- `pytest-cov` (for unit test coverage)
- `mypy` (for static typing)
- `mypy-extensions` (for static typing)
- `mypy-protobuf` (for static typing)
- `types-PyYaml` (for static typing)
- `types-paramiko` (for static typing)
- `types-protobuf` (for static typing)
- `types-requests` (for static typing)
- `types-urllib3` (for static typing)
- `sphinx` (for API documentation)
- `sphinxcontrib-napoleon` (for API documentation)
- `sphinx-rtd-theme` (for API documentation)
- `pytest-mock` (for mocking tests)
- `pytest-grpc` (for grpc tests)

## Installation

```bash
# install from pip
pip install llm_recovery==<version>
# local install from source
$ pip install -e llm_recovery
# or (equivalently):
make install
# force upgrade deps
$ pip install -e llm_recovery --upgrade
# git clone and install from source
git clone https://github.com/Limmen/llm_recovery
cd llm_recovery
pip3 install -e .
# Install development dependencies
$ pip install -r requirements_dev.txt
```

### Development tools

Install all development tools at once:
```bash
make install_dev
```
or
```bash
pip install -r requirements_dev.txt
```

## Static code analysis

To run the Python linter, execute the following command:
```
flake8 .
# or (equivalently):
make lint
```

To run the mypy type checker, execute the following command:
```
mypy .
# or (equivalently):
make types
```

## Unit tests

To run all the unit tests, execute the following command:
```
pytest
# or (equivalently):
make unit_tests
```

To run tests of a specific test suite, execute the following command:
```
pytest -k "ClassName"
```

To generate a coverage report, execute the following command:
```
pytest --cov=llm_recovery
```

## Run tests and code analysis in different python environments

To run tests and code analysis in different python environments, execute the following command:

```bash
tox
# or (equivalently):
make tests
```

## Create a new release and publish to PyPi

First build the package by executing:
```bash
python -m build
# or (equivalently)
make build
```
After running the command above, the built package is available at `./dist`.

Push the built package to PyPi by running:
```bash
python -m twine upload dist/*
# or (equivalently)
make push
```

To run all commands for the release at once, execute:
```bash
make release
```

## Author & Maintainer

Kim Hammar <kimham@kth.se>

## Copyright and license

[LICENSE](LICENSE.md)

Creative Commons

(C) 2025, Kim Hammar

