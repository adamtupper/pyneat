# PyNEAT

A Python implementation of the NEAT neuroevolution algorithm.

## 1. Description

A longer description...

## 2. Installation

Installation instructions are as follows.

### 2.1. Common

Network visualisations are created using Graphviz. First install Graphviz
using:

```bash
sudo apt update
sudo apt install graphviz graphviz-dev
```

To install this package, firstly create an isolated conda environment with all
required dependencies using the environment file provided:

```bash
conda env create -f environment.yml
```

Following this, activate the conda environment (`conda activate pyneat`)
and install the remaining dependencies that cannot be installed through conda
using:

```bash
pip install -r requirements.txt
```

### 2.2. Basic Usage

For basic usage, install PyNEAT using:

```bash
python setup.py install
```

PyNEAT can then be imported in your Python programs using:

```python
import pyneat
```

## 2.3. Development

To install PyNEAT for continued development use:

```bash
python setup.py develop
```

This prevents the need to reinstall the package each time after making changes.

## Note

This project has been set up using PyScaffold 3.2.2. For details and usage
information on PyScaffold see https://pyscaffold.org/.
