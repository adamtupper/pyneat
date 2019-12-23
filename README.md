# Custom-NEAT

A customised version of NEAT-Python.

## 1. Description

A longer description...

## 2. Installation

Installation instructions are as follows.

### 2.1. Common

To install this package, firstly create an isolated conda environment with all
required dependencies using the environment file provided:

```bash
conda env create -f environment.yml
```

Following this, activate the conda environment (`conda activate custom_neat`)
and install the remaining dependencies that cannot be installed through conda
using:

```bash
pip install -r requirements.txt
```

### 2.2. Basic Usage

For basic usage, install Custom-NEAT using:

```bash
python setup.py install
```

Custom-NEAT can then be imported in your Python programs using:

```python
import custom_neat
```

## 2.3. Development

To install Custom-NEAT for continued development use:

```bash
python setup.py develop
```

This prevents the need to reinstall the package each time after making changes.

## Note

This project has been set up using PyScaffold 3.2.2. For details and usage
information on PyScaffold see https://pyscaffold.org/.
