# MPACTPy

## Table of Contents
1. [Installation Instructions](#installation-instructions)
2. [Developer Tools](#developer-tools)

## Installation Instructions
Installing MPACTPy using the following instructions will install the package along with the required dependencies.

### End User Installation 
```bash
python -m pip install .
```
### Developer Installation
```bash
python -m pip install -e .[dev]
```
### Testing Installation (Primarily for CI)
```bash
python -m pip install .[test]
```

## Developer Tools
The configuration settings for the developer tools can be found in `path/to/MPACTPy/pyproject.toml`.

### Testing Python code
Execute this line from the `path/to/MPACTPy` directory to run the code's testing
```bash
pytest .
```

### Linting Python code with pylint
Execute this line from the `path/to/MPACTPy` directory to lint the code with [pylint](https://pypi.org/project/pylint/):
```bash
pylint ./mpactpy
```

### Formatting code with black
Execute this line from the `path/to/MPACTPy` directory to automatically format the code to PEP8 standard using [black](https://pypi.org/project/black/):
```bash
black ./mpactpy
```


