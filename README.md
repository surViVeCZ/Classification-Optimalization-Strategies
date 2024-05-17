# Master's Thesis: Optimization of Classification Models for Malicious Domain Detection

**Author:** Bc. Petr Pouč  
**Date:** April 4, 2024  
**School:** Brno University of Technology (BUT)


# Manual

This readme contains a full guide on installing and running the software created for this master's thesis. The project utilizes Poetry for dependency management and packaging to ensure a consistent development environment.

## Software Requirements

The software is developed in Python and uses Poetry as a dependency management tool. Before proceeding with the installation, ensure that the following requirements are met:

- Python 3.10 or higher
- Poetry for Python

## Installing Poetry

Poetry is a tool for dependency management and packaging in Python, allowing for reproducible builds and straightforward dependency resolution. Once Poetry is installed, you can set up the project by following these steps:

1. Clone the project repository from its source. Replace `<url-to-repository>` with the actual URL of the repository.
2. Navigate into the project directory:
3. Install all dependencies managed by Poetry:

This command reads the `pyproject.toml` file and installs all necessary Python packages into a virtual environment specifically for this project.

## Running the Software

To run the software, ensure that you are within the project's virtual environment. If not already activated, you can activate the Poetry-managed virtual environment by using:

After activating the virtual environment, you can run the software according to the project's documentation or as follows:
```
python3 <path/to/script.py>
```

### Preprocess CLI

File called `cli.py` is the only file with additional arguments. It can be run as:
```plaintext
python3 cli.py -eda --model <cnn|xgboost|adaboost|svm> [--scaling]

python3 cli.py -eda --model cnn --scaling
```

## Additional Notes

- The `pyproject.toml` file in the project directory includes all the necessary configurations and dependencies for the project.
- Always ensure that you are operating within the virtual environment to avoid conflicts with other Python projects or system-wide packages.
- For detailed documentation on Poetry, visit the official Poetry documentation website.
- If you somehow run into a missing dependency error, you can easily add it to the `pyproject.toml` file and run: `poetry add <package-name>`.
