## Contribution Guidelines
1. Fork the repository
2. Create a new branch
3. Make your changes and commit them
4. Push your changes to your fork
5. Create a pull request
6. Check out the [issues](https://github.com/NanoNets/docext/issues) and pick one to work on

## Development installation
```bash
# create the virtual environment
uv venv --python=3.11

# activate the virtual environment
source .venv/bin/activate

# install the dependencies
pip install -e .

# pre-commit
uv pip install pre-commit
pre-commit install
```
