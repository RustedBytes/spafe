# Python Examples

These examples use the PyO3 extension module and matplotlib, then write CSV/PNG outputs to
`target/python-examples`.

Install the extension in a virtual environment first:

```bash
python -m pip install maturin matplotlib
maturin develop
```

Run examples from the repository root:

```bash
python examples/python/features.py
python examples/python/fbanks.py
python examples/python/pitch.py
python examples/python/cochleagram.py
```

Generated PNG plots can be opened directly from `target/python-examples`.
