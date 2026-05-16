Methodology to estimate gene family prevalence in microbe populations using incomplete genomes (e.g. from single-cell amplified genomes or metagenome assembled genomes).

Usage notes
-----------

Use `BernoulliPrevalenceModel.fit()` to compute one prevalence estimate and,
optionally, one confidence interval:

```python
result = model.fit(
    estimator="mle",
    interval_method="likelihood_ratio",
    alpha=0.05,
    clip=True,
)
lower, upper = result.confidence_interval()
```

API docs
--------

The project includes a minimal Sphinx configuration that renders Python
docstrings, including ``.. math::`` blocks, as HTML.

Install the dev dependencies and build the docs with:

```bash
poetry install --with dev
poetry run sphinx-build -b html docs docs/_build/html
```

Open ``docs/_build/html/index.html`` in a browser to view the rendered API
reference.
