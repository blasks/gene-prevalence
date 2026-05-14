Methodology to estimate gene family prevalence in microbe populations using incomplete genomes (e.g. from single-cell amplified genomes or metagenome assembled genomes).

Usage notes
-----------

`BernoulliPrevalenceModel.fit()` remains the single-interval entrypoint:

```python
result = model.fit(
    estimator="mle",
    interval_method="likelihood_ratio",
    alpha=0.05,
    clip=True,
)
lower, upper = result.confidence_interval()
```

To compute multiple confidence intervals from one fitted prevalence estimate,
use `BernoulliPrevalenceModel.fit_multiple_ci()`:

```python
result = model.fit_multiple_ci(
    estimator="mle",
    intervals=[
        {"method": "likelihood_ratio", "alpha": 0.05},
        {"method": "likelihood_ratio", "alpha": 0.10},
    ],
    clip=True,
)

intervals = result.get_confidence_intervals()
lower_95, upper_95 = result.confidence_interval(
    method="likelihood_ratio",
    alpha=0.05,
)
```

When exactly one confidence interval was computed, `result.confidence_interval()`
keeps its legacy zero-argument behavior. When multiple intervals are present,
call `confidence_interval(method=..., alpha=...)` to select one.

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
