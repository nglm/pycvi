**Important note:** As of now (June 2026), the latest version of `scikit-learn-extra` (0.3.0) is not compatible with `numpy>= 2.0.0`. To avoid adding restrictions on the version of `numpy` for users who do not want to run the example scripts that require `scikit-learn-extra`, no explicit version of `numpy` is specified in the `pyproject.toml` file.

This means that if you want to run example scripts, you'll have to make sure that you are using a version of `numpy` that is strictly below 2.0.0 in your environment.
