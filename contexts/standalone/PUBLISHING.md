# Publishing

The PyPI distribution is `adaptive-terrain-diffusion`; users import it as
`terrain_diffusion` and install it with:

```bash
pip install adaptive-terrain-diffusion
```

## First release

1. Create a PyPI account with two-factor authentication.
2. In PyPI's **Publishing** settings, add a pending GitHub publisher:
   - Owner: `youwyu`
   - Repository: `Adaptive-Diffusion-Terrain`
   - Workflow: `publish-standalone.yml`
   - Environment: `pypi`
3. Create the `pypi` environment in the GitHub repository and require approval
   for deployment.

## Release

1. Set a new version in `pyproject.toml`. PyPI versions cannot be overwritten.
2. Validate locally:

   ```bash
   uv pip install ".[dev]" --no-cache
   python -m build
   python -m twine check dist/*
   ```

3. Commit the version, create its GitHub tag, and publish a GitHub Release. The
   release workflow builds the wheel and source archive, then publishes them to
   PyPI through short-lived trusted-publisher credentials.

The ONNX checkpoint is included in both distributions. Before releasing, keep
each artifact below PyPI's configured project upload limit.
