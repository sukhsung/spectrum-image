# Spectrum-Image: Interactive Data Explorer for Spatial Spectroscopic Dataset
Spectrum-Image is Python packag for spectroscopic data, built with Jupter lab interactivity in mind, with minimal dependencies
Most of code is written with EELS (Electron Energy Loss Spectroscopy) in mind, but is relevant for any highdimensional spectroscopic dataset.

# Relevant techniques:
EELS (Electron Energy Loss Spectroscopy)
RIXS (Resonant Inelastic Xray Scattering)
Raman Spatial Mapping

Check out the notebook example (`example_eels.ipynb`) to get started.

## Installation
`pip install spectrum-image`

## Dependencies
`matplotlib`
`jupyterlab`
`ipympl`
`scipy`
`tqdm`
`numpy`
`lmfit`

## Optional Dependencies for running example notebook
`hyperspy`
`tifffile`

## Acknowledgements
Some functionalities (background subtraction, local background averaging) was copied from eels.py of https://github.com/paradimdata/Cornell_EM_SummerSchool_2021/tree/main/Tutorial%204%20-%20Spectroscopy
