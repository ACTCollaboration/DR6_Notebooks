# DR6_Notebooks
These notebooks have been developed by the Atacama Cosmology Telescope collaboration to demonstrate how one can use data products from ACT's Data Release 6 (DR6) to perform common analyses. If you make use of these notebooks as a development aid in a publication-worthy analysis, please follow instructions in the [attribution section](#attribution) below.

The data products used in these notebooks are publicly available on [LAMBDA](https://lambda.gsfc.nasa.gov/product/act/actadv_prod_table.html) and at [NERSC](https://crd.lbl.gov/divisions/scidata/c3/c3-research/cosmic-microwave-background/cmb-data-at-nersc/). See the ACT Data Products [webpage](https://act.princeton.edu/act-dr6-data-products) for up-to-date information. These notebooks make use of an alternative server located at Princeton University (`phy-act1`) to host select data products. Products on `phy-act1` are for demsontration purposes only; please use one of the official data hosting locations for publication-worthy analyses.  

Each notebook has been written to work in Google colab and can be accessed either by clicking the colab button in the notebook file above, or using the links below. In case the notebook is too large to render in your browser, simply access colab by changing `github` in the notebook url to `githubtocolab` and refresh. Note: this will open a frozen, read-only version. It will still run, but if you want to edit or save your work, you will need to first save a copy of the colab notebook in your personal Google drive. If you want to adapt these notebooks to run on your laptop or on a cluster, you can also clone the repository! However, you may need to manage the notebook dependencies manually.

## Notebooks:

### Maps and TOD Notebooks

- [ACT DR6 map manipulation and showcase](https://github.com/ACTCollaboration/DR6_Notebooks/blob/main/ACT_DR6_maps.ipynb)

- [Working with ``depth1" maps for transients](https://github.com/ACTCollaboration/DR6_Notebooks/blob/main/ACT_DR6_depth1_maps.ipynb)

- [Understanding the TOD processing](https://github.com/ACTCollaboration/DR6_Notebooks/blob/main/ACT_DR6_detector_cuts.ipynb)

- [Classifying TOD events](https://github.com/ACTCollaboration/DR6_Notebooks/blob/main/ACT_DR6_detector_glitch_classification.ipynb)

### Power Spectrum and Likelihood Notebooks

- [Working with the ACT DR6 power spectra and likelihoods](https://github.com/ACTCollaboration/DR6_Notebooks/blob/main/ACT_DR6_ps_likelihood.ipynb)

### Lensing Notebooks

- [CIB correlations with ACT CMB Lensing Maps](https://github.com/ACTCollaboration/DR6_Notebooks/blob/main/ACT_DR6_lensing_CIB_correlation.ipynb)

- [Viewing the lensing power spectrum](https://github.com/ACTCollaboration/DR6_Notebooks/blob/main/ACT_DR6_lensing_power.ipynb)

- [Deriving the lensing transfer function](https://github.com/ACTCollaboration/DR6_Notebooks/blob/main/ACT_DR6_lensing_transfer_function.ipynb)

- [The lensing likelihood with cosmopower](https://github.com/ACTCollaboration/DR6_Notebooks/blob/main/ACT_DR6_lensing_likelihood_with_cosmopower.ipynb)

### ILC Maps Notebooks

- [Investigating CIB deprojection in the ACT ymap](https://github.com/ACTCollaboration/DR6_Notebooks/blob/main/ACT_DR6_ymap_CIB.ipynb)

- [Stacking on tSZ clusters in the ACT ymap](https://github.com/ACTCollaboration/DR6_Notebooks/blob/main/ACT_DR6_ymap_stacking.ipynb)

## Attribution:
If you use the data products referenced in these notebooks please cite the relevant papers from ACT and acknowledge this repository (https://github.com/ACTCollaboration/DR6_Notebooks).

### Maps and TOD Notebooks
- [Naess et. al. (2025)](https://arxiv.org/abs/2503.14451)
- [Nerval et. al. (2025)](https://arxiv.org/abs/2503.10798)

### Power Spectrum and Likelihood Notebooks
- [Louis et. al. (2025)](https://arxiv.org/abs/2503.14452)
- [Calabrese et. al. (2025)](https://arxiv.org/abs/2503.14454)

### Lensing Notebooks
- [Qu et al. (2023)](https://arxiv.org/abs/2304.05202)
- [Madhavacheril et al. (2023)](https://arxiv.org/abs/2304.05203)
- [MacCrann et al. (2023)](https://arxiv.org/abs/2304.05196)
- [Farren et al. (2023)](https://arxiv.org/abs/2309.05659)

### ILC Maps Notebooks
- [Coulton et al. (2024)](https://arxiv.org/abs/2307.01258)
  


