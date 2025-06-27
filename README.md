# BrainCAP

The analysis of moment-to-moment changes in co-activation patterns (CAPs) in functional MRI (fMRI) has been useful for studying dynamic properties of neural activity. This method is based on clustering fMRI time-frames into several recurrent spatial patterns within and across subjects. Studies have also focused on quantifying properties of the temporal organization of CAPs, such as fractional occupancy and dwell time. The analyses of co-activations are computationally intensive, requiring the clustering of high-dimensional data concatenated over subjects. Further, while a variety of analytic choices are involved in studying CAPs, the field lacks a unified open-source platform to allow a robust feature selection required for reproducible mappings of brain and behavioral measurements. We are currently developing **BrainCAP**, an open-source Python-based toolkit for quantifying CAPs from fMRI data in cross-sectional and longitudinal studies. 

This repository serves as the `develop` branch for the ongoing development and enhancement of BrainCAP. 

See `brainCAP/examples` for example code.

To clone the Anaconda environment, use 
`conda env create -f environment_linux.yml`


---

## Important Notes
1. The official release of BrainCAP has not been announced yet. The developer team is working on the first release of BrainCAP.
2. The current version of this repository is specifically tailored for research environments with access to Yale University’s High Performance Computing (HPC) cluster. Future versions will allow the use of various job schedulers, in addition to Slurm.

---

## Branches

### `main`
The `main` branch contains the latest developments and optimizations for BrainCAP. These codes are optimized for local use on the Yale University High-Performance Computing (HPC) cluster. If you are looking to reproduce the data and results from Lee et al. (2024), please refer to the archived version on Zenodo linked above.

### `develop`
The `develop` branch focuses on building an open-source software toolkit for BrainCAP. We aim to release the first version of this open-source toolkit by the end of **2025**. Contributions, feedback, and collaboration are welcome to help shape the future of BrainCAP.

---

## Citation
If you use BrainCAP in your research, please cite:
> Kangjoo Lee, Jie Lisa Ji, Clara Fonteneau, Lucie Berkovitch, Masih Rahmati, Lining Pan, Grega Repovš, John H. Krystal, John D. Murray, and Alan Anticevic, Human brain state dynamics are highly reproducible and associated with neural and behavioral features, PLOS Biology 22(9): e3002808 (2024)

The specific version of the code used for Lee et al. (2024) is archived and available at **[Zenodo](https://zenodo.org/records/13251563)**.

---

## Maintainers

BrainCAP is currently maintained by:

- **Kangjoo Lee, PhD**  
  Email: [kangjoo.lee@yale.edu](mailto:kangjoo.lee@yale.edu)

- **Samuel Brege, Postgraduate Associate**  
  Email: [samuel.brege@yale.edu](mailto:samuel.brege@yale.edu)

For inquiries, questions, or collaborations, please contact either maintainer.
