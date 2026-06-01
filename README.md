# Factorized-Neural-Operators (FaNO)

Official implementation of **Factorized Neural Operators (FaNO)**.

FaNO is a neural operator framework for modeling heterogeneous physical systems by decomposing operator responses into:

- **Dynamic response**: captures rapidly evolving and state-dependent physical dynamics.
- **Persistent response**: captures coherent and slowly evolving structures.

The framework provides unified implementations across Euclidean, spherical, and manifold domains.

> The manuscript is currently under review. Citation information will be updated after the preprint becomes available.

---
## Repository Structure

Each folder contains the implementation, training scripts, and evaluation protocols for the corresponding experiments.

Example structure:

```text
FaNO/
│
├── SWE/
│   ├── models/
│   ├── train_swe.py
│   ├── test_swe.py
│   ├── requirements.txt
│   └── README.md
│
├── NS/
│   ├── train_ns.py
│   ├── test_ns.py
│   ├── ...
│   └── README.md
│
├── ...
│
└── README.md
```

Additional experiments follow a similar organization.

---

## Datasets and Benchmarks

All datasets and benchmark implementations used in this work are publicly available from their official repositories.

| Benchmark | Source |
|---|---|
| WeatherBench | https://github.com/pangeo-data/WeatherBench |
| Shallow Water Equation | https://github.com/NVIDIA/torch-harmonics |
| Navier--Stokes & Darcy Flow | https://github.com/li-Pingan/fourier-neural-operator |
| Cylinder Flow | https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets |
| Helmholtz Equation | https://github.com/ShashankSubramanian/neuraloperators-TL-scaling |
| RNA & Human Mesh | https://github.com/nmwsharp/diffusion-net |
| Spherical MNIST | https://github.com/jonkhler/s2cnn |

We thank the authors of these datasets and benchmarks for making their resources publicly available.

---

## Citation

The manuscript is currently under review.

Citation information will be updated after the preprint becomes publicly available.
