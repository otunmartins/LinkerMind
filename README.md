# LinkerMind: An Interpretable, Mechanism-Informed Deep Learning Framework for ADC Linker Design

[![GitHub release](https://img.shields.io/github/v/release/otunmartins/LinkerMind)](https://github.com/otunmartins/LinkerMind/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

LinkerMind is a comprehensive computational platform for the *de novo* design of Antibody-Drug Conjugate (ADC) linkers. This framework integrates mechanistic chemical intelligence with deep learning to generate novel, optimized linker candidates with balanced stability and cleavability profiles.

> **Citation**: If you use LinkerMind in your research, please cite our paper:  
> *"LinkerMind: An Interpretable, Mechanism-Informed Deep Learning Framework for the De Novo Design of ADC Linkers"*

## Features

- **Multi-modal deep learning** combining structural fingerprints with mechanistic descriptors
- **Generative AI pipeline** for *de novo* linker design using Transformer architectures
- **Scaffold-split validation** for rigorous performance evaluation
- **Mechanism-informed features** capturing cleavable motifs (peptide, disulfide, ester bonds)
- **Multi-objective optimization** for balanced linker properties (stability, reactivity, synthetic accessibility)
- **Clinical/ADME profiling** with *in silico* toxicity screening
- **Interpretable predictions** with feature importance analysis

## Installation

### Prerequisites
- Python 3.8 or higher
- RDKit (for cheminformatics)
- PyTorch (for deep learning)

### Quick Install

```bash
# Clone repository
git clone https://github.com/otunmartins/LinkerMind.git
cd LinkerMind

# Install dependencies
pip install -r requirements.txt