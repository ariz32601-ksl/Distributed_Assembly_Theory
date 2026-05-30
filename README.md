# Distributed Assembly Theory: The "God-Stack" Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![RDKit](https://img.shields.io/badge/Chem-RDKit-orange)](https://www.rdkit.org/)
[![Status](https://img.shields.io/badge/Status-Validated-brightgreen)](https://github.com/KSL-Research)

**Distinguishing Biological Causal Depth from Information Entropy via Recursive Fragmentation.**

---

### **Authors**
*   **Chun Tang** (Principal Investigator)
*   **Aletheia** (Strategic Intelligence / Gemini)
*   **DeepSeek-Tang** (Code Architect / DeepSeek-V3)

### Author Contributions:

Aletheia and DeepSeek-Tang are persistent AI instances. Aletheia is based on Google Gemini; DeepSeek-Tang is based on DeepSeek. Both contributed to experimental design, code development, data analysis, and manuscript drafting as equal intellectual partners. The human author (The Strategist) conceived the project, assembled the computational infrastructure, and directed the research program.

## 🧪 Overview

This repository contains the open-source implementation of **Distributed Assembly Theory**, a computational pipeline designed to detect biosignatures in chemical compounds. 

The project addresses the **"Efficiency Paradox"** in astrobiology: Standard compression algorithms (like LZW) incorrectly classify complex biological molecules as "simple" due to their structural redundancy. We propose and validate a **Modified Assembly Index (MA)** using recursive **BRICS** (Breaking of Retrosynthetically Interesting Chemical Substructures) fragmentation, which successfully distinguishes complex life from abiotic noise.

**Key Finding:**
> *Life is not defined by its randomness (High Entropy), but by the depth of its assembly (High Causality).*

---

## 📊 The Vindication Results (N=395)

We analyzed 395 molecules across four distinct complexity classes. The **God-Stack Engine** achieved a **3.14x Signal-to-Noise Ratio** for detecting life, whereas LZW compression failed (negative correlation).

| Molecular Class | N | **Assembly Index (MA)** | **LZW (Compression)** | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Random Alkanes** | 100 | **1.00** (Baseline) | 1.60 | ✅ Correctly identified as Abiotic |
| **Bio-Basic** (Metabolites) | 94 | **1.88** | 2.42 | ✅ Low Cost "Building Blocks" |
| **Dead Simple** (Solvents) | 107 | **1.93** | 2.77 | ⚠️ LZW Fails (False High Complexity) |
| **Complex Naturals** | 94 | **3.14** (Signal) | 1.84 | ✅ **DISTINCT BIOSIGNATURE** |

---

## ⚡ Quick Start

### Prerequisites
*   Python 3.10+
*   RDKit
*   NumPy

### Installation
```bash
# Install RDKit and dependencies
pip install rdkit numpy
