# Objectness-Filter
Implementation of class-agnostic objectness measures to prune 99% of image search space via Bayesian fusion of saliency, edge density, and straddleness cues.

# Beyond-Brute-Force: Generic Objectness Estimation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

### **"Do we really need a GPU to tell us that a flat, blue sky isn’t a car?"**

In modern Computer Vision, we often prioritize classification accuracy while ignoring the computational bottleneck of region proposals. This repository implements the foundational research from **"What is an object?" (Alexe et al., CVPR 2010)** to prune the search space of an image by 99% using low-level, class-agnostic cues.

---

## 📸 The Core Concept

A standard image contains over **100,000 potential bounding boxes**. Passing all of them through a Deep Neural Network is computationally expensive. By measuring **Objectness**, we identify "Significant Points" of interest before any classification happens.



### **Comparative Analysis**
| **The Noise** | **The Signal** |
| :--- | :--- |
| ![Proposals Placeholder](https://via.placeholder.com/400x300?text=2000+Raw+Proposals) | ![Results Placeholder](https://via.placeholder.com/400x300?text=Top+5+Objectness+Results) |
| **2000 Sampled Windows:** A dense mesh of candidate boxes across multiple scales. | **Top 5 Candidates:** Bayesian fusion identifies the most likely "things." |

---

## 🛠 Technical Implementation

This project uses a **Bayesian Framework** to fuse independent image cues into a single probability score.

### **The Objectness Cues**
* **Multi-Scale Saliency (MS):** Implements Spectral Residual analysis to find visual "surprises" that contrast with global image statistics.
* **Edge Density (ED):** Measures the concentration of edges (Canny) within a window. Objects typically have well-defined, localized boundaries.
* **Straddleness (SS):** Analyzes superpixels (Felzenszwalb) to see if they "straddle" the window boundary. High straddleness suggests a poor fit for an object.



### **Efficiency Engine**
To score thousands of windows in real-time, the pipeline utilizes **Summed Area Tables (Integral Images)**. This reduces the complexity of calculating window sums from $O(N^2)$ to $O(1)$ per window, regardless of size.



---

## Getting Started

### **Installation**
```bash
git clone [https://github.com/](https://github.com/)[YourUsername]/Beyond-Brute-Force.git
cd Beyond-Brute-Force
pip install -r requirements.txt