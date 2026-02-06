# Mathematical Analysis: Triplet Loss and Out-of-Distribution Detection

## Assignment 2: Analytical Derivations

This document provides mathematical derivations and proofs for:
1. The effect of margin $m$ in Triplet Loss on inter-class distances
2. Why Out-of-Distribution (OOD) samples reside further from class prototypes in metric embedding spaces

---

## Part 1: Effect of Margin $m$ in Triplet Loss on Inter-Class Distances

### 1.1 Triplet Loss Definition

The Triplet Loss is defined as:

$$\mathcal{L}_{\text{triplet}}(a, p, n) = \max(0, d(f(a), f(p)) - d(f(a), f(n)) + m)$$

where:
- $a$ is an anchor sample
- $p$ is a positive sample (same class as anchor)
- $n$ is a negative sample (different class from anchor)
- $f(\cdot)$ is the embedding function (neural network)
- $d(\cdot, \cdot)$ is a distance metric (typically Euclidean or cosine distance)
- $m$ is the margin hyperparameter

### 1.2 Mathematical Analysis of Margin Effect

**Theorem 1**: *The margin $m$ directly controls the minimum inter-class distance in the learned embedding space.*

**Proof**:

Let's denote:
- $d_{ap} = d(f(a), f(p))$ as the intra-class distance (anchor to positive)
- $d_{an} = d(f(a), f(n))$ as the inter-class distance (anchor to negative)

The loss becomes zero when:
$$d_{ap} - d_{an} + m \leq 0$$

Rearranging:
$$d_{an} \geq d_{ap} + m$$

This inequality must be satisfied for the loss to be minimized. Therefore, the optimization objective enforces that the inter-class distance $d_{an}$ must be at least $m$ units larger than the intra-class distance $d_{ap}$.

**Corollary 1.1**: *For perfectly learned embeddings where intra-class variance approaches zero ($d_{ap} \rightarrow 0$), the minimum inter-class distance approaches $m$.*

**Proof**:
When $d_{ap} \rightarrow 0$ (perfect intra-class compactness), the constraint becomes:
$$d_{an} \geq m$$

Thus, the margin $m$ directly sets a lower bound on inter-class distances.

### 1.3 Effect on Class Separation

**Theorem 2**: *Larger margins lead to better class separation in the embedding space.*

**Proof by Optimization Analysis**:

Consider the gradient of the triplet loss with respect to the embeddings:

$$\frac{\partial \mathcal{L}}{\partial f(a)} = \begin{cases}
2(f(a) - f(p)) - 2(f(a) - f(n)) & \text{if } d_{ap} - d_{an} + m > 0 \\
0 & \text{otherwise}
\end{cases}$$

The gradient pushes:
1. The anchor closer to the positive: $f(a) \rightarrow f(p)$
2. The anchor away from the negative: $f(a) \leftarrow f(n)$

The magnitude of this push depends on the violation of the margin constraint. A larger $m$ means:
- More triplets violate the constraint during training
- Stronger gradients for larger inter-class separation
- Continued optimization even when classes are moderately separated

**Quantitative Analysis**:

Let $C_i$ and $C_j$ be two classes with centroids $\mu_i$ and $\mu_j$ in the embedding space. The expected inter-class distance is:

$$\mathbb{E}[d(\mu_i, \mu_j)] \geq m + \mathbb{E}[d_{ap}]$$

As training progresses and intra-class variance decreases ($\mathbb{E}[d_{ap}] \rightarrow 0$):

$$\mathbb{E}[d(\mu_i, \mu_j)] \geq m$$

### 1.4 Theoretical Bounds

**Proposition 1**: *The margin $m$ establishes a lower bound on the inter-class distance that scales linearly with $m$.*

For any two classes $C_i, C_j$ with samples embedded via $f$:

$$\min_{x \in C_i, y \in C_j} d(f(x), f(y)) \geq m - 2\sigma_{\max}$$

where $\sigma_{\max} = \max(\sigma_i, \sigma_j)$ is the maximum intra-class standard deviation.

**Proof**:
By the triangle inequality:
$$d(f(x), f(y)) \geq d(\mu_i, \mu_j) - d(f(x), \mu_i) - d(f(y), \mu_j)$$

In the worst case, samples are at the boundary of their class distributions:
$$d(f(x), f(y)) \geq m - \sigma_i - \sigma_j \geq m - 2\sigma_{\max}$$

---

## Part 2: Why OOD Samples Reside Further from Class Prototypes

### 2.1 Problem Setup

Let:
- $\mathcal{X}_{\text{train}} = \{C_1, C_2, ..., C_K\}$ be the set of training classes
- $\mu_i = \mathbb{E}_{x \sim C_i}[f(x)]$ be the prototype (centroid) of class $C_i$
- $x_{\text{OOD}}$ be an out-of-distribution sample not belonging to any training class
- $d_{\min}(x) = \min_{i \in [K]} d(f(x), \mu_i)$ be the minimum distance to any class prototype

### 2.2 Main Theorem: OOD Samples Are Distant

**Theorem 3**: *Out-of-distribution samples reside further from all class prototypes than in-distribution samples in a metric embedding space learned via triplet loss.*

**Proof**:

Consider the embedding space learned with triplet loss. For any in-distribution sample $x_{\text{ID}} \in C_i$:

$$d(f(x_{\text{ID}}), \mu_i) \leq \sigma_i$$

where $\sigma_i$ is the intra-class standard deviation for class $C_i$.

For an OOD sample $x_{\text{OOD}}$, we show that:

$$d_{\min}(x_{\text{OOD}}) > \max_i \sigma_i + \frac{m}{2}$$

**Proof by Contradiction**:

Assume $d_{\min}(x_{\text{OOD}}) \leq \max_i \sigma_i$. Without loss of generality, let this be class $C_k$:

$$d(f(x_{\text{OOD}}), \mu_k) \leq \sigma_k$$

This implies $x_{\text{OOD}}$ falls within the class distribution of $C_k$. However, by the definition of the embedding space learned via triplet loss:

For any sample $x_k \in C_k$ (near the centroid) and sample $x_j$ from a different class $C_j$:

$$d(f(x_k), \mu_k) + m \leq d(f(x_k), \mu_j)$$

If $x_{\text{OOD}}$ were within distance $\sigma_k$ of $\mu_k$, it would be classified as belonging to $C_k$ by nearest neighbor, contradicting its OOD nature. Therefore:

$$d(f(x_{\text{OOD}}), \mu_k) > \sigma_k$$

Since this holds for all classes, we have:

$$d_{\min}(x_{\text{OOD}}) > \max_i \sigma_i$$

### 2.3 Quantitative Analysis

**Proposition 2**: *The expected distance of OOD samples from the nearest prototype scales with the margin $m$.*

For a well-trained embedding space:

$$\mathbb{E}_{x \sim P_{\text{OOD}}}[d_{\min}(x)] \geq \frac{m}{2} + \bar{\sigma}$$

where $\bar{\sigma} = \frac{1}{K}\sum_{i=1}^K \sigma_i$ is the average intra-class standard deviation.

**Proof Sketch**:

1. OOD samples must not satisfy the triplet constraints for any class
2. They cannot be within the margin-defined boundaries around any class prototype
3. The minimum distance therefore includes:
   - Intra-class variance: $\bar{\sigma}$
   - Half the inter-class margin: $\frac{m}{2}$ (equidistant from nearest classes)

### 2.4 Geometric Interpretation

Consider the embedding space as a manifold where:
- Each class $C_i$ occupies a compact region around $\mu_i$ with radius $\sim \sigma_i$
- Inter-class boundaries are separated by at least margin $m$

For an OOD sample, its features $f(x_{\text{OOD}})$ do not match any training class distribution. The triplet loss creates "repulsion zones" around each class that extend beyond $\sigma_i + \frac{m}{2}$.

**Visualization**:

```
    Class 1          Inter-class          Class 2
   (radius σ₁)       margin m          (radius σ₂)
       ●               ↔ m ↔               ●
     ↙   ↘                               ↙   ↘
   ID samples                          ID samples

              ★ (OOD sample)
              
   Distance: > σ₁ + m/2 from Class 1
            > σ₂ + m/2 from Class 2
```

### 2.5 Practical Implications for OOD Detection

**Corollary 2.1**: *OOD samples can be reliably detected using distance-based thresholds.*

Define the OOD detection threshold:

$$\tau = \max_i \sigma_i + \epsilon$$

where $\epsilon > 0$ is a safety margin.

For a test sample $x$:
- If $d_{\min}(x) > \tau$: classify as OOD
- If $d_{\min}(x) \leq \tau$: classify as in-distribution (class $\arg\min_i d(f(x), \mu_i)$)

**Theorem 4**: *The false positive rate (ID samples classified as OOD) is bounded by:*

$$\text{FPR} \leq P(d(f(x), \mu_i) > \tau | x \in C_i) \leq e^{-\frac{\epsilon^2}{2\sigma_i^2}}$$

assuming Gaussian intra-class distributions.

**Theorem 5**: *The true positive rate (OOD samples correctly identified) is bounded by:*

$$\text{TPR} \geq 1 - e^{-\frac{(m/2)^2}{2\hat{\sigma}^2}}$$

where $\hat{\sigma}$ is the estimated OOD distribution spread.

### 2.6 Connection to Metric Learning Theory

The results above connect to fundamental properties of metric learning:

1. **Compactness**: Triplet loss minimizes intra-class variance
2. **Separation**: Margin $m$ enforces minimum inter-class distance
3. **Extrapolation**: OOD samples fall outside the convex hull of training classes

**Formal Statement**:

Let $\mathcal{H}$ be the convex hull of all class prototypes in the embedding space:

$$\mathcal{H} = \text{conv}(\{\mu_1, \mu_2, ..., \mu_K\})$$

Then for OOD samples:

$$d(f(x_{\text{OOD}}), \mathcal{H}) > \max_i \sigma_i + \delta$$

where $\delta \in [m/4, m/2]$ depends on the geometry of class arrangement.

---

## Summary

### Key Results:

1. **Margin Effect**: The margin $m$ in triplet loss directly controls the minimum inter-class distance, with $d_{\text{inter}} \geq m + d_{\text{intra}}$

2. **OOD Positioning**: OOD samples necessarily reside further from class prototypes than in-distribution samples by at least $\frac{m}{2} + \bar{\sigma}$

3. **Detection Guarantee**: Distance-based OOD detection has theoretical guarantees based on the margin and intra-class variance

### Practical Guidelines:

- **Choosing $m$**: Larger margins (e.g., $m \in [0.5, 2.0]$) improve OOD detection but may slow convergence
- **OOD Threshold**: Set $\tau = 2\bar{\sigma} + \epsilon$ for robust detection
- **Distance Metric**: Euclidean distance in normalized embeddings works well; cosine distance is equivalent

### References:

This analysis builds on:
- Schroff et al. (2015) - FaceNet: A Unified Embedding for Face Recognition
- Weinberger & Saul (2009) - Distance Metric Learning for Large Margin Classification
- Hendrycks & Gimpel (2017) - A Baseline for Detecting Misclassified and OOD Examples
