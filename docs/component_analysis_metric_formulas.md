# Component Analysis Metric Formula Notes

## 1. Split-Half Analysis Stability Metrics

### 1.1 Loading Consistency (Ex_TCC_1 / Em_TCC_1)

This metric evaluates loading stability in split-half models using cosine similarity (TCC):

$$
\text{TCC}(A, B) = \frac{A \cdot B}{\|A\| \cdot \|B\|}
$$

Parameter definitions:
- $A$, $B$: Two loading matrices (for example, Ex or Em), with shape $n \times r$
- $\cdot$: Matrix inner product
- $\|\cdot\|$: Frobenius norm

---

### 1.2 Factor Similarity

This metric is the minimum cosine similarity between split-half model factors and full-model factors:

$$
\text{Factor Similarity} = \min_{i=1}^{8} \text{TCC}(F^{(i)}, F_{\text{full}})
$$

Parameter definitions:
- $F^{(i)}$: Factor loading matrix from the $i$-th split-half model
- $F_{\text{full}}$: Factor matrix from the full model

---

### 1.3 Core Consistency

Core consistency evaluates whether the model has a strong trilinear structure:
- Core Consistency $\approx 100$: Excellent structure
- $50 \sim 90$: Good structure
- $< 50$: Weak structure, possible overfitting

Calculation method:
- Computed by `tlviz.model_evaluation.core_consistency(model)`

---

### 1.4 Explained Rate

This metric measures how well the model fits the original tensor:

$$
\text{Explained Rate} = 1 - \frac{\|X - \hat{X}\|_F^2}{\|X\|_F^2}
$$

Parameter definitions:
- $X$: Original fluorescence data tensor with shape $(I, J, K)$
- $\hat{X}$: Reconstructed tensor from the model
- $\|\cdot\|_F$: Frobenius norm

---

### 1.5 Validated

Validation status is defined by the rule below:

$$
\text{Validated} =
\begin{cases}
\mathrm{Yes}, & \text{if } \text{Factor Similarity} \geq 0.95 \\
\mathrm{No}, & \text{otherwise}
\end{cases}
$$

Interpretation:
- If Factor Similarity is at least $0.95$, the model is considered to have high stability.
