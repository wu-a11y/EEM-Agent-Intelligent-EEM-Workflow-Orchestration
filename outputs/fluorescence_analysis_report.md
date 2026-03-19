# Fluorescence Component Analysis Report

## 1. Component Analysis Metric Formula Notes

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

---

## 2. analysis_result_2

| Fluorescence component (sheet) | Sample count with q > 0.7 | Best sample ID (global q) | Max global q | Best standard component (global q) | Global-Reference | Global-Sources | Global-Ecozones | Matched category | Best sample ID (category q) | Max category q | Best standard component (category q) | Category-Reference | Category-Sources | Category-Ecozones |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | 132 | 9189 | 0.9352 | C4 | Liu C, Shen Q, Gu X, et al. Burial or mineralization: Origins and fates of organic matter in the water–suspended particulate matter–sediment of macrophyte- and algae-dominated areas in Lake Taihu[J]. Water Research, 2023, 243: 120414. | lake water, suspended particulate matter, and sediment | Changjiang Plain evergreen forests | 腐殖质类（Humic-like） | 9189 | 0.9352 | C4 | Liu C, Shen Q, Gu X, et al. Burial or mineralization: Origins and fates of organic matter in the water–suspended particulate matter–sediment of macrophyte- and algae-dominated areas in Lake Taihu[J]. Water Research, 2023, 243: 120414. | lake water, suspended particulate matter, and sediment | Changjiang Plain evergreen forests |
| C2 | 258 | 62 | 0.991 | C5 | Wünsch UJ, Murphy KR and Stedmon CA (2015). Fluorescence quantum yields of natural organic matter and organic compounds: Implications for the fluorescence-based interpretation of organic matter composition. Front. Mar. Sci. 2:98. | Phenylalanine;Salicylic acid;Vanillic acid;Syringic acid;Tryptophan;Tyrosine;Coumarin | - | 其它 | 62 | 0.991 | C5 | Wünsch UJ, Murphy KR and Stedmon CA (2015). Fluorescence quantum yields of natural organic matter and organic compounds: Implications for the fluorescence-based interpretation of organic matter composition. Front. Mar. Sci. 2:98. | Phenylalanine;Salicylic acid;Vanillic acid;Syringic acid;Tryptophan;Tyrosine;Coumarin | - |


---

## 3. Split-half Summary

| Component | Ex_TCC_1 | Ex_TCC_2 | Em_TCC_1 | Em_TCC_2 | Validated | Factor_Similarity | Core_Consistency | Explanation_Rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 0.9931478326191953 | 0.9956344572558693 | 0.996154352515737 | 0.9970540955331506 | Yes | 0.9976992467675259 | 99.99770414194738 | 0.9516432460018832 |


---

## 4. Literature Interpretation by Component

### Sample 9189 - Component C1

Literature title: Burial or mineralization: Origins and fates of organic matter in the water-suspended particulate matter-sediment of macrophyte- and algae-dominated areas in Lake Taihu

In the context of this study, **C1** refers to **Fluorescence Component 1** identified through Excitation-Emission Matrix (EEM) spectroscopy combined with Parallel Factor Analysis (PARAFAC). Based on the document, C1 is interpreted as follows:

---

### **1. Identification and Nature**
- **Method**: EEM fluorescence spectra of dissolved organic matter (DOM) from water, suspended particulate matter (SPM), and sediment samples were decomposed via PARAFAC.
- **Result**: Four components (C1–C4) were identified.  
  **C1 is described as a *terrestrially derived humic-like material*, potentially including compounds such as lignin and tannin** (Section 3.5, Fig. 5).

---

### **2. Potential Source**
- **Primary Source**: **Terrestrial/allochthonous origin**—likely from decaying land plants, soil organic matter, or watershed runoff (e.g., plant litter, forest/soil humics).
- **Evidence**:  
  - C1 is predominant in SPM and sediment samples but less abundant in water (Fig. 6).  
  - The authors associate it with terrestrial humic substances, consistent with typical PARAFAC components linked to weathered plant material.

---

### **3. Key Properties**
- **Fluorescence Signature**: Exhibits excitation/emission wavelengths characteristic of humic-like fluorophores (common in terrestrial organic matter).
- **Behavior in the System**:  
  - More abundant in SPM and sediment than in water.  
  - Indicates that terrestrial OM accumulates in particles and sediments rather than remaining dissolved.
- **Stability**: Likely more recalcitrant (resistant to degradation) compared to protein-like autochthonous components (e.g., C4).

---

### **4. Environmental Implications**
- **Carbon Cycling**: Represents a **terrestrial carbon pathway** into lake systems. Its accumulation in sediments suggests a **burial pathway** for organic carbon, contributing to long-term carbon sequestration.
- **Eutrophication Context**:  
  - In **macrophyte-dominated areas (MDAs)**, higher terrestrial OM (like C1) may reflect greater plant-derived inputs.  
  - In **algae-dominated areas (ADAs)**, terrestrial signals are still present but mixed with algal-derived OM.
- **Water Quality**: Humic-like materials can influence nutrient binding, light attenuation, and microbial activity. Their degradation may release nutrients (e.g., N, P) over time.

---

### **5. Role in the Study’s Conclusions**
- The dominance of C1 in SPM/sediment (vs. C4 in water) highlights a **shift from autochthonous (algal/microbial) DOM in water to terrestrial-dominated OM in deposited particles**.
- This pattern supports the authors’ finding that **OM burial is dominant in both MDAs and ADAs**, with terrestrial humics contributing to sediment carbon storage.
- The persistence of C1 in sediments suggests that **terrestrial OM is relatively preserved**, whereas algal-derived OM (C4) undergoes faster mineralization.

---

### **Summary**
**C1 is a terrestrially derived humic-like fluorescence component** representing recalcitrant organic matter from land sources. It accumulates in suspended particles and sediments, playing a key role in carbon burial and long-term organic matter preservation in Lake Taihu. Its behavior underscores differences in OM fate between macrophyte- and algae-dominated zones, with implications for carbon cycling in eutrophic shallow lakes.

---

### Sample 62 - Component C2

Literature title: Fluorescence Quantum Yields of Natural Organic Matter and Organic Compounds: Implications for the Fluorescence-based Interpretation of Organic Matter Composition

In the context of this document, **C2 refers to a fluorescence component identified through Parallel Factor Analysis (PARAFAC) of Excitation-Emission Matrix (EEM) data**. It is not a molecular formula or a sample ID, but rather a mathematically derived spectral signature representing a group of fluorophores with similar optical properties within dissolved organic matter (DOM).

Here is a detailed interpretation based on the literature context provided:

### 1. **Source and Nature**
- **Mathematical Origin**: C2 is extracted via PARAFAC modeling, a multi-way statistical technique that decomposes complex fluorescence EEMs into underlying components (C1, C2, C3...). Each component represents a co-varying set of fluorophores that behave similarly across samples.
- **Chemical Ambiguity**: The paper emphasizes that PARAFAC components like C2 represent **fluorophore groups** (e.g., phenol-like, indole-like) rather than specific chemical compounds. For example, the study finds that structurally similar compounds (e.g., tyrosine and *p*-cresol) can match the same PARAFAC component, making unambiguous chemical assignment difficult.
- **Potential Matches**: The study’s spectral matching via the OpenFluor database links C2-like components (from other studies) to organic compounds such as **coniferyl alcohol** or **coumarin** (see Table 2 and Figure 7). However, these matches are based on spectral similarity (Tucker congruence coefficient >0.9), not definitive identifications.

### 2. **Optical Properties**
- **Spectral Characteristics**: C2 typically exhibits:
  - **Excitation maxima**: Often in the UV range (~260–310 nm), depending on the specific component.
  - **Emission maxima**: Can vary from UV to visible regions (~300–400 nm), reflecting the degree of conjugation in the fluorophore group.
  - **Stokes shift**: Moderate to large (e.g., ~0.5–1.2 eV), indicating energy loss between absorption and emission.
- **Quantum Yield**: The apparent fluorescence quantum yield (AQY) of DOM components is generally low (0.001–0.02). C2’s AQY would be influenced by the presence of non-fluorescent chromophores, which suppress overall AQY.

### 3. **Environmental Implications**
- **Biogeochemical Indicator**: Components like C2 can trace DOM sources and transformations. For instance:
  - **Surface vs. Deep Waters**: The study finds higher AQYs at 350 nm in deep marine waters, suggesting microbial reprocessing ("humification") increases the fluorescent fraction of DOM. C2 could be part of this signal.
  - **Terrestrial vs. Marine Inputs**: In coastal systems, C2 might be associated with terrestrial-derived phenolic or lignin-like compounds (e.g., from coniferyl alcohol), indicating allochthonous DOM influence.
- **Limitations in Interpretation**: The paper cautions that spectral similarity alone (e.g., matching C2 to a pure compound) is insufficient to confirm chemical identity. Components integrate signals from multiple compounds with overlapping spectra, so C2 should be viewed as an **operational fluorophore class** rather than a discrete molecule.

### 4. **Key Takeaways from the Document**
- C2 is a **PARAFAC-derived fluorescence component** used to simplify and interpret DOM fluorescence EEMs.
- It likely represents a **group of structurally related fluorophores** (e.g., phenolic or aromatic amino acid-like compounds) common in natural organic matter.
- Environmental trends in its fluorescence intensity or AQY can indicate shifts in DOM composition (e.g., microbial degradation, terrestrial inputs), but its chemical identity remains ambiguous without corroborating data (e.g., mass spectrometry).

In summary, **C2 is a spectral component in PARAFAC modeling that helps track DOM dynamics, but it corresponds to a mixture of fluorophores rather than a single compound, reflecting the complexity of natural organic matter.**

---

