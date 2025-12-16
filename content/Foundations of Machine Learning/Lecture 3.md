---
title: Lecture 3
---

---
# Lecture 3: The Foundations of Learning Theory

## 1. The Learning Model Setup

### 1.1 Hypothesis Class ($\mathcal{H}$)

- **Input Domain ($X$):** The space of all possible inputs.
    
- **Labels ($Y$):** Binary labels, $Y = \{0, 1\}$.
    
- Hypothesis Class ($\mathcal{H}$): A predefined set of candidate predictor functions. Each function $h \in \mathcal{H}$ maps inputs to labels.
    
    $$\mathcal{H} \subseteq \{h \mid h: X \rightarrow \{0,1\}\}$$
    

### 1.2 Error Measures

- **Data Generation (Realizable Case):** We assume data points are drawn i.i.d. from an unknown distribution $D$ over $X$, and a true labeling function $f$ exists.
    
- **Target Function ($f$):** $f \in \mathcal{H}$ (Realizability Assumption).
    
- True (Generalization) Error ($L_{D,f}(h)$): The probability of $h$ making an error on a new, unseen sample drawn from $D$.
    
    $$L_{D,f}(h) = \Pr_{x \sim D}[h(x) \neq f(x)]$$
    

---

## 2. Empirical Risk Minimization (ERM)

### 2.1 The Learning Paradigm

ERM is a high-level learning principle, not a specific algorithm, that guides the choice of a hypothesis based on observed data.

- **Input:** A labeled training sample $S=\{(x_1,y_1), \ldots, (x_m,y_m)\}$.
    
- Empirical Error (Training Error): The fraction of errors $h$ makes on the training sample $S$.
    
    $$L_S(h) = \frac{1}{m} \sum_{i=1}^m \mathbf{1}[h(x_i) \neq y_i]$$
    
- ERM Output: The hypothesis $h$ that minimizes the training error.
    
    $$\text{ERM}_{\mathcal{H}}(S) \in \arg\min_{h \in \mathcal{H}} L_S(h)$$
    

> **Note on ERM:** ERM is called a paradigm because it abstracts away computational concerns (finding the minimizer) and does not specify how to handle cases where multiple hypotheses achieve the minimum empirical error.

---

## 3. PAC Learning (The Realizable Case)

### 3.1 The Main Theorem: Learnability of Finite Classes

The theorem establishes that any finite hypothesis class $\mathcal{H}$ is learnable in the PAC sense, provided the Realizability Assumption holds.

Theorem: If $\mathcal{H}$ is finite, then for any $f \in \mathcal{H}$ and any distribution $D$:

$$\Pr_{S \sim D^m} \left[ L_{D,f}(\text{ERM}_{\mathcal{H}}(S)) > \epsilon \right] \le |\mathcal{H}| (1-\epsilon)^m$$

This bound holds for all $\epsilon > 0$ and all sample sizes $m$.

### 3.2 Proof Intuition (Union Bound)

1. **Step 1: Focus on one "Bad" Hypothesis.** Consider a single hypothesis $h$ that is bad, meaning its true error is greater than $\epsilon$: $L_{D,f}(h) > \epsilon$. The probability that this bad $h$ achieves zero empirical error ($L_S(h)=0$) on a random sample of size $m$ is bounded by $(1-\epsilon)^m$.
    
2. Step 2: Union Bound. The "failure event" is when the ERM hypothesis $\text{ERM}_{\mathcal{H}}(S)$ is bad. This happens if at least one bad hypothesis $h \in \mathcal{H}$ appears perfect on the sample. We sum the probabilities of this event over all hypotheses in $\mathcal{H}$.
    
    $$\Pr[\text{Failure}] \le \sum_{h \in \mathcal{H}} \Pr[h \text{ looks perfect on } S \mid L_{D,f}(h) > \epsilon] \le |\mathcal{H}| (1-\epsilon)^m$$
    
3. Exponential Decay: Using the approximation $(1-\epsilon)^m \le e^{-\epsilon m}$, the failure probability decreases exponentially with the sample size $m$:
    
    $$\Pr[\text{Failure}] \le |\mathcal{H}| e^{-\epsilon m}$$
    

### 3.3 PAC Learning Definition

A hypothesis class $\mathcal{H}$ is **Probably Approximately Correct (PAC) learnable** if a learner $A$ exists such that for any accuracy $\epsilon>0$ and confidence $\delta>0$, the learner can output a hypothesis $A(S)$ whose true error is at most $\epsilon$, with probability at least $1-\delta$.

$$\Pr_{S \sim D^m} \left[ L_{D,f}(A(S)) > \epsilon \right] \le \delta$$

### 3.4 Sample Complexity

By inverting the failure probability bound and setting it equal to $\delta$, we derive the minimum sample size required for PAC learning:

$$m \ge m_{\mathcal{H}}(\epsilon,\delta) = \frac{\ln |\mathcal{H}| + \ln(1/\delta)}{\epsilon}$$

**Conclusion:** Every finite hypothesis class is PAC learnable, and $\text{ERM}_{\mathcal{H}}$ is a valid PAC learner.

---

## 4. Agnostic PAC Learning (Moving Beyond Realizability)

### 4.1 The Need for Agnostic Learning

The **Weakness of PAC Learning** is the strong **Realizability Assumption** ($f \in \mathcal{H}$). In reality:

- Labels are often **noisy**.
    
- No deterministic target function $f$ may exist in $\mathcal{H}$ (or even in the universe of functions).
    
- Data is better modeled by a joint distribution $D$ over $X \times Y$.
    

### 4.2 Bayes Optimal Predictor

In the general setting, the data model is $(x,y) \sim D$ over $X \times Y$. The best possible predictor $h^*$ minimizes the classification error:

$$h^*(x) = \begin{cases} 1 & \text{if } \Pr(y=1 \mid x) \ge \tfrac{1}{2} \\ 0 & \text{otherwise} \end{cases}$$

This requires knowing the true distribution $D$.

### 4.3 Agnostic PAC Definition

The goal is to perform almost as well as the **Best-in-Class Hypothesis** $h_{\mathcal{H}}^*$, where $h_{\mathcal{H}}^* = \arg\min_{h \in \mathcal{H}} L_D(h)$.

A class $\mathcal{H}$ is **agnostic PAC learnable** if there exists a learner $A$ and function $m_{\mathcal{H}}$ such that, whenever $m \ge m_{\mathcal{H}}(\epsilon,\delta)$:

$$\Pr_{S \sim D^m} \left[ L_D(A(S)) > \min_{h \in \mathcal{H}} L_D(h) + \epsilon \right] \le \delta$$

> **Interpretation:** The learner guarantees that, with high probability ($1-\delta$), its generalization error $L_D(A(S))$ is at most $\epsilon$ greater than the error of the _best possible_ hypothesis within the class $\mathcal{H}$. The guarantee is **relative** rather than absolute.

---

## 5. Key Takeaways

|**Feature**|**Standard (Realizable) PAC Learning**|**Agnostic PAC Learning**|
|---|---|---|
|**Assumption**|Realizability: $f \in \mathcal{H}$|No Realizability Assumption|
|**Data Model**|$x \sim D$, $y = f(x)$|$(x,y) \sim D$ (stochastic labels)|
|**Learning Goal**|Achieve $L_D(A(S)) \approx 0$|Achieve $L_D(A(S)) \approx \min_{h \in \mathcal{H}} L_D(h)$|
|**Guarantee**|Absolute error bounded by $\epsilon$|Relative error bounded by $\epsilon$ above the best-in-class error|
|**Learner**|ERM is a valid learner|ERM is often used (requires complexity control)|

[[Lecture 2|← Previous: Lecture 2]] | [[Lecture 4|Next: Lecture 4 →]]

