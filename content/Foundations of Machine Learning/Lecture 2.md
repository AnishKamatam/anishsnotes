---
title: Lecture 2
---

## 1. The Learning Setup


### 1.1 The Domain Set ($\mathcal{X}$)
This is the space from which our examples are drawn.
* **Example:** Predicting if a Papaya is tasty.
* **Feature Representation:** We do not feed the physical object into the model; we feed a vector of features.
    * Let features be Softness and Color, scaled between 0 and 1.
    * $\mathcal{X} = [0, 1] \times [0, 1]$ (A 2D plane).
    * Every point $x \in \mathcal{X}$ represents a specific papaya.

### 1.2 The Label Set ($\mathcal{Y}$)
The set of possible outcomes or answers.
* **For Papayas:** $\mathcal{Y} = \{0, 1\}$ (where 1 is tasty, 0 is not).

### 1.3 The Training Sample ($S$)
A finite sequence of labeled data points available to the learner.
$$S = \{ (x_1, y_1), (x_2, y_2), ..., (x_m, y_m) \}$$
* **Note:** These are historical observations (papayas we have already tasted).

### 1.4 The Learner's Output ($h$)
The learner produces a prediction rule, called a **Hypothesis**.
$$h: \mathcal{X} \rightarrow \mathcal{Y}$$
* **Goal:** $h(x)$ should accurately predict the label of a *new*, unseen $x$.

---

## 2. Measuring Quality: True Loss vs. Empirical Loss

To evaluate a hypothesis, we need to distinguish between how well it does on **observed data** versus **reality**.

### 2.1 The Generating Distribution ($\mathcal{D}$) and True Function ($f$)
We assume:
1.  There is a probability distribution $\mathcal{D}$ over $\mathcal{X}$ (some papayas are more common than others).
2.  There is a "Truth" function $f: \mathcal{X} \rightarrow \mathcal{Y}$ that determines the correct label.
    * $y_i = f(x_i)$ for all training examples.

### 2.2 True Loss (The Generalization Error)
The probability that the hypothesis makes an error on a random sample drawn from the real world.
$$L_{\mathcal{D}}(h) = \mathbb{P}_{x \sim \mathcal{D}} [ h(x) \neq f(x) ]$$
* **Challenge:** We cannot calculate this directly because we do not know $\mathcal{D}$. This is a theoretical quantity.

### 2.3 Empirical Loss (The Training Error)
The error rate calculated on the specific training sample $S$.
$$L_S(h) = \frac{1}{m} \sum_{i=1}^{m} \mathbb{I}[ h(x_i) \neq y_i ]$$
* **Note:** This is simply the fraction of mistakes made on the training data.

---

## 3. Empirical Risk Minimization (ERM)

Since we cannot minimize the True Loss (unknown), we minimize what we can measure.

> **ERM Principle:** Choose the hypothesis that performs best on the training data $S$.

$$h_S \in \text{argmin}_{h} L_S(h)$$

### The Problem: Overfitting
If we allow the learner to choose *any* possible function, ERM fails.

**The "Memorization" Hypothesis:**
Imagine a hypothesis $h_{mem}$ defined as:
$$h_{mem}(x) = \begin{cases} y_i & \text{if } x = x_i \text{ (matches a training point)} \\ 0 & \text{otherwise (predict "not tasty")} \end{cases}$$

* **Result:**
    * $L_S(h_{mem}) = 0$ (Perfect training accuracy).
    * $L_{\mathcal{D}}(h_{mem}) \approx \text{High}$ (Fails on everything else).
* **Conclusion:** Minimizing empirical loss alone is not enough to guarantee learning.

---

## 4. Inductive Bias & Hypothesis Classes

To fix overfitting, we must restrict the search space. We introduce **Inductive Bias**.

### 4.1 The Hypothesis Class ($\mathcal{H}$)
We restrict the learner to choose $h$ only from a predefined set of functions, $\mathcal{H}$.

* **Papaya Example:** We assume "Tastiness" is defined by a specific range of color and softness.
* **The Class:** Let $\mathcal{H}$ be the set of all axis-aligned rectangles in the 2D plane.
    * Inside the rectangle = Tasty ($1$).
    * Outside the rectangle = Not Tasty ($0$).

### 4.2 ERM over $\mathcal{H}$
$$ERM_{\mathcal{H}}(S) = \text{argmin}_{h \in \mathcal{H}} L_S(h)$$
By searching only within $\mathcal{H}$ (rectangles), the learner cannot simply "memorize" scattered points. It is forced to find a structure, which leads to better generalization.

---

## 5. The First Fundamental Theorem of Learning

This theorem provides the mathematical guarantee that ERM works for finite hypothesis classes.

### 5.1 The Assumptions
1.  **Finite Class:** The size of the hypothesis class $|\mathcal{H}|$ is finite.
2.  **Realizability:** The true function $f$ is actually inside $\mathcal{H}$ (i.e., $L_{\mathcal{D}}(f) = 0$).
3.  **i.i.d.:** The training samples are Independent and Identically Distributed according to $\mathcal{D}$.

### 5.2 The Conclusion
If the sample size $m$ is large enough, $ERM_{\mathcal{H}}$ will output a hypothesis that is probably approximately correct.

Mathematically, for any error tolerance $\epsilon$ and confidence parameter $\delta$, if:
$$m \geq \frac{\log(|\mathcal{H}|/\delta)}{\epsilon}$$
Then with probability at least $1-\delta$:
$$L_{\mathcal{D}}(h_S) \leq \epsilon$$

---

## 6. Proof Intuition (Step-by-Step)

We want to prove that the probability of picking a "bad" hypothesis becomes tiny as we see more data.

**Step 1: Define "Bad" Hypotheses**
Let $\mathcal{H}_{bad}$ be the set of hypotheses that have a high true error (worse than $\epsilon$).
$$\mathcal{H}_{bad} = \{ h \in \mathcal{H} : L_{\mathcal{D}}(h) > \epsilon \}$$

**Step 2: Misleading Samples**
ERM only fails if a bad hypothesis $h \in \mathcal{H}_{bad}$ looks perfect on the training data (i.e., $L_S(h) = 0$). We need to calculate the probability of this happening.

**Step 3: Probability for a Single Bad Hypothesis**
Take *one* specific bad hypothesis $h_b$.
* Probability it predicts *one* random point correctly: $\leq (1 - \epsilon)$.
* Probability it predicts *all* $m$ points correctly (due to i.i.d.):
    $$\leq (1 - \epsilon)^m$$
* *Interpretation:* As $m$ grows, this number decays exponentially.

**Step 4: Union Bound (The Crucial Step)**
We don't just have one bad hypothesis; we might have many. The probability that *at least one* bad hypothesis fools us is bounded by the sum of their individual probabilities.
$$P(\exists h \in \mathcal{H}_{bad} \text{ s.t. } L_S(h)=0) \leq |\mathcal{H}_{bad}| \cdot (1 - \epsilon)^m$$

Since $|\mathcal{H}_{bad}| \leq |\mathcal{H}|$:
$$\text{Total Probability of Failure} \leq |\mathcal{H}| \cdot (1 - \epsilon)^m$$

**Step 5: Conclusion**
By taking log and rearranging, we see that if $m$ is large enough, the "Probability of Failure" drops below $\delta$.

---

## 7. Summary & Key Takeaways

| Concept | Definition |
| :--- | :--- |
| **True Loss** $L_{\mathcal{D}}(h)$ | The error on the real world (unknown). |
| **Empirical Loss** $L_S(h)$ | The error on the training set (known). |
| **Overfitting** | When $L_S(h)$ is low but $L_{\mathcal{D}}(h)$ is high. Caused by memorization. |
| **Inductive Bias** | Restricting $\mathcal{H}$ to prevent overfitting (e.g., only rectangles). |
| **Theorem 1** | If $\mathcal{H}$ is finite and realizable, enough data guarantees learning. |
