---
title: Lecture 1
---

# Machine Learning 

## 1. What Learning Means
> **Definition:** Learning is the process of moving from **experience** to **expertise**.

Machine learning formalizes this: given data (experience), produce a rule or system (expertise) that performs well on new inputs.

A central goal is **generalization**. Memorizing training data is not enough; the learner must handle unseen examples. This ability to extend from past cases to new ones is called **inductive reasoning**.

***

## 2. Learning in Nature

### 2.1 Bait Shyness (Rats)
Rats take a very small bite of unfamiliar food. If they become sick hours later, they associate the sickness with that food and avoid it permanently.

**Key Characteristics:**
*   **One-shot learning**: Happens instantly after one event.
*   **Correct generalization**: They correctly identify the cause.
*   **Evolutionary advantage**: Vital for survival.

### 2.2 Pigeon Superstition (Skinner)
Skinner placed pigeons in a cage with colored objects and released food at random times. Each pigeon reinforced whatever random action it was doing (e.g., pecking a toy) when food appeared.
*   **Result**: This is still learning, but it produces **incorrect generalization**.

### 2.3 Garcia’s Experiment: Prior Knowledge
Garcia tested whether rats could associate:
1.  Eating poisoned food.
2.  A bell sound played at the moment of eating.

**Result**: Rats **failed** to learn this association, despite easily learning associations with taste or smell.

**Takeaway (Prior Knowledge):**
*   Rats have evolved to treat **taste/smell** as relevant cues for sickness.
*   Random sounds are ignored.
*   **Conclusion**: Without prior knowledge to filter relevant features, learning is impossible.

***

## 3. Why Machine Learning Is Needed

### 3.1 Tasks Too Complex to Program
Real-world tasks often lack simple rule-based definitions:
*   Speech recognition
*   Autonomous driving
*   Spam detection

### 3.2 Tasks Requiring Massive Data
Humans cannot process the scale of data required for:
*   Search ranking
*   Ad recommendation
*   Genomics

### 3.3 Adaptivity
Systems must adjust to changing environments:
*   Handwriting recognition (adapting to specific users).
*   Spam filters (adapting to new spammer tactics).

***

## 4. Types of Machine Learning

### 4.1 Core Paradigms
1.  **Supervised Learning**: Training data includes labels (e.g., Spam vs. Not Spam).
2.  **Unsupervised Learning**: No labels; goal is structure discovery (e.g., Clustering).
3.  **Reinforcement Learning**: Feedback is delayed/indirect (e.g., winning a game).

### 4.2 Learning Settings
*   **Batch vs. Online**
    *   *Batch*: All data is available at once.
    *   *Online*: Data arrives sequentially; predictions are made on the fly.
*   **Passive vs. Active**
    *   *Passive*: Learner merely observes data.
    *   *Active*: Learner queries specific data points or runs experiments.
*   **Data Source Nature**
    *   *Cooperative*: Teacher tries to help.
    *   *Indifferent*: Nature (random).
    *   *Adversarial*: Source tries to fool the learner (e.g., email spam).

***

## 5. Comparison to Other Fields

| Feature | Machine Learning | Statistics |
| :--- | :--- | :--- |
| **Focus** | Algorithmic efficiency & computation | Model inference & properties |
| **Assumptions** | Distribution-free (minimal) | Strong generative assumptions |
| **Guarantees** | Finite-sample (real-world size) | Asymptotic (limit $\to \infty$) |

**vs. Classical AI:**
ML is mathematically grounded and avoids heuristics. The goal is not to replicate human reasoning, but to build reliable learning procedures.

***

## 6. Formal Example: Papaya Tasting

### 6.1 Setup
We determine if a papaya is tasty based on two features:
*   **Color**: (green $\to$ red)
*   **Softness**: (hard $\to$ mushy)

Representation: Each papaya is a vector $x \in \mathbb{R}^2$.

### 6.2 Training Data
We observe a dataset $S$ of $m$ labeled examples:
$$ S = \{ (x_1, y_1), (x_2, y_2), \dots, (x_m, y_m) \} $$

Where:
*   $x_i \in \mathbb{R}^2$ is the feature vector.
*   $y_i \in \{ \text{tasty}, \text{not tasty} \}$ is the label.

### 6.3 Prediction Rule (Hypothesis)
We seek a function $f$ to predict the label of new data:
$$ f: \mathbb{R}^2 \to \{ \text{tasty}, \text{not tasty} \} $$

### 6.4 Assumptions
1.  **Random Data Generation**: Examples are sampled independently from a distribution $D$.
2.  **Realizability**: A perfect separating rectangle exists in the feature space (simplifying assumption).

### 6.5 Measuring Success
We evaluate the predictor by its **Probability of Error** on a new random draw:

$$ L_{D}(f) = \underset{x \sim D}{\mathbb{P}} [ f(x) \neq y ] $$

*This equation forms the basis for theoretical analysis in ML.*