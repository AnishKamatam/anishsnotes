Model Training can be derived into 3 steps:
1. Forward Pass - inputs through model to yield outputs
2. Backward Pass - compute gradients
3. Optimization Step - update parameters via gradient descent

![[Screenshot 2026-01-19 at 9.51.51 AM.png]]

The Three Key Challenges:
1. Memory Usage
2. Compute Efficiency
3. Communication Overhead


The *batch size* (bs) is one of the important *hyperparameters* for model training; it affects both model convergence and throughput.

	Small batch size -> noisy gradients, model may not converge to optimal finalperformance
	- requires more optimizer steps, optimizer steps are expensive in compute + add total time to train

	Large batch size -> less importance per training token, slower convergence
	- more accurate gradient estimations, potentially waste compute resources
