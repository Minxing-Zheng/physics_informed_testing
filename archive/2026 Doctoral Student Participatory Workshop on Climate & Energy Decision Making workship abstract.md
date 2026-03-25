**Learning to Test: Physics-Informed Representation for Electric-System Instability Detection**

**Minxing Zheng**$^1$, **Zewei Deng**$^2$, **Liyan Xie**$^2$, **Shixiang Zhu**$^1$

$^1$Carnegie Mellon University, $^2$University of Minnesota

**Abstract:** Many modern electric power systems evolve according to differential–algebraic equations (DAEs) that couple network constraints with generator and load dynamics. In practice, these systems are subject to stochastically varying external operating conditions, so stability is not a static property but must be reassessed as the external input distribution shifts. Rather than re-estimating physical parameters or repeatedly solving the underlying DAE, we learn a physics-informed latent representation of external variables that captures stability-relevant structure and is regularized toward a tractable reference distribution. Trained on baseline data from a certified safe regime, the learned representation enables deployment-time safety monitoring to be formulated as a distributional hypothesis test in latent space, with controlled Type I error. By integrating neural dynamical surrogates, uncertainty-aware calibration, and uniformity-based testing, our approach provides a scalable and statistically grounded method for detecting instability risk in general stochastic constrained dynamical systems without repeated simulation.

