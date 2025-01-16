In this problem, there are three steps. 

Step 1: Complete Data  
Input : Incomplete Data (6x + 4y + z)  
Output : Complete Data (6x + 4y)  
Methods:
- Regression  
- Moving Average  
- ML/DL

Step 2: Velocity Estimation
Input : Complete Data (6x + 4y)  
Output : $V_x,V_y,V_z$  
Methods:
- MLP  
- LSTM  
- CNN  
- MNN  
- FAN  
- TCN  

Step 3: Bias Estimation followed by Correction  
Input : $V_x,V_y,V_z$   
Output : Bias Free Data (Cleaner Data)  

Methods:  
- Standard Kalman Filter (KF): Best for linear systems  
- Extended Kalman Filter (EKF): For mildly non-linear systems.  
- Unscented Kalman Filter (UKF): Best for strongly non-linear systems.  
- Ensemble Kalman Filter (EnKF): For high-dimensional, complex non-linear systems.  
- Particle Filter (PF): For non-Gaussian, highly non-linear systems.  
- Unscented Particle Filter (UPF): Combination of UKF and Particle Filter for non-linear systems.  
- Kalman-Bucy Filter: Continuous-time systems.  
- Rauch-Tung-Striebel (RTS) Smoother: For smoothing past estimates.  
- Cubature Kalman Filter (CKF): A second-order non-linear approximation.  
- Information Filter (IF): Works in information form for sensor fusion and high precision.  
- Square Root Kalman Filter (SRKF)  

_We need to find out the best combinatino for the velocity prediction._

So, the tasks are to write codes individually and then run simularions on all the possible combinations. 