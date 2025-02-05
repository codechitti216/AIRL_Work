## Inertial Navigation Equations & Proofs

### Introduction
This document contains **detailed derivations and proofs** of **inertial navigation equations**, starting from **first principles** (Newton’s Laws) and building up to **real-world implementations**.

---

### Table of Contents
1. [Fundamental Principles](#fundamental-principles)
2. [Reference Frames & Transformations](#reference-frames-and-transformations)
3. [Attitude Representation & Update](#attitude-representation-and-update)
4. [Core Navigation Equations](#core-navigation-equations)
5. [Gravity Models & Non-Inertial Effects](#gravity-models-and-non-inertial-effects)
6. [Error Models & Sensor Corrections](#error-models-and-sensor-corrections)
7. [Numerical Methods & Implementations](#numerical-methods-and-implementations)
8. [Common Mistakes & Misconceptions](#common-mistakes-and-misconceptions)
9. [Diagrams & Visuals](#diagrams-and-visuals)
10. [References](#references)

---

### 1. Fundamental Principles
The entire derivation is based on **Newtonian mechanics** and classical kinematics.

#### 1.1 Kinematic Equations
- **Velocity Definition**  
$$ v = \frac{ds}{dt} $$
  
- **Acceleration Definition**  
$$ a = \frac{dv}{dt} $$
  
- **Newton’s Second Law**  
$$ F = ma $$

#### 1.2 Placeholder: **Basic Examples for Intuition**
📌 **TODO:** Add simple examples of motion equations **before adding complexity**. Example: A ball rolling in a straight line vs. on a rotating platform.

#### 1.3 Conservation Laws  
- Work-Energy Theorem  
- Conservation of Angular Momentum  

---

### 2. Reference Frames and Transformations
Navigation requires transforming motion equations across multiple **reference frames**.

#### 2.1 Inertial and Non-Inertial Frames  
- **ECI (Earth-Centered Inertial) Frame**  
- **ECEF (Earth-Centered Earth-Fixed) Frame**  
- **Local Navigation Frames (NED, ENU)**  
- **Body Frame (IMU Sensor Frame)**  

#### 2.2 Placeholder: **Interactive Animations / Diagrams**
📌 **TODO:** Add animations or **step-by-step diagrams** to show how a moving vehicle’s velocity changes in different frames.

#### 2.3 Transformation Between Frames  
- **Rotation Matrix**  
  $$ \mathbf{v}_{\text{new}} = \mathbf{R} \cdot \mathbf{v}_{\text{old}} $$
  
- **Velocity Transformation (ECI to ECEF)**  
  $$ \mathbf{v}_{\text{ECEF}} = \mathbf{R}(\theta) \mathbf{v}_{\text{ECI}} - \boldsymbol{\omega}_{\text{Earth}} \times \mathbf{r}_{\text{ECI}} $$
  
- **Acceleration Transformation (Coriolis & Centrifugal Terms)**  

---

### 3. Attitude Representation and Update
Orientation of a moving body is represented using **rotation matrices, Euler angles, and quaternions**.

#### 3.1 Euler Angles  
- Pitch, Roll, Yaw definitions  

#### 3.2 Placeholder: **Explain Why Euler Angles Are Not Always the Best Choice**  
📌 **TODO:** Explain **gimbal lock** using diagrams and why quaternions are often used instead.

#### 3.3 Direction Cosine Matrix (DCM)  
- Rotation transformation  

#### 3.4 Quaternion Representation  
- Quaternion algebra and advantages  

#### 3.5 Attitude Update Equation  
- Using Gyro Angular Velocity:  
  $$ \frac{d}{dt} C^b_n = C^b_n \Omega^b_{nb} $$

---

### 4. Core Navigation Equations
The **position, velocity, and attitude** of a vehicle are computed iteratively using IMU sensor readings.

#### 4.1 Specific Force Transformation  
- Converting accelerometer measurements from **body frame** to **navigation frame**  
  $$ \mathbf{f}^n_{ib} = C^n_b \mathbf{f}^b_{ib} $$

#### 4.2 Placeholder: **Physical Meaning of Specific Force**  
📌 **TODO:** Explain **what IMUs actually measure** and how they relate to true acceleration.

#### 4.3 Velocity Update  
- Coriolis and transport rate corrections  
  $$ \frac{d}{dt} \mathbf{v}^n_{eb} = \mathbf{f}^n_{ib} + \mathbf{g}^n - (2\mathbf{\Omega}^n_{ie} + \mathbf{\Omega}^n_{en}) \mathbf{v}^n_{eb} $$
  
#### 4.4 Position Update  
- Updating latitude, longitude, and height  

---

### 5. Gravity Models and Non-Inertial Effects  
Gravity is a function of location and affects the motion equations.  

#### 5.1 Placeholder: **Intuitive Explanation of Gravity Variations**  
📌 **TODO:** Explain **why gravity varies with latitude and height** (mention Earth’s oblate shape).

---

### 6. Error Models & Sensor Corrections
IMU sensors introduce **biases and noise**, which must be corrected.

#### 6.1 Sources of Errors  
- Bias Instability  
- Scale Factor Errors  

#### 6.2 Placeholder: **Step-by-Step Example of Drift**  
📌 **TODO:** Show an **example of a 1% drift in velocity** and how it leads to large errors in position.

---

### 7. Numerical Methods & Implementations
📌 **TODO:** Explain **how to implement these equations in code** (Python, MATLAB, C++).  
- **Basic IMU Integration Algorithm**  
- **Filtering (Kalman, Complementary Filter)**  

---

### 8. Common Mistakes & Misconceptions
📌 **TODO:** Add a **list of common mistakes** beginners make when working with navigation equations.  

Example:  
❌ **Wrong:** Using velocity directly from an IMU (IMU only gives acceleration).  
✅ **Correct:** Integrate accelerometer readings to get velocity.  

---

### 9. Diagrams and Visuals
**Add images for better understanding:**  
- **Coordinate Systems:** ![Frames](images/frames.png)  
- **Rotation Matrix Example:** ![Rotation](images/rotation_matrix.png)  
- **Gravity Model:** ![Gravity](images/gravity.png)  

---

### 10. References
- [1] Paul D. Groves, *Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems*, Artech House, 2013.  
- [2] Titterton & Weston, *Strapdown Inertial Navigation Technology*.  
- [3] Jekeli, *Inertial Navigation Systems with Geodetic Applications*.  

---

### **How to Contribute**
If you'd like to contribute equations, proofs, or corrections, feel free to submit a **pull request** or open an **issue**.
