# Complete Derivation of the Velocity Error Equation in INS    

## 1. Newton’s Second Law: The Foundation of Motion   

We begin with Newton’s Second Law, which describes the relationship between force and acceleration:

$$
F = ma
$$

where:    

- $F$ is the net force acting on an object,   
- $m$ is the mass of the object,   
- $a$ is the acceleration of the object.    

This law holds only in an inertial frame (a frame that is not accelerating or rotating). However, the Earth is a rotating frame, so we need to modify this equation when analyzing motion relative to Earth.    

## 2. Acceleration in Different Frames of Reference    

### 2.1 Motion in an Inertial Frame   

In a true inertial frame (not rotating or accelerating), an object's acceleration is caused by:    

- External forces (e.g., thrust, drag),   
- Gravity,   
- Environmental effects (e.g., friction).    

However, most navigation systems operate in a rotating frame (e.g., an aircraft moving over Earth). This requires us to modify Newton's laws to include rotation effects.    

### 2.2 Transforming Newton’s Laws to a Rotating Frame   

Consider an inertial coordinate system (denoted by $i$) and a rotating coordinate system (denoted by $n$ for navigation frame). The position of an object is given by:

$$
r_i = C_{n i} r_n
$$

where:    

- $C_{n i}$ is the rotation matrix from the navigation frame to the inertial frame,   
- $r_n$ is the position vector in the navigation frame.    

The velocity in the inertial frame is:

$$
v_i = \frac{d r_i}{d t}
$$

Using the chain rule:

$$
v_i = C_{n i} \frac{d r_n}{d t} + \frac{d C_{n i}}{d t} r_n
$$

Since the navigation frame is rotating with angular velocity $ \Omega_{i e n} $, we use the property:

$$
\frac{d C_{n i}}{d t} = C_{n i} \Omega_{i e n}
$$

Thus, the velocity in the navigation frame is:

$$
v_n = C_{i n} v_i - \Omega_{i e n} r_n
$$

Taking the time derivative:

$$
\dot{v}_n = C_{i n} \dot{v}_i - \Omega_{i e n} v_n - \dot{\Omega}_{i e n} r_n
$$

Ignoring the term $ \dot{\Omega}_{i e n} $ (since Earth's rotation rate is constant), we get:

$$
\dot{v}_n = a_n + g_n - 2 \Omega_{i e n} v_n
$$

where:

- $a_n$ = acceleration in the navigation frame,   
- $g_n$ = gravity vector in the navigation frame,   
- $2 \Omega_{i e n} v_n$ = Coriolis force term.    

This is the fundamental velocity equation used in inertial navigation.    

## 3. Specific Force and Accelerometer Measurements    

### 3.1 What Do Accelerometers Measure?   

An accelerometer does not measure total acceleration; instead, it measures the specific force, which removes gravity:

$$
f = a - g
$$

Thus, the actual acceleration is:

$$
a = f + g
$$

### 3.2 Transforming Acceleration from Body to Navigation Frame   

Since accelerometers are mounted in the body frame, they measure acceleration in the body frame. To convert it to the navigation frame, we use the direction cosine matrix (DCM):

$$
a_n = C_{b n} a_b
$$

where:    

- $C_{b n}$ = DCM converting body-frame vectors to navigation frame,   
- $a_b$ = acceleration measured in the body frame.    

Substituting this into our velocity equation:

$$
\dot{v}_n = C_{b n} a_b + g_n - 2 \Omega_{i e n} v_n
$$

This is the fundamental velocity equation for an inertial navigation system (INS).    

## 4. Introducing the Velocity Error    

### 4.1 Defining the Velocity Error   

Since errors exist in real-world measurements, the estimated velocity differs from the true velocity:

$$
\delta v_n = v_n - v_{n}^{\text{True}}
$$

Taking the time derivative:

$$
\dot{\delta v}_n = \dot{v}_n - \dot{v}_{n}^{\text{True}}
$$

### 4.2 Substituting the Velocity Equations   

For both the estimated and true cases:

$$
\dot{\delta v}_n = (C_{b n} a_b + g_n - 2 \Omega_{i e n} v_n) - (C_{b, \text{True}} a_b^{\text{True}} + g_n^{\text{True}} - 2 \Omega_{i e n} v_n^{\text{True}})
$$

Canceling $g_n$:

$$
\dot{\delta v}_n = C_{b n} a_b - C_{b, \text{True}} a_b^{\text{True}} - 2 \Omega_{i e n} \delta v_n
$$

## 5. Expanding Accelerometer Errors    

### 5.1 Errors in Acceleration Measurements   

Accelerometers introduce:    

- Bias Error ($b_a^b$),   
- Random Noise ($\delta a_b$).    

Thus, the actual accelerometer reading is:

$$
a_b = a_b^{\text{True}} + b_a^b + \delta a_b
$$

Substituting this:

$$
\dot{\delta v}_n = C_{b n} a_b^{\text{True}} + C_{b n} b_a^b + C_{b n} \delta a_b - C_{b, \text{True}} a_b^{\text{True}} - 2 \Omega_{i e n} \delta v_n
$$

## 6. Expanding Orientation Error    

Since gyroscopes introduce errors, the DCM $C_{b n}$ also has errors, modeled as:

$$
C_{b n} = C_{b, \text{True}} + \delta C_{b n}
$$

where:

$$
\delta C_{b n} \approx -[\delta E_n \times] C_{b, \text{True}}
$$

Substituting this:

$$
\dot{\delta v}_n = -[\delta E_n \times] C_{b, \text{True}} a_b^{\text{True}} + (I - [\delta E_n \times]) C_{b, \text{True}} b_a^b + C_{b, \text{True}} \delta a_b - 2 \Omega_{i e n} \delta v_n
$$

## 7. Adding Gravity Error    

For small altitude errors $\delta h_n$:

$$
\delta g_n = -\frac{2 g_0 \delta h_n}{R_E} u_D^n
$$

Substituting into the velocity error equation:

$$
\dot{\delta v}_n = -[\delta E_n \times] C_{b, \text{True}} a_b^{\text{True}} + (I - [\delta E_n \times]) C_{b, \text{True}} b_a^b + C_{b, \text{True}} \delta a_b - 2 \Omega_{i e n} \delta v_n - \frac{2 g_0 \delta h_n}{R_E} u_D^n
$$

✅ **Final velocity error equation derived!** 🎯
