Kalman Filters are recursive mathematical functions which help us calculate the exact value of a observable/unobservable measurements using the sensor data. 

Variants:
 - EKF (Extended Kalman Filters)
 - UKF (Unscented Kalman Filters)


-------- Implementing a basic Kalam Filters ----------------------------------------------------      

*This will be the most detailed implementation of a Kalman Filter*

So, for every system, there are viarable and Constant Matrices. 

The variable Matrices are the ones which get updated after every loop.. Where are the constant matrices dont change. 

The equations are : 

Prediction  
  


$X_k^-$ = $AX_{k-1}$ + $BU_k$  
$P_k^-$ = $AP_{k-1}A^T$ + Q  

Update  

$V_k$ = $Y_k$ - $H_kX_k^-$
$S_k$ = $H_kP_k^-H_k^T$ + $R_k$  
$K_k$ = $P_k^-H_k^TS_k^-1$  
$X_k$ = $X_k^-$ + $K_kV_k$  
$P_k$ = $P_k^-$ - $K_kS_kK_k^T$  

Constants:   
    - A  
    - B  
    - Q  
    - H  
    - R  
Variables:  
    - $X_k^-$  
    - $P_k^-$  
    - $V_k$  
    - $S_k$  
    - $K_k$  
    - $X_k$  
    - $P_k$  

----------------------------------------------------      

This part will be about how Kalman Filters are used to calculate non measureable values

----------------------------------------------------      
