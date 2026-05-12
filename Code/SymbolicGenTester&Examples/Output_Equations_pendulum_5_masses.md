# Equations of Motion: 5-Mass Pendulum

This document contains the second-order differential equations obtained for a multiple pendulum system. Dot notation is used for time derivatives, and multi-line formatting has been applied to improve readability.

## Equation 1

$$
\begin{aligned}
& l_{1}^{2} \left(m_{1} + m_{2} + m_{3} + m_{4} + m_{5}\right) \ddot{\theta}_{1} + g l_{1} \left(m_{1} + m_{2} + m_{3} + m_{4} + m_{5}\right) \sin{\left(\theta_{1} \right)} \\ 
& \quad + l_{1} l_{2} \left(m_{2} + m_{3} + m_{4} + m_{5}\right) \sin{\left(\theta_{1} - \theta_{2} \right)} \dot{\theta}_{2}^{2} + l_{1} l_{2} \left(m_{2} + m_{3} + m_{4} + m_{5}\right) \cos{\left(\theta_{1} - \theta_{2} \right)} \ddot{\theta}_{2} \\ 
& \quad + l_{1} l_{3} \left(m_{3} + m_{4} + m_{5}\right) \sin{\left(\theta_{1} - \theta_{3} \right)} \dot{\theta}_{3}^{2} + l_{1} l_{3} \left(m_{3} + m_{4} + m_{5}\right) \cos{\left(\theta_{1} - \theta_{3} \right)} \ddot{\theta}_{3} \\ 
& \quad + l_{1} l_{4} \left(m_{4} + m_{5}\right) \sin{\left(\theta_{1} - \theta_{4} \right)} \dot{\theta}_{4}^{2} + l_{1} l_{4} \left(m_{4} + m_{5}\right) \cos{\left(\theta_{1} - \theta_{4} \right)} \ddot{\theta}_{4} \\ 
& \quad + l_{1} l_{5} m_{5} \sin{\left(\theta_{1} - \theta_{5} \right)} \dot{\theta}_{5}^{2} + l_{1} l_{5} m_{5} \cos{\left(\theta_{1} - \theta_{5} \right)} \ddot{\theta}_{5} = 0
\end{aligned}
$$

---

## Equation 2

$$
\begin{aligned}
& l_{2}^{2} \left(m_{2} + m_{3} + m_{4} + m_{5}\right) \ddot{\theta}_{2} + g l_{2} \left(m_{2} + m_{3} + m_{4} + m_{5}\right) \sin{\left(\theta_{2} \right)} \\ 
& \quad + l_{1} l_{2} \left(m_{2} + m_{3} + m_{4} + m_{5}\right) \cos{\left(\theta_{1} - \theta_{2} \right)} \ddot{\theta}_{1} + l_{2} l_{3} \left(m_{3} + m_{4} + m_{5}\right) \sin{\left(\theta_{2} - \theta_{3} \right)} \dot{\theta}_{3}^{2} \\ 
& \quad + l_{2} l_{3} \left(m_{3} + m_{4} + m_{5}\right) \cos{\left(\theta_{2} - \theta_{3} \right)} \ddot{\theta}_{3} + l_{2} l_{4} \left(m_{4} + m_{5}\right) \sin{\left(\theta_{2} - \theta_{4} \right)} \dot{\theta}_{4}^{2} \\ 
& \quad + l_{2} l_{4} \left(m_{4} + m_{5}\right) \cos{\left(\theta_{2} - \theta_{4} \right)} \ddot{\theta}_{4} + l_{2} l_{5} m_{5} \sin{\left(\theta_{2} - \theta_{5} \right)} \dot{\theta}_{5}^{2} \\ 
& \quad + l_{2} l_{5} m_{5} \cos{\left(\theta_{2} - \theta_{5} \right)} \ddot{\theta}_{5} -  l_{1} l_{2} \left(m_{2} + m_{3} + m_{4} + m_{5}\right) \sin{\left(\theta_{1} - \theta_{2} \right)} \dot{\theta}_{1}^{2} = 0
\end{aligned}
$$

---

## Equation 3

$$
\begin{aligned}
& l_{3}^{2} \left(m_{3} + m_{4} + m_{5}\right) \ddot{\theta}_{3} + g l_{3} \left(m_{3} + m_{4} + m_{5}\right) \sin{\left(\theta_{3} \right)} \\ 
& \quad + l_{1} l_{3} \left(m_{3} + m_{4} + m_{5}\right) \cos{\left(\theta_{1} - \theta_{3} \right)} \ddot{\theta}_{1} + l_{2} l_{3} \left(m_{3} + m_{4} + m_{5}\right) \cos{\left(\theta_{2} - \theta_{3} \right)} \ddot{\theta}_{2} \\ 
& \quad + l_{3} l_{4} \left(m_{4} + m_{5}\right) \sin{\left(\theta_{3} - \theta_{4} \right)} \dot{\theta}_{4}^{2} + l_{3} l_{4} \left(m_{4} + m_{5}\right) \cos{\left(\theta_{3} - \theta_{4} \right)} \ddot{\theta}_{4} \\ 
& \quad + l_{3} l_{5} m_{5} \sin{\left(\theta_{3} - \theta_{5} \right)} \dot{\theta}_{5}^{2} + l_{3} l_{5} m_{5} \cos{\left(\theta_{3} - \theta_{5} \right)} \ddot{\theta}_{5} \\ 
& \quad -  l_{1} l_{3} \left(m_{3} + m_{4} + m_{5}\right) \sin{\left(\theta_{1} - \theta_{3} \right)} \dot{\theta}_{1}^{2} -  l_{2} l_{3} \left(m_{3} + m_{4} + m_{5}\right) \sin{\left(\theta_{2} - \theta_{3} \right)} \dot{\theta}_{2}^{2} = 0
\end{aligned}
$$

---

## Equation 4

$$
\begin{aligned}
& l_{4}^{2} \left(m_{4} + m_{5}\right) \ddot{\theta}_{4} + g l_{4} \left(m_{4} + m_{5}\right) \sin{\left(\theta_{4} \right)} \\ 
& \quad + l_{1} l_{4} \left(m_{4} + m_{5}\right) \cos{\left(\theta_{1} - \theta_{4} \right)} \ddot{\theta}_{1} + l_{2} l_{4} \left(m_{4} + m_{5}\right) \cos{\left(\theta_{2} - \theta_{4} \right)} \ddot{\theta}_{2} \\ 
& \quad + l_{3} l_{4} \left(m_{4} + m_{5}\right) \cos{\left(\theta_{3} - \theta_{4} \right)} \ddot{\theta}_{3} + l_{4} l_{5} m_{5} \sin{\left(\theta_{4} - \theta_{5} \right)} \dot{\theta}_{5}^{2} \\ 
& \quad + l_{4} l_{5} m_{5} \cos{\left(\theta_{4} - \theta_{5} \right)} \ddot{\theta}_{5} -  l_{1} l_{4} \left(m_{4} + m_{5}\right) \sin{\left(\theta_{1} - \theta_{4} \right)} \dot{\theta}_{1}^{2} \\ 
& \quad -  l_{2} l_{4} \left(m_{4} + m_{5}\right) \sin{\left(\theta_{2} - \theta_{4} \right)} \dot{\theta}_{2}^{2} -  l_{3} l_{4} \left(m_{4} + m_{5}\right) \sin{\left(\theta_{3} - \theta_{4} \right)} \dot{\theta}_{3}^{2} = 0
\end{aligned}
$$

---

## Equation 5

$$
\begin{aligned}
& l_{5}^{2} m_{5} \ddot{\theta}_{5} + g l_{5} m_{5} \sin{\left(\theta_{5} \right)} \\ 
& \quad + l_{1} l_{5} m_{5} \cos{\left(\theta_{1} - \theta_{5} \right)} \ddot{\theta}_{1} + l_{2} l_{5} m_{5} \cos{\left(\theta_{2} - \theta_{5} \right)} \ddot{\theta}_{2} \\ 
& \quad + l_{3} l_{5} m_{5} \cos{\left(\theta_{3} - \theta_{5} \right)} \ddot{\theta}_{3} + l_{4} l_{5} m_{5} \cos{\left(\theta_{4} - \theta_{5} \right)} \ddot{\theta}_{4} \\ 
& \quad -  l_{1} l_{5} m_{5} \sin{\left(\theta_{1} - \theta_{5} \right)} \dot{\theta}_{1}^{2} -  l_{2} l_{5} m_{5} \sin{\left(\theta_{2} - \theta_{5} \right)} \dot{\theta}_{2}^{2} \\ 
& \quad -  l_{3} l_{5} m_{5} \sin{\left(\theta_{3} - \theta_{5} \right)} \dot{\theta}_{3}^{2} -  l_{4} l_{5} m_{5} \sin{\left(\theta_{4} - \theta_{5} \right)} \dot{\theta}_{4}^{2} = 0
\end{aligned}
$$

---

