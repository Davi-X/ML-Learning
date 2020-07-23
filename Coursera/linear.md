# Regression
# One Variable $f(x) = \theta_0 + \theta_1 x$
### Cost Function
He uses Ordinary Least Square
*$J(\theta_0 \theta_1)$* = $\frac{1}{2m}\sum\limits^m_{i=1}(h_\theta(x_i)-y_i)^2$  
(Here add 2 does not effect, but simplify for derivation)
### Gradient Descent
Used to estimate the minimum parameters  
$\theta_i = \theta_i - \alpha\frac{\partial}{\partial\theta_i}J(\theta_0 \theta_1)$  
where $\alpha$ is *Learning rate* (size of each step in descent)
* $\alpha$ is too small --> too slow
* $\alpha$ is too large --> overshoot
* No need to decrease $\alpha$ over time   
	* $\theta_0 = \theta_0 - \alpha\frac{1}{m}\sum\limits^m_{i=1}(f(x^{(i)} - y^{(i)})$ 
	* $\theta_1 = \theta_1 - \alpha\frac{1}{m}\sum\limits^m_{i=1}(f(x^{(i)} - y^{(i)}) \times x^{(i)}$   
(Calculate the partial derivation results)
## Multivariate   
* $x^{(i)}$ is the ith data element
* $x_{j}^{(i)}$ is the jth feature of ith element  
---
* Linear 
  $f(x) = \theta_0(x_0) + \theta_1x_1 + ... + \theta_2x_2 + \theta_mx_m$  
  For convinence, $x_0 = 1, f(x) = \theta^Tx$
  ### Cost Function
  $J(\bold{\theta}) = \frac{1}{2m}\sum\limits^m_{i=1}(f(x^{(i)})-y^{(i)})^2$
  ### Gradient Descent
  $\theta_j = \theta_j - \alpha\frac{1}{m}\sum\limits_{i=1}^m(f(x^{(i)}) - y^{(i)})x_j^{(j)}$
  ### Feature Scaling + Mean normalization
  * Feature Scaling   
  	To **speed up the gradient descent**, need to modify the range of $x_i$   
    * $-1 \le x_i \le 1$ 
    * $-0.5 \le x_i \le 0.5$
    * ...
  * Mean Normalization  
    To **make features have approximately zero mean**,
	replace $x_i$ with $x_i - \mu_i$ (not on $x_0$)
  * $x_i$ = $\frac{x_i -\mu_i}{s_i}$
    * $s_i = max - min$   
  	  $s_i =$ std variation 
	* $\mu_i$ is mean
  * Learning Rate  
	Choose $\alpha$ each time $\times 3$  
	..., 0.001, ... , 0.01, ..., 0.1, ..., 1
  ### Normal Equation  
	A formula to calculate the $min$ $\theta$ in one step  
	* we have m data, each consists n fatures
	* construct a $m \times (n + 1)$ matrix with an extra column $x_o = 1$
	* $min$ $\theta = (X^TX)^{-1}X^Ty$
  	* When data set is rather small, using normal function, but for example n = 10^6, switch using gradient descent
  	* If $X^TX$ is non-invertible
    	* dedundant features, non-linear-independent
    	* Too many features, $m \le n$
          	* Use regularzation
          	* Delete some features
* Polynomials  
  To fit the data, modify the linear version  
  e.g. $f(x) = \theta_0 + \theta_1x_1 + \theta_2\sqrt{x} + \theta_3x^2$  
  **Feature Scaling is crucial**
# Classfication
## Logistic Regression (Linear Regression applies on Binary Classfication)
$z = \theta^T x$  
$f(x) = g(z) = \frac{1}{1 + e^{-z}}$
### Cost Function
We can not apply the original cost function, since its results would be a non-convex function, cannot guarantees the minimum for some gradient algorithm.
* If y = 1  $Cost(f(x), y) = -log(f(x))$  
  then if $f(x) -> 0$, $J(\theta) -> \infty$ as a penality
* If y = 0  $Cost(f(x), y) = -log(1 - f(x))$  
  then if $f(x) -> 1$, $J(\theta) -> \infty$ as a penality

### Simplified Cost Function
* $Cost(f(x), y) = -y\times log(f(x)) - (1-y)\times log(1-f(x))$  
* $J(\theta) = -\frac{1}{m}\sum\limits^m_{i=1}[y^{(i)}\times log(f(x^{(i)})) + (1-y^{(i)})\times log(1-f(x^{(i))})]$
* Using gradient descent to estimate / calculate the min $\theta$  
  $\theta_j = \theta_j - \alpha\frac{1}{m}\sum\limits_{i=1}^m(f(x^{(i)}) - y^{(i)})x_j^{(j)}$  
  Seems the same as linear regression, but note that $f(x) = \frac{1}{1+e^{-\theta^Tx}}$
* Vectorized Implemntation  
  $\theta = \theta - \alpha\frac{1}{m}X^T(g(X\theta)- y)$
### Advanced Optimazation
* Gradient Descent
* Conjugate gradient
* BFGS
* L-BFGS  
  try using some built-in libraries to simplify your code
## MultiClass classification
* OvO (One vs One)
  N(N-1) / 2 classifiers, select the result whose appearance is the most
* OvR (One vs Rest)  
  Select each one as positive at one time, calculate its possibiility, and compare with others.   
  Select the highest p as the result
* MvM (Many vs Many)
  
# Overfitting
* Reason:
  Too many features
* Approaches:
  * Reduce number of features (some are redundant)
    * Select manaully
    * Model selection algorithm
  * Regularization (each feature is useful)
	* Reduce the magnitude of parameters $\theta_j$​.
	* Works well when we have a lot of slightly useful features.
# Regularization  
Idea behind regularization is to minimize the weights, so that we can 'eliminate' some features to fit the curve
## On Linear Regression
* $J(\theta) = min$ $\frac{1}{2m}\sum\limits_{i=1}^m(f(x^{(i)}) - y^{(i)})^2 + some\_particular\_terms$
* If want to reduce all weights  
  $J(\theta) = min$ $\frac{1}{2m}\sum\limits_{i=1}^m(f(x^{(i)}) - y^{(i)})^2 + \lambda\sum\limits_{j=1}^n\theta_j^2$  
  where $\lambda$ is called **regularization parameter**
* $\lambda$ too large, it would smooth out the function, underfitiing
* $\lambda \approx 0$, nearly no regularization
  
### Gradient Descent
Note: DO NOT change $\theta$ at any time  
Repeat until converge
* $\theta_0 = \theta_0 - \alpha\frac{1}{m}\sum\limits_{i=1}^m(f(x^{(i)}) - y^{(i)})x_o^{(i)}$  
* $\theta_j = \theta_j - \alpha[\frac{1}{m}\sum\limits_{i=1}^m(f(x^{(i)}) - y^{(i)})x_o^{(i)} + \frac{\lambda}{m}\theta_j]$  
  where $j \in {1,2,3,...}$ 

$\theta_j = \theta_j(1 - \alpha\frac{\lambda}{m}) - \alpha\sum\limits_{i=1}^m(f(x^{(i)} - y^{(i)}))x_j^{(i)}$  
where ($1 - \alpha\frac{\lambda}{m}) < 1$ 
### Normal Equation
$\theta = (X^TX + \lambda L)^{-1}X^Ty$  
where $L = \begin{bmatrix}
			0 &   &   & ... & \\
			  & 1 &   & ... & \\
			  &   & 1 & ... & \\
			..& ..& ..& ... & \\
			  &   &   & ... & 1
		   \end{bmatrix}$

This also solves the non-invertible problems earlier, since **$X^TX + \lambda L$ is invertible**
## On Logistic Regression
New loss $J(\theta) = -\frac{1}{m}\sum\limits_{i=1}^m[y^{(i)}\times log(f(x^{(i)}) + (1 - y^{(i)})\times log(1- f(x^{(i)}))] + \frac{\lambda}{2m}\sum\limits_{j=1}^n\theta^2_j$
### Gradient Descent
Repeat until converge
* $\theta_0 = \theta_0 - \alpha\frac{1}{m}\sum\limits_{i=1}^m(f(x^{(i)}) - y^{(i)})x_o^{(i)}$  
* $\theta_j = \theta_j - \alpha[\frac{1}{m}\sum\limits_{i=1}^m(f(x^{(i)}) - y^{(i)})x_o^{(i)} + \frac{\lambda}{m}\theta_j ]$  
  where $j \in {1,2,3,...}$   
  same as linear regression loss function(mind the $f(x)$ is different here)

# Non-linear
## Non-linear hypothesis

Since if we have many features, and be quadratic / cubic /... the number will be explosion, which do not fit for linear study