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

# NN (Neural Network)
Based on neurons (nodes) which consists with input and output
Generally it has one input layer, multiple hidden layers and one output layer
* $a^{(j)}_i$ is the activation of unit i in layer j
* $\theta^{(j)}$ matrix of weights mapping from layer j to j + 1  
  If j layer - $S_j$ unis, j + 1 layer - $S_{j + 1}$ units   
  $dim(\theta^{(j)}) = S_{j + 1} \times (S_j + 1)$
  Since each one consists with a 'bias unit' ($x_0$)

## Forward Propagation
* $a^{(i)} = \begin{bmatrix}
			a_0^{(i)} \\
			a_1^{(i)} \\
			a_2^{(i)} \\
			a_3^{(i)}
			\end{bmatrix}$ 
* $z^{(i + 1)} = \begin{bmatrix}
			z_1^{(i + 1)} \\
			z_2^{(i + 1)} \\
			z_3^{(i + 1)}
			\end{bmatrix}$   
where $z^{(i + 1)} = \theta^{(i)} a^{(i)}$  
* $a^{(i + 1)} = g(z^{(i + 1)})$
* Add $a_o^{(i + 1)} = 0$, bias unit
* Repeat until finish
  Note in the last step, it is just the same as logistic regression

## Simple example
* Single layer neuron can deal with some logical operations, given $\theta$ matrix values, we can judge which logical operation it is.

* Some complex logical operations need to use multiple layers to express   
  e.g. XNOR = OR v (~A ^ ~B)

## Multiclass classification
$y \in \mathbb{R}^n$   e.g. $\begin{bmatrix}0 \\1 \\0 \\0\end{bmatrix}$

**General Process**:  
$\begin{bmatrix}
	x_0 \\ x_1 \\... \\ x_n
\end{bmatrix} 
	\rightarrow 
\begin{bmatrix}
	a^{(2)}_0 \\a^{(2)}_1 \\... \\a^{(2)}_n
\end{bmatrix}
	\rightarrow ... \rightarrow
\begin{bmatrix}
	f(x)_1 \\f(x)_2 \\... \\f(x)_n
\end{bmatrix}$ 

## Cost Function
* L # layers
* K # classifications
  Note if binary, then K = 1  
* $J(\theta) = - \frac{1}{m}[\sum\limits_{i = 1}^m\sum\limits_{k = 1}^Ky_k^{(i)}log(f(x^{(i)})_k) + (1 - y_k^{(i)})log(1 - f(x^{(i)})_k)] + \frac{\lambda}{2m}\sum\limits_{l = 1}^{l-1}\sum\limits_{i = 1}^ {s_l}\sum\limits_{j = 1}^ {s_l +1}\theta_{ji}^{(l)}$

## Back Propagation
Used to find the partial derivate for gradient descent

Intitution: $\delta^{(l)}_j$ is the 'error' in node j of layer l
* $\delta^{(L)} = a^{(L)} - y$
* Repeat $\delta^{(i)}$ :   
  $\delta^{(L-i)} = (\theta^{(L-i)})^T\delta^{(L-i + 1)}\cdot g^{'}(z^{L-i})$  
   = $(\theta^{(L-i)})^T\delta^{(L-i + 1)} \odot a^{(L-i)} \odot (1-a^{(L-i)})$  
   where $\odot$ is the element wise multiplication  
   This simple version is by complex derivation  
* Psecudo code: (Vectorizaton)  
Input: ${(x_1, y_1), (x_2, y_2), ..., (x_m, y_m)}$  
  * For $l$ in range(L-1):  
	* set $\Delta^{(l)} = 0$
  * For $i$ in range(m + 1):  
	* set $a^{(1)} = x^{(i)}$  
	* For $l$ in range(L-1):  
 		* getFP($a^{(l)}$) (from 1 to $l - 1$)
    	* get$\delta$() (from $L$ to 2)
  	* $\Delta^{(l)}$ += $\delta^{(l+1)}(a^{(l)})^T$
* If j $\neq$ 0:  
   $D^{(l)} = \frac{1}{m}\Delta^{(l)} + \lambda\theta^{(l)}$ 
* otherwise, $\lambda = 0$
* Using Gradient Descent to compute weights
  $\theta_l$ -= $\alpha D^{(l)}$

Note: $\delta_j^{(i)}$ is the linear combination of the $\delta_j^{(i + 1)}$ which connected with it 

### Gradient Checking
When implementating greadient descent, it might seems working well and desreasing which gives you a 'correct' answer, but there might be sutble problems.

Using gradient checking would detect this.

calculate $\frac{\partial J(\theta)}{\partial \theta^{(i)}} \approx \frac{J(\theta1, \theta2, ... \theta_i + \epsilon, ..., \theta_n) - J(\theta1, \theta2, ... \theta_i - \epsilon, ..., \theta_n)}{2\epsilon}$ 

### Weights Initialization
If you initialize $\theta_s$ all be zeros, then the hidden layers are going to compute the same feature, highly redundant
```python
# Randomly
weights = np.random.randn(layer_size[l],layer_size[l-1])*0.01
# he et al
weights = np.random.randn(layer_size[l],layer_size[l-1])*np.sqrt(2/layer_size[l-1])

## Credit: https://towardsdatascience.com/random-initialization-for-neural-networks-a-thing-of-the-past-bfcdd806bf9e
```
---
# Evaluating + Debugging Algorithm
*Diagonstic*: A test to gain what is/isn't working with a learning algorithm, gain guidance as to how to best improve performance

## Evaluating 
1. shuffle and randomly reorder data, and generally get 70% for training, 30% fot testing
2. compute $J(\theta)$ and _Test error_
Test error: $\frac{sum-errors}{size}$ 

## Model Selection
1. 60% training; 20% cross validation; 20% test   
2. Optimize $\theta$ for each polynomial
3. using **cross-validation** to get coorsponding and $\theta$ we got to compute $J_{cv}$. Pick the d with least $J_{cv}$
4. Using test data to estimate generalization error with polynomial with degree d
(Potential problem: polynomial with degree d might give the over optiminstic result) 

## Reasons
As degree increase, Training error goes down, Cross-validation error goes down (just fit) then goes up(overfitting)

* Underfit(High Bias): 
  1. $J_{train}(\theta)$ is high 
  2. $J_{cv}(\theta) \ge J_{train}(\theta)$ 
* Overfit (High Variance):
  1. $J_{train}(\theta)$ is low
  2. $J_{cv}(\theta) \gg J_{train}(\theta)$ 

## Regularization
1. Create a set of models with different degrees or any other variants.
2. Try $\lambda$ from 0 to some_number(e.g. 10)
3. Iterate through the $\lambda$s and for each $\lambda$ go through all the models to learn some $\Theta$.
4. For each $\lambda$, using the $\Theta$ we got to compute $J_{cv}(\Theta)$ without regularization or $\lambda$ = 0.
5. Select the $\theta$ whose $J_{cv}(\Theta)$ is the minmium, since it is a convex function
6. Apply on $J_{test}(\Theta)$
  
(Note that the $\lambda-Error$ graph is just the opposite of y-axis with $degree-Error$ graph)

## Learning Curve
* High Bias
  * $J_{train}$ is high
  * $J_{cv}$ is still high even if data feed is more
  * $J_{train} \approx J_{cv}$
* High Variance 
  * $J_{train}$ is low
  * $J_{cv}$ is lower when data feed is more
  * $J_{train} < J_{cv}$ There is a gap  
    When training data is more, gap is smaller

## Debugging
* High Bias
  * Add features
  * Add polynomial features
  * Decrease $\lambda$
* High variance
  * Getting more training data
  * Reduce features
  * Increase $\lambda$

NN:
* Fewer parameters: prone to underfit
* More parameters: prone to overfit  
  * One single layer, many parameters is default  
  * Compare to underfit, recommend overfit + regularization

**Premature optimization**:
Let the evidence guide your decisions, rather than use gut feeling

# ML Sysmetic Design
1. Design a simple and dirty algorithm to sort the problem (generally 24h) (using simple evaluation above)
2. Plot learning curve and debug 
(using simple debugging above)
3. Manually examine errors on the cross-validation set, try to spot trend and modify your algorithm to repeat this process

## Error metrics for Skewed Data
* Skewed Data:
The propotion for positive / negative set is too small (e.g. 0.5%)  
* So that the accuracy number would not help us to judge the algorithm
* The Error metrics (precision + recall) would give us insight to judge

|Predicted\Actual|1|0|
|:--|:--:|--:|
|1|True Positive|False Positive|    
|0|False Negative|True Negative|    

Precison/Recall
* Precision
  $\frac{TP}{Predict_1}$
* Recall
  $\frac{TP}{Actual_1}$


# SVM
**Supporting vector machine**  
![](https://miro.medium.com/max/2000/1*qfZnRoVp-A0a4jMLwTBL6g.png)
Two loss functions break into 2 lines  
![](https://miro.medium.com/max/1400/1*AndL5FYso8ad7LSrie8zoA.png)
Original: $A + \lambda B$, to $CA + B$ (could understand C plays a role of $\frac{1}{\lambda}$)
![](https://miro.medium.com/max/2000/1*h5QqCdm48pt84bazOPvExA.png)


## Linear SVM
Generally we have multiple ways to classify, SVM will find the decision boundary with the largest margin **Large Margin Classifier**

Support vector is a sample that is incorrectly classified or a sample close to a boundary

Since that using normal sigmoid function to classify, the boundary is 0, but why SVM boundary is 1?
SVM pentalizes the incorrect predictions and the data close to the boundary. So that removing non-supporting vector won't affect.

C is a parameter of sensitivity, using large C would be more sensitive about outliers(some particular points), also the tolerance of misclassfication, control width of margin, small C would give large margin, possibly more data would violate margin   
(Note that C's role with $\lambda$, so that there is a trade-off between these)

![](https://miro.medium.com/max/2000/1*DSHIKH8TiiN1FeTKtfDzrQ.png)

## Maths theory
2 vectors $\vec u$, $\vec v$, $p$ is the projecton from one to the other

$\vec u^T \vec v = p * ||u||$

Since $\Theta$ and $x$ are vectors, using the theorem above,

* $\Theta^Tx^{(i)} = p^{(i)} * ||\Theta||$
* $\Theta$ is normal to decision boundary 
* We could like to min CA + B, so min both, for B: $\frac{1}{2}\sum\limits_{j=1}^n\theta_j^2 = \frac{1}{2}||\Theta||^2$
* Would like to max $p^{(i)}$ so to min $||\Theta||$

## Non-Linear
How to classify the non-linear data using SVM?

Using similarity function $f_i$ to denote data $x_i$ with landmarks $l^{(i)}$
![](https://miro.medium.com/max/1400/1*LL3sMir4miRFIsALaJITiw.png)
There are many choices for $f_i$, generally using **Gaussian Kernel**
(Euclidean distance), where $\sigma$ represents the smoothness
![](https://miro.medium.com/max/1400/1*NGScLYV3sNwD3kZim5JYlA.png)

 $f_{(i)} \approx \left\{ \begin{array}{rcl} 1 & x \approx l^{(i)}\\ 0 &  otherwise\end{array}\right.$  

 $\Theta$ and $\sigma$ are inputs, $\sigma$ value $\uparrow$, range of data of plot $\uparrow$

 result = $\theta_o + \theta_1f_1 + \theta_2f_2 + ... + \theta_nf_n$
 * If result $\ge 0$, predict = 1
 * otherwise, predict = 0

But how to choose $l^{(i)}$ and other parameters?
