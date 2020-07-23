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