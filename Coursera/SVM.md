# SVM (pictures credit: https://towardsdatascience.com/@shuyuluo)
**Supporting vector machine**  
![](https://miro.medium.com/max/2000/1*qfZnRoVp-A0a4jMLwTBL6g.png)
Two loss functions break into 2 lines  
![](https://miro.medium.com/max/1400/1*AndL5FYso8ad7LSrie8zoA.png)
Original: $A + \lambda B$, to $CA + B$ (could understand C plays a role of $\frac{1}{\lambda}$)
![](https://miro.medium.com/max/2000/1*h5QqCdm48pt84bazOPvExA.png)


## Linear SVM (Linear Kernel)
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

 $\sigma$ is input, value $\uparrow$, range of data of plot $\uparrow$

 result = $\theta_o + \theta_1f_1 + \theta_2f_2 + ... + \theta_nf_m$
 * If result $\ge 0$, predict = 1
 * otherwise, predict = 0
  
After feeding data, we can draw the decision boundary

But how to choose $l^{(i)}$ and other parameters?

1. Using feacture scaling to scale the data
2. Given data set with size m, for each $x^{(i)}$, let $l^{(i)} = x^{(i)}$
3. Compute $\vec f^{(i)} = \begin{bmatrix} 
                              f_0^{(i)} \\  
                              f_1^{(i)} \\ 
                              f_2^{(i)} \\
                              ...       \\
                              f_i^{(i)} \\
                              ...       \\
                              f_m^{(i)}  
                            \end{bmatrix}$
                         = $\begin{bmatrix} 
                              1 \\  
                              sim(x^{(i)}, l^{(1)}) \\
                              sim(x^{(i)}, l^{(2)}) \\
                              ...                   \\
                              sim(x^{(i)}, l^{(i)}) = 1 \\ 
                              ...       \\
                              sim(x^{(i)}, l^{(m)})  
                            \end{bmatrix}$
4. Training:
   ![](https://miro.medium.com/max/2000/1*ssIbIMbFrpireIvOq_N_IA.png)
5. Debug:
    * Overfit: 
      * Increase $\sigma^2$
      * Decrease $C$
    * Underfit:
      * Decrease $\sigma^2$
      * Increase $C$

Other kernels:
* Polynomial Kernel  
  $sim = (x^{(i)^T}l^{(i)} + c)^d$
* More esoteric
  * String kernel
  * chi-square kernel
  * histogram kernel
  * intersection kernel
  * ... 
## Choices between Logistic Regression & SVMs
* n is large(relatively to m):  
  $n \ge m$, using Logistic Regression / Linear Kernel
* n is small, m is intermediate:   
  e.g. (n: 1 - 1000, m: 10 - 10,000)  
  using SVMs with Gaussian Kernel
* n is small, m is large:  
  e.g. (n: 1 - 1000, m: 50,000+)
  Create/add more features, and using logistic Regression / Linear Kernel