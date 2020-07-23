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
