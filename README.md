# Vector-Form BFGS Algorithm in PyTorch
## Repository for final year project: Enhancing the backpropagration algorithm of CNN using Optimal Memory BFGS algorithm.

### User Instruction
```
git clone https://github.com/vincent-teh/Vectorform_BFGS.git
Pending to add more instrution
```


### Class Relationship
```mermaid
flowchart TD
    A[Optimizer (Native PyTorch)] --> B[BfgsBaseOptimizer];
    B[BfgsBaseOptimizer] --> C[ConjGrad];
    B[BfgsBaseOptimizer] --> D[MLBFGS];
    B[BfgsBaseOptimizer] --> E[VMBFGS];
    B[BfgsBaseOptimizer] --> F[OMBFGS];

```

### Abstract
For years, Stochastic Gradient Descent (SGD) has been the predominant training algorithm for AI model development. With the exception of the LBFGS algorithm, all algorithms in PyTorch modules are variations of SGD. Having an ineffective training algorithm for a state-of-the-art model is akin to employing a suboptimal teaching method for a highly intelligent child - it costs extensive time for training, and the SGD algorithm resembles this challenge. This is primarily due to the need for learning rate tuning, which is determined heuristically. Despite this, SGD remains the preferred option due to its consistent convergence. Consequently, my research delved into exploring the potential of the Vector-form Quasi-Newton’s method for training Convolutional Neural Networks. In the 20th century, second-order methods including Quasi-Newton’s method were highly anticipated in AI research. However, as we entered the 21st century, the focus shifted from innovating AI to building applications with AI. This change resulted from the introduction of GPUs, which proved effective in training AI even using existing optimizers, including SGD, albeit at high cost. Therefore, this study implemented three Vector-Form BFGS algorithms, a type of Quasi-Newton's method, through a custom-written optimizer PyTorch modules. In contrast to the classic BFGS algorithm, which demands O(n²) memory complexity to approximate the inverse of the Hessian matrix, these solutions directly compute the new search direction in vector format. As a result, they require only linear-scale memory while ensuring an effective search direction. In addition, instead of the trial-and-error approach found in SGD, learning rate tuning was replaced with line search. Finally, the results of this study showcased a 30% reduction in training time compared to SGD. Besides, particularly noteworthy is the LBFGS, another variant of the BFGS provided by PyTorch, failed to converge in this study.

### Results
example results


### References
@article{paszke2017automatic,
  title={Automatic differentiation in PyTorch},
  author={Paszke, Adam and Gross, Sam and Chintala, Soumith and Chanan, Gregory and Yang, Edward and DeVito, Zachary and Lin, Zeming and Desmaison, Alban and Antiga, Luca and Lerer, Adam},
  year={2017}
}

@inproceedings{mcloone2002memory,
  title={A memory optimal BFGS neural network training algorithm},
  author={McLoone, Se{\'a}n F and Asirvadam, Vijanth Sagayan and Irwin, George W},
  booktitle={Proceedings of the 2002 International Joint Conference on Neural Networks. IJCNN'02 (Cat. No. 02CH37290)},
  volume={1},
  pages={513--518},
  year={2002},
  organization={IEEE}
}
