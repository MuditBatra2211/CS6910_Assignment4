# CS6910_Assignment4
# Authors: EE19S015: Mudit Batra, EE20S008: Vrunda Sukhadia
RBM algorithm with Block Gibbs Sampling and Contrastive Divergence


## Description of Files
1. To replicate results with different algorithms refer "RBM_main.py"
2. To replicate plots and Data visulations refer "RBM_tsne_m64_plots.py"


### "RBM_main.py"

If you want to replicate results for particular configuration select "RBM_main.py" file. We have seperatley defined all the hyperparameter
configuration in the "setting hyperparameter values" so you can directly change the parameters in the "setting hyperparameter values" using below description:

```
method_of_rbm = 'Contrastive Divergence'   # put 'Gibbs sampling' for block sampling algorithm and 'Contrastive Divergence' for cd algorithm.
lr = 0.01               #[0.001, 0.01, 0.1]
hidden_dim = 256        #[64, 128, 256]
steps_Gibbs = 10        #for contrastive divergence [1,5,10]
max_epochs = 1          # 100 for Contrastive Divergence and 10 for Block Gibbs Sampling
k = 200                 # markov chain runs for gibbs sampling [100, 200, 300]
r = 10                  # after convergence runs for gibbs sampling [10, 20, 30]
```
All the Parameters that can be tuned are provided in comments<br/>


### "RBM_tsne_m64_plots.py"

If you want to replicate plots for particular configuration select "RBM_tsne_m64_plots.py" file. We have seperatley defined all the hyperparameter
configuration in the "setting hyperparameter values" so you can directly change the parameters in the "setting hyperparameter values" using below description:

```
method_of_rbm = 'Contrastive Divergence'   # put 'Gibbs sampling' for block sampling algorithm and 'Contrastive Divergence' for cd algorithm.
lr = 0.01               #[0.001, 0.01, 0.1]
hidden_dim = 256        #[64, 128, 256]
steps_Gibbs = 10        #for contrastive divergence [1,5,10]

#Note: Enter that epoch Number when training gets saturated
max_epochs = 2          # Put Stable number when Training gets saturated

k = 200                 # markov chain runs for gibbs sampling [100, 200, 300]
r = 10                  # after convergence runs for gibbs sampling [10, 20, 30]
```
All the Parameters that can be tuned are provided in comments<br/>

### "Results"
In this folder all results have been submitted.<br/>
1. Tsne without pca plot.png: TSNE plot genreated for MNIST data.
2. Tsne with pca 50 plot.png : TSNE plot generated for MNIST data, by first applying PCA on it.
3. reconstructed data visualtion within 1 epoch.png : 64 visulaization of a reconstructed data over 1 epoch.
4. reconstructed data visualtion within 40 epoch.png : 64 visulaization of a reconstructed data over 40 epoch.

### [Check out our project workspace](https://wandb.ai/vrunda/CS6910_Assignment_4?workspace=user-vrunda)
### [Check out project detailed report](https://wandb.ai/vrunda/CS6910_Assignment_4/reports/CS6910-Assignment-4--Vmlldzo3MjI5MjA)
