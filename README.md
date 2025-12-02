# Coordinate Update for CFN Model

This github repo contains code used to generate simulations for a research project on the CFN model. 

The CFN model is a Markov model on trees. Given a tree $T = (V,E)$ and a paramter $\boldsymbol{\theta}\in [-1,1]^{E}$, a spin $\sigma = (\sigma_v;v\in T)$ is generated as follows:
* Pick a vertex $v\in T$ arbitrariliy (this does not matter).
* Assign a spin $\sigma_v\in \{-1,1\}$ with probability $${P}_{T,\boldsymbol{\theta}}( \sigma_v = 1)={P}_{T,\boldsymbol{\theta}}(\sigma_v = -1)=\frac{1}{2}.$$
* Accross any edge $e = \{x,y\}\in E$, we independtely flip the signal with probability 
${P}_{T,\boldsymbol{\theta}}(\sigma_x = \sigma_y | \sigma_y) = \frac{1+\theta_e}{2}\\
{P}_{T,\boldsymbol{\theta}}(\sigma_x =- \sigma_y | \sigma_y) = \frac{1-\theta_e}{2}.$

The goal of the project was to understand the coordinate update mechanism used to estimate the parameter $\boldsymbol{\theta}$ from repeated obeservations of the spins at the leaves $\sigma|_L = (\sigma_v;v\in L(T))$ for a known unrooted binary tree $T$. 


The theoretical arguments are contained in a sequence of three papers:
* [Liklihood-based root state reconstruction of binary latent model on a tree: sensitivity to parameters and applications](https://arxiv.org/abs/2501.13208)
* [Likelihood landscape of binary latent model on a tree](https://arxiv.org/abs/2501.17622)
* **Published at ICML:** [Sample complexity of branch-length estimation by maximum likelihood](https://icml.cc/virtual/2025/poster/46053) 

Key to this analysis was the magnetization for a *rooted* binary tree $T_u$ with root $u$. The magnetization 
$Z_u = Z_u^{T_u, \widehat{\boldsymbol{\theta}}}(\sigma_{{L_u}}) = {E}_{\widehat{\boldsymbol{\theta}}}[\sigma_u | \sigma_{{L_u}}]$ is the expected spin at the vertex $u$ (computed with the parameter ${\widehat{\boldsymbol{\theta}}}$) given the spins at the leaves $L_u$ generated under the true distribution $\sigma\sim {P}_{T_u,\boldsymbol{\theta}^*}$. 

It turns out that when estimate $\widehat{\boldsymbol{\theta}}$ is close to true parameter $\boldsymbol{\theta}^*$ then $Z_u \approx \sigma_u$. Namely, if 
$\theta_e^*, \widehat{\theta}_e\in [1-\frac{1}{2}\delta, 1-\frac{1}{4}\delta]\qquad \forall e\in E$
then
${P}_{\boldsymbol{\theta}^*}(\sigma_u Z_u^{\widehat{\boldsymbol{\theta}}} \ge 1-\frac{121}{5} \delta^2) \ge 1-\frac{7}{2}\delta$ for all trees $T$ and all $\delta\in(0, 1/924]$. 

Empirically, this appears to be the case for larger $\delta$ as well. Below is the empirical histogram of of $\sigma_u Z_u$ for 50 thousand simulations on 10 uniformly generated 1000 leaf trees (half a million simulations in total) with $\delta = 0.1$.
![Magnetization picture](/images/image1/image1.png)

Here $90.8\%$ of the samples lie above the value $1-\frac{121}{5}\delta^2 = 0.758$ when the theoretical bound (for much smaller $\delta$) is $65\%$.