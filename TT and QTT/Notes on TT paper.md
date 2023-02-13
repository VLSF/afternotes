# Introduction
I. V. Oseledets introduced the TT format in the article Tensor-Train Decomposition https://doi.org/10.1137/090752286. Here, I summarise a theoretical part of this paper (the existence of TT and constructive results on how to build TT with SVD and so on) as I understand it.

For a $d$ dimensional tensor, TT decomposition reads
$$A(i_1, \dots, i_d) = \sum_{\alpha_0,\dots,\alpha_{d+1}} G_{1}(\alpha_0,i_1,\alpha_1)\cdots G_{1}(\alpha_{d},i_{d},\alpha_{d+1}),\,\alpha_i=1,\dots r_i,\,r_0=r_{d+1} = 1,$$
so we only need to store cores (carriages) $G_i$ which requires $\leq d \left(\max(r_i)\right)^2$ floats.
# Existence
It is easy to show that tensor train decomposition exists and find TT ranks $r_k$ with SVD.
The main techniques here are
1. Unfolding
2. SVD
The unfolding of a tensor is a standard reshape operation (`A.reshape(m, n)`), where tensor indices are mapped on indices for matrix rows and columns. Symbolically this is written as follows
$$A_k = A(i_1,\dots, i_k;i_{k+1},\dots,i_{d}),$$
i.e., indices before semicolon mark rows and indices after mark column of matrix $A_k$.

**Theorem 2.1**
If each unfolding of $A_k$ has rank $r_k$, there exists a TT decomposition with $k$-th TT-ranks $\leq r_k$.
**Proof**
The proof is by induction.
Consider the first unfolding $A_1\in\mathbb{R}^{n_1\times m_1}$. Since this is an ordinary matrix, it can be presented as a product of two matrices (e.g., using SVD) with contraction dimension of size $r_1$, i.e.,
$$A_1 = \underset{n_1\times r_1}{U}\,\underset{r_1\times m_1}{V^{T}}.$$
If we restore indices, the identity above reads
$$A(i_1,\dots,i_k) = \sum_{\alpha}U(i_1,\alpha) V(\alpha,i_2,\dots, i_d),$$
so $U(i_1, \alpha)$ is a legitimate first TT core. To proceed, we need to show that tensor $V$ has the same property that all its unfoldings $k=2,\dots, d$ have ranks $r_k$.
To show that we use
$$V = A_1^{T}U \left(U^T U\right)^{-1} = A_1^{T}W \Leftrightarrow V(\alpha,i_2,\dots, i_d) = \sum_{i_1}A(i_1, \dots, i_d)W(i_1,\alpha),$$
and the fact that $k$-th unfoldings of $A$ has rank $r_k$, that is,
$$A(i_1, \dots, i_d) = \sum_{\alpha = 1}^{r_k} X(i_1, \dots, i_k, \alpha) Y(\alpha, i_{k+1},\dots, i_{d}).$$
Combining these two facts, we obtain
$$
V(\alpha, i_2, \dots, i_d) = \sum_{\beta}\widetilde{X}(\alpha, i_2, \dots, i_k, \beta) Y(\beta, i_{k+1},\dots, i_{d}),
$$
so indeed $k$-th unfolding of $V$ has rank $r_k$.
To continue the induction process, we reshape $V$ and again consider the first unfolding
$$V(\alpha i_2, i_3, \dots, i_d) = \sum_{\beta} U(\alpha i_2, \beta)V^{'}(\beta, t_3,\dots, i_d) = \sum_{\beta} G(\alpha, i_2, \beta) V^{'}(\beta, t_3,\dots, i_d).
$$
The proof is constructive, so it is not hard to deduce an SVD-based algorithm for the sequential computation of tensor cores.
# Approximate TT

**Theorem 2.2.**
Suppose that unfoldings are only approximately low-rank, that is, 
$$A_k = R_k + E_k,\,\text{rank }R_k = r_k,\,\left\|E_k\right\|_F\leq \epsilon_k.$$

In this case, TT-SVD algorithm (implicitly given in the existence part) constructs approximate tensor $B$ such, that
$$\left\| A - B\right\|_F \leq \sqrt{\sum_{k=1}^{d-1}\epsilon_k^2}.$$

**Proof**
This proof is by induction in the number of dimensions.
For $d=2$, everything is known since TT-SVD is SVD.
Consider the first unfolding for $d > 2$ and use SVD to find
$$A_1 = U_1 B_1 + E_1.$$
Now, tensor $B_1$ (considered as tensor with $d$ dimensions with first and second indices merged) will be approximated by TT-SVD with a different tensor $\hat{B}_1$. So, for the distance between the original tensor and the approximate one, we get
$$\left\|A - B\right\|_F^2 = \left\|A_1 - U_1\hat{B}\right\|_F^2 = \left\|A_1 - U_1\hat{B}_1 - U_1 B +U_1 B_1\right\|_F^2 = \left\|E_1 + U_1(B_1 - \hat{B}_1)\right\|_F^2
$$
$$= \left\|E_1\right\|_F^2 + \left\|B_1 - \hat{B}_1\right\|_{F}^2,$$
where we used $U_1^T U_1 = I$ and $U_1^T E_1 = 0$, which is direct consequence of SVD used to construct $U_1$, $B_1$ and $E_1$.
We can proceed by induction if $B_1$ has the same property that $A$, i.e., its remaining unfoldings have approximate rank $r_k$. Let's show that this is the case.
Observe that
$$B_1 = U_1^T A_1 \Leftrightarrow B(\alpha i_{2},\dots i_{d}) = \sum_{i_1} U_1(\alpha, i_1) A(i_1, \dots, i_d).$$
Next, we use that each unfolding of $A_k$ is approximately of rank $r_k$, i.e.,
$$A(i_1, \dots, i_d) = \sum_{\beta = 1}^{r_k} H(i_1, \dots, i_k, \beta) D(\beta, i_{k+1},\dots, i_{d}) + E_k(i_1, \dots, i_d),\,\left\|E_k\right\|_F \leq \epsilon_k.$$
Inserting this identity into the definition of $B_1$, we get
$$B(\alpha i_{2},\dots i_{d}) = \sum_{i_1} U_1(\alpha, i_1) \left(\sum_{\beta = 1}^{r_k} H(i_1, \dots, i_k, \beta) D(\beta, i_{k+1},\dots, i_{d}) + E_k(i_1, \dots, i_d)\right).$$
The first sum is of the correct structure, so we only need to show the Frobenius norm of the reminder $\leq \epsilon_{k}$. To show that, we consider the Frobenius norm of the first unfolding of $E_{k}$. This gives us
$$\left\|\sum_{i_1} U_1(\alpha, i_1)E_k(i_1, \dots, i_d)\right\|_F = \left\|U_1 \left(E_k\right)_1\right\|_F = \left\|\left(E_k\right)_1\right\|_F = \left\|E_k(i_1, \dots, i_d)\right\|_F = \epsilon_k.$$
Since $B_1$ has the same properties as $A$ but is a tensor with $d-1$ dimensions, we are done with the induction step and have
$$\left\|A - B\right\|_F^2 \leq \sum_{i=1}^{d-1}\epsilon_i^2.$$
# Algorithm
The proofs wer constructive, so TT-SVD algorithm is easy to recover. In the simplest form when we can store the whole tensor in memory, it can be realized with SVD and reshapes.

Require: tensor $A$ having shapes $\left(n_1, \dots, n_d\right)$, required relative accuracy $\epsilon$
Init: find truncation parameters $\delta = \epsilon \left\|A\right\|_F \big/ \sqrt{d-1}$, set $r_0 = 1$, $C = A$
for $k = 1,\dots, d-1$:
	$C = C.\text{reshape}(r_{k-1}n_k, |C| - r_{k-1}n_k)$
	Use SVD to find low-rank apparoximation $C = U S V^{T} + E,\,\left\|E\right\|_F \leq \delta$ 
	Set $r_k = \text{rank}(U)$
	Form TT core $G_k = U.\text{reshape}(r_{k-1}, n_k, r_{k})$
	Set $C = S V^{T} = U^T C$ 
$G_d = C$

# Operations in TT format
To do

## Addition

## Hadamard product

## Scalar product

## Matrix product

