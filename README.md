# ConvolutionalAnalysisOperatorLearning.jl
Learn multi-dimensional convolutional analysis operators
(i.e., sparsifying filters) from data.

## Installation

Install using Julia's Pkg REPL-mode
(hitting `]` as the first character of the command prompt):
```
(v1.0) pkg> add https://github.com/dahong67/ConvolutionalAnalysisOperatorLearning.jl
```

## Example usage

Create `10` random `100 x 50` images
```julia
julia> x = [randn(100,50) for _ in 1:10]
10-element Array{Array{Float64,2},1}:
 [-0.5091539349939715 1.7386490064501035 … -0.10434839673212205 1.4913718856758826; -0.35225316678828295 -0.8815962907259653 … -1.6677487621776104 -1.449212618073853; … ; -1.0848353411048959 -0.9587418527517606 … 0.09690836554809144 -0.43486276266634505; -0.887554388134407 -0.03434441237942858 … -0.4621474853155247 -0.7715344824904893]
[...]
```

Create initial `3 x 3` filters, e.g., using DCT,
```julia
julia> using FFTW, LinearAlgebra

julia> H0 = dct(Matrix(I,9,9),1)'/sqrt(9)
9×9 Array{Float64,2}:
 0.111111   0.154748      0.147658    0.136083     …   0.0785674   0.0537433   0.0272862
 0.111111   0.136083      0.0785674  -1.74455e-17     -0.157135   -0.136083   -0.0785674
 0.111111   0.101004     -0.0272862  -0.136083         0.0785674   0.154748    0.120372
 0.111111   0.0537433    -0.120372   -0.136083         0.0785674  -0.101004   -0.147658
 0.111111   4.36137e-18  -0.157135   -1.74455e-17     -0.157135    0.0         0.157135
 0.111111  -0.0537433    -0.120372    0.136083     …   0.0785674   0.101004   -0.147658
 0.111111  -0.101004     -0.0272862   0.136083         0.0785674  -0.154748    0.120372
 0.111111  -0.136083      0.0785674  -1.74455e-17     -0.157135    0.136083   -0.0785674
 0.111111  -0.154748      0.147658   -0.136083         0.0785674  -0.0537433   0.0272862

julia> h0 = [reshape(H0[:,k],3,3) for k in 1:9]
9-element Array{Array{Float64,2},1}:
 [1.6974804288204 -0.28533505650798413 -0.3468237944985989; 2.0573445402844377 0.5555863463318774 0.29019803930983806; -0.31306149537884065 0.18390961335489334 0.08856921972995008]
[...]
```

Run CAOL
```julia
julia> using ConvolutionalAnalysisOperatorLearning

julia> λ = 1e-4      # regularization parameter
0.0001

julia> CAOL(x,h0,λ,maxiters=30)
9-element Array{Array{Float64,2},1}:
 [0.110998600520466 0.11108306994831485 0.11108677706357252; 0.11128303442353249 0.11124729130532747 0.11102515586383724; 0.11099786500962032 0.11099544188717123 0.11128220146007992]
[...]
```

The output has `9` filters of size `3 x 3`.

**TODO: Clean-up and add documentation**

## Optimization problem and algorithm

`CAOL` attempts to minimize the following function
(written in partly Julia notation)
```julia
sum(1/2*norm(x[l]✪h[k] - z[l,k])^2 + λ*norm(z[l,k],0) for k in 1:K, l in 1:L)
```
with respect to `z[l,k]` and `H = [vec(h[1]) ... vec(h[K])]` where
+ `H` is constrained to have (scaled) orthonormal columns,
  i.e., `H'H == (1/R)*I`, where `R = size(H,1)`
+ `✪` denotes circular correlation, namely `xl ✪ hk` is an `OffsetArray`
  indexed along each dimension of size `n` by lag in `0:n-1`,
  where (for the one-dimensional case) the `i`th lag is
  ```julia
  (xl ✪ hk)[i] = sum(xpadl[j+i]*hk[j] for j in 1:R)
  ```
  with `xpadl = padarray(xl,Pad(:circular, [...]))`
  being a circularly padded version of `xl`.
  This calculation is accomplished in-place with `ImageFiltering.jl` via
  ```julia
  imfilter!(out,xpadl,(hk,),NoPad(),Algorithm.FIR())
  ```
  where `out` has axes of the form `0:n-1` in each dimension.

The optimization is carried out via alternating minimization.

1. **Sparse code update.**
  The objective is minimized with respect to `z[l,k]`
  by hard-thresholding `x[l]✪h[k]` as follows
  ```julia
  imfilter!(z[l,k],xpad[l],(h[k],),NoPad(),Algorithm.FIR())
  z[l,k] .= hard.(z[l,k],sqrt(2λ))
  ```
  It turns out that only an accumulated version of `z[l,k]` is needed,
  so the code only stores one at a time,
  reusing the memory across `l` and `k` for efficiency.

2. **Filter update.**
  Minimizing the objective with respect to `H` turns out
  to be a Procrustes problem and is solved by
  the polar factor of
  ```julia
  sum([XPADL'z[l,1] ... XPADL'z[l,K]] for l in 1:L)
  ```
  where `XPADL` is the matrix such that `XPADL * h == xl ✪ h`.
  In one dimension,
  ```julia
  XPADL = [
  xl[1] xl[2] ... xl[R-1] xl[R];
  xl[2] xl[3] ... xl[R]   xl[1];
  ...
  xl[n] xl[1] ... xl[R-2] xl[R-1]
  ]
  ```
  yielding
  ```julia
  XPADL' = [
  xl[1]   xl[2] ... xl[n];
  xl[2]   xl[3] ... xl[1];
  ...
  xl[R-1] xl[R] ... xl[R-2]
  xl[R]   xl[1] ... xl[R-1]
  ]
  ```
  so `XPADL'z` is another circular correlation
  and can be accomplished in-place with `ImageFiltering.jl` via
  ```julia
  imfilter!(out,xpad[l],(z[l,k],),NoPad(),Algorithm.FIR())
  ```
  by having `out` be indexed from `1:r` in each dimension
  where `r` is the size of the filters along that dimension.
  Note that this convenient property is for correlation, and not convolution.


**TODO:**
double check the derivation (especially `R`s and `K`s, and dimension > 1),
write up the version for handcrafted filters,
and put into docs with LaTeX.

## Relevant papers

[1] Il Yong Chun and Jeffrey A. Fessler, "Convolutional analysis operator learning: Acceleration and convergence," submitted, Jan. 2019.
[Online] Available: http://arxiv.org/abs/1802.05584

[2] Caroline Crockett, David Hong, Il Yong Chun, Jeffrey A. Fessler, "Incorporating handcrafted filters in convolutional analysis operator learning for ill-posed inverse problems," Accepted to CAMSAP 2019.
