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
julia> h0 = [reshape(H0[:,k],3,3) for k in 1:9]
9-element Array{Array{Float64,2},1}:
 [1.6974804288204 -0.28533505650798413 -0.3468237944985989; 2.0573445402844377 0.5555863463318774 0.29019803930983806; -0.31306149537884065 0.18390961335489334 0.08856921972995008]
[...]
```

Run CAOL
```julia
julia> λ = 1e-4   # regularization parameter
julia> CAOL(x,h0,λ,maxiters=30)
```

**TODO: Clean-up and add documentation**

## Relevant papers

[1] Il Yong Chun and Jeffrey A. Fessler, "Convolutional analysis operator learning: Acceleration and convergence," submitted, Jan. 2019.
[Online] Available: http://arxiv.org/abs/1802.05584

[2] Caroline Crockett, David Hong, Il Yong Chun, Jeffrey A. Fessler, "Incorporating handcrafted filters in convolutional analysis operator learning for ill-posed inverse problems," Accepted to CAMSAP 2019.
