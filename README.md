# ConvolutionalOperatorLearning.jl

[![Build Status](https://travis-ci.org/dahong67/ConvolutionalOperatorLearning.jl.svg?branch=master)](https://travis-ci.org/dahong67/ConvolutionalOperatorLearning.jl)
[![codecov](https://codecov.io/gh/dahong67/ConvolutionalOperatorLearning.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/dahong67/ConvolutionalOperatorLearning.jl)

Learn multi-dimensional convolutional analysis operators
(i.e., sparsifying filters) from data.
Based on the papers:

> Il Yong Chun and Jeffrey A. Fessler, "Convolutional analysis operator learning: Acceleration and convergence," IEEE Trans. Image Process. (to appear), Aug. 2019.
[Online] Available: http://arxiv.org/abs/1802.05584
>
> Caroline Crockett, David Hong, Il Yong Chun, Jeffrey A. Fessler, "Incorporating handcrafted filters in convolutional analysis operator learning for ill-posed inverse problems," Accepted to IEEE CAMSAP 2019.

## Installation

Install using Julia's Pkg REPL-mode
(hitting `]` as the first character of the command prompt):
```
(v1.1) pkg> add ConvolutionalOperatorLearning
```

## Example usage

Create `10` random `100 x 50` images
```julia
julia> x = [randn(100,50) for _ in 1:10]
10-element Array{Array{Float64,2},1}:
 [-0.06103119834541305 0.17968724639365347 … -0.004269304417885406 -0.05473288638503366; -0.029082836823963076 -2.234356751701322 … 0.5306985197349057 0.5196510275200548; … ; 0.32864611768808494 -0.16509219769339606 … -0.7188306906351658 -1.3351708287182686; -0.8460760611398502 0.6741543032561107 … -1.7931435443456931 2.072398627809654]
 [-1.0823675555226044 1.8113983682279433 … -0.25814439615462886 -0.11342027638737794; -2.6512204811658844 0.6056278086922918 … 0.49428954950393583 0.9291513037613385; … ; 1.528072430975617 1.1576527001448074 … -0.13353151521165682 1.4552503091473545; -1.143499264158193 0.17686952628772404 … 0.5386755547858878 0.11253501428137602]       
 [-0.9748965681300071 -0.7942177416850443 … 0.647652241602665 -0.26162723540242966; -1.550320939343062 0.17934164372628594 … 0.07070502518981514 -0.3398872009535432; … ; -1.2000135265342693 -1.3040261404206082 … -0.537817304957513 -0.3194718348301661; -1.942367365002938 -0.8345181323639871 … -0.49691601543708913 0.13452223196414928]    
 [0.14109450766686116 -1.6227669465434267 … -0.14074338327795177 1.7247670372829123; -0.4530997280418346 1.2905655811601933 … -1.113412718124876 0.4224429822535648; … ; -0.8259273645794405 0.8120620193970456 … -0.15587253579758759 1.1574695830467834; 1.0788611984412293 1.2284093434139047 … 0.8824088821353901 0.3813812083882932]         
 [0.3918670037264455 0.22950182665140914 … 0.6770590224331631 -0.25256031424123226; 1.9204591807195388 -0.6076084890625175 … 1.0310040057616838 -1.7671208039765596; … ; 0.5799626195415907 0.569222606661083 … -0.6207019719221616 -0.20391984832884374; -0.9211372187326794 0.44983197168515526 … -0.5049251408980626 -0.17916820255012375]     
 [-1.2825390392798686 0.3693776463439366 … 0.42456048203585195 0.4091195519692529; 0.37893454217288014 0.33825718394132354 … -0.22838574521832017 -1.3427839180011332; … ; -0.803117711548536 1.3428601980024508 … -1.099475110503509 0.8837953536952086; -0.3160402227917924 1.7788621181954565 … -0.1181775330304786 0.051252762995059806]      
 [-0.21602745353895164 0.012000140979508711 … 0.896956844416174 0.22928973833631625; -0.7842241785543619 -0.32949835028447044 … -0.8048870286625319 0.16559858376597783; … ; 1.390319971887969 0.4193290677230986 … -0.749695268782869 -0.5448365210194996; 0.34791591520010057 0.2972162852854982 … -0.2026141522858165 1.3383401586637362]      
 [0.6594695901696367 1.5320772079720624 … -0.19847478092312748 -0.8653458363609802; 1.4585495614063246 -1.2300347093485384 … 2.1313306980929454 1.2275580250098121; … ; 0.9297648333661448 0.36369987357191985 … -2.196675279232564 0.852743816866466; -0.5375674199466393 0.923326234067758 … -0.16939398815990775 1.7503227614136636]           
 [0.12034611808950076 -1.1590390150338736 … -0.6039706346882843 -0.7583855141108757; -0.7606317585112351 0.9554944399438954 … 0.10425768324174194 -0.8995822359312021; … ; 0.5535277421769873 2.140671177435082 … -1.347488594326773 -0.2901472796237467; 1.4890853603600709 -0.6078320966265716 … 0.6995557559187338 1.797947737070229]          
 [-0.9218924802677713 0.4770979031282421 … -0.5055466339174239 0.8738141299971941; 0.73665174584806 2.1342570036702084 … -0.040302585687501044 -1.756282942531084; … ; -0.7406339259737408 0.8871629875178407 … 0.07589856412209975 1.204299863671966; -0.8082412377179505 -0.23452321526257708 … -0.39562475685025467 2.1299724960724045]
```

Create initial `3 x 3` filters, e.g., using DCT,
```julia
julia> using ConvolutionalOperatorLearning

julia> H0 = generatefilters(:DCT,(3,3),form=:matrix)
9×9 Array{Float64,2}:
 0.111111   0.136083      0.0785674   0.136083      0.166667      0.096225      0.0785674   0.096225      0.0555556
 0.111111  -1.74455e-17  -0.157135    0.136083     -2.13663e-17  -0.19245       0.0785674  -1.23358e-17  -0.111111
 0.111111  -0.136083      0.0785674   0.136083     -0.166667      0.096225      0.0785674  -0.096225      0.0555556
 0.111111   0.136083      0.0785674  -1.74455e-17  -2.13663e-17  -1.23358e-17  -0.157135   -0.19245      -0.111111
 0.111111  -1.74455e-17  -0.157135   -1.74455e-17   2.7391e-33    2.46716e-17  -0.157135    2.46716e-17   0.222222
 0.111111  -0.136083      0.0785674  -1.74455e-17   2.13663e-17  -1.23358e-17  -0.157135    0.19245      -0.111111
 0.111111   0.136083      0.0785674  -0.136083     -0.166667     -0.096225      0.0785674   0.096225      0.0555556
 0.111111  -1.74455e-17  -0.157135   -0.136083      2.13663e-17   0.19245       0.0785674  -1.23358e-17  -0.111111
 0.111111  -0.136083      0.0785674  -0.136083      0.166667     -0.096225      0.0785674  -0.096225      0.0555556
```

Run CAOL
```julia
julia> λ = 1e-4      # regularization parameter
0.0001

julia> CAOL(x,λ,(H0,(3,3)),maxiters=30)
9×9 Array{Float64,2}:
 0.111265   0.136016      0.0784865   0.136052      0.166719      0.0960939     0.0785114   0.0962194     0.0557584
 0.111156   5.02982e-5   -0.157221    0.135858     -3.13768e-5   -0.19253       0.0786885  -1.47525e-5   -0.110996
 0.111159  -0.136107      0.0785521   0.136102     -0.166563      0.0962433     0.0785622  -0.096378      0.0553945
 0.110976   0.136055      0.0786667   2.48235e-5    0.000248852  -0.000180995  -0.157047   -0.192529     -0.111196
 0.111119  -0.000109812  -0.157062   -0.000173108  -5.33203e-5   -7.34933e-5   -0.157193   -0.000328285   0.222228
 0.111348  -0.136088      0.0785111  -0.000119645   1.59918e-5   -0.000121294  -0.157086    0.192387     -0.111086
 0.111146   0.136012      0.0787575  -0.136207     -0.166629     -0.0961987     0.0785864   0.0959427     0.0557062
 0.111195   8.71936e-5   -0.157024   -0.136244      0.000148541   0.192415      0.0785643  -0.000150345  -0.111049
 0.110634  -0.136218      0.0786252  -0.136034      0.166755     -0.0962737     0.0786482  -0.0963271     0.0555741
```

The output has `9` filters of size `3 x 3`.

**TODO: Clean-up and add more examples, documentation**

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

## Benchmarking

A small benchmark script is in `benchmark/benchmarks.jl`.
To use this, you will need to install
[`PkgBenchmark.jl`](https://github.com/JuliaCI/PkgBenchmark.jl)
and [`BenchmarkTools.jl`](https://www.github.com/JuliaCI/BenchmarkTools.jl).
Then run
```julia
using PkgBenchmark
b = benchmarkpkg("ConvolutionalOperatorLearning")
export_markdown(stdout,b)
```
to get a markdown representation of the results to `stdout`.

To benchmark against the previous commit use
```julia
using PkgBenchmark
b = judge("ConvolutionalOperatorLearning","HEAD~1")
export_markdown(stdout,b)
```

**TODO:** add more benchmarks, benchmark individual updates
