module ConvolutionalOperatorLearning

export CAOL, generatefilters

include("util.jl")          # utility functions
include("core.jl")          # core implementation
include("signatures.jl")    # CAOL function signatures
include("filters.jl")       # filters for initialization

"""
    CAOL(x, λ, h0; p=0, maxiters=2000, tol=1e-13, trace=false)

Learn convolutional analysis operators, i.e., sparsifying filters,
for signals `x` with sparsity regularization `λ` and initial filters `h0`.
Preserve the first `p` filters in `h0`, i.e., consider them handcrafted.

`x` can be either:
+ a vector of training signals each a D-dimensional array
+ a (D+1)-dimnensional array with training sample `i` being slice `x[:,...,:,i]`

`h0` can be either:
+ a vector of filters each a D-dimensional array
+ a tuple `(H0,R)` where each column of the matrix `H0` is a vectorized filter
  and where `R` gives the size/shape of the filter.
The filters must be orthogonal to one another
and each filter must be normalized to have norm `1/sqrt(filter length)`,
i.e., `H0'H0 ≈ (1/size(H0,1))*I`.

When `trace=false`, CAOL returns just the learned filters (in the same form as `h0`).
When `trace=true`, CAOL also returns
+ the iterates (only the learned filters since handcrafted filters do not change)
+ the objective function values (evaluated on only the learned filters)
+ the iterate differences `norm(H[t]-H[t-1])/norm(H[t])` used for the stopping criterion

# Examples
Passing in a vector of ten 100x50 training signals and a vector of 3x3 DCT filters
```julia-repl
julia> x = [randn(100,50) for _ in 1:10];
julia> h0 = generatefilters(:DCT,(3,3));
julia> h = CAOL(x,1e-3,h0,maxiters=10);
```
Passing the ten 100x50 training signals as an array
```julia-repl
julia> X = randn(100,50,10);
julia> h0 = generatefilters(:DCT,(3,3));
julia> h = CAOL(X,1e-3,h0,maxiters=10);
```
Passing the DCT filters in matrix form
```julia-repl
julia> x = [randn(100,50) for _ in 1:10];
julia> H0 = generatefilters(:DCT,(3,3),form=:matrix);
julia> H = CAOL(x,1e-3,(H0,(3,3)),maxiters=10);
```
Getting the trace
```julia-repl
julia> x = [randn(100,50) for _ in 1:10];
julia> H0 = generatefilters(:DCT,(3,3),form=:matrix);
julia> H, Htrace, objtrace, Hdifftrace = CAOL(x,1e-3,(H0,(3,3)),maxiters=10,trace=true);
```
"""
function CAOL end

end # module
