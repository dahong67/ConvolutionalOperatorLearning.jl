module ConvolutionalAnalysisOperatorLearning

using OffsetArrays, ImageFiltering, LinearAlgebra

export CAOL, generatefilters

# Utility functions
sosdiff(a::Number,b::Number) = abs2(a-b)
sosdiff(A,B) = sum(ab -> sosdiff(ab...),zip(A,B))

hard(x, beta) = abs(x) < beta ? zero(x) : x
_obj(zlk,λ) = sum(z -> (abs(z) < sqrt(2λ) ? abs2(z)/2 : λ), zlk)

_filtermatrix(hlist) =
    (hcat([vec(h) for h in hlist]...)::Matrix{eltype(first(hlist))},
     size(first(hlist)))
_filterlist(Hmatrix,R) = [reshape(h,map(n->1:n,R)) for h in eachcol(Hmatrix)]

# Core steps
function _initvars(x,H0,R)
    K = size(H0,2)

    # Form padded images
    xpad = [padarray(xl,Pad(:circular,ntuple(_->0,ndims(xl)),R)) for xl in x]

    # Initialize filters
    H = copy(H0)
    h = _filterlist(H,R)
    Hprev = similar(H)  # for stopping condition

    # Initialize temporary variables
    zlk = similar(first(x),map(n->0:n-1,size(first(x))))
    ΨZ = similar(H)
    ψz = _filterlist(ΨZ,R)
    ψztemp = similar(first(ψz))
    HΨZ = similar(H,K,K)
    UVt = HΨZ

    return xpad, H, h, Hprev, zlk, ΨZ, ψz, ψztemp, HΨZ, UVt
end
@inline function _updateΨZ!(ΨZ,ψz,xpad,h,λ,zlk,ψztemp)
    obj = 0.0
    fill!(ΨZ,zero(eltype(ΨZ)))
    for xpadl in xpad, k in 1:length(h)
        imfilter!(zlk,xpadl,(h[k],),NoPad(),Algorithm.FIR())
        obj += _obj(zlk,λ)
        zlk .= hard.(zlk,sqrt(2λ))
        imfilter!(ψztemp,xpadl,(zlk,),NoPad(),Algorithm.FIR())
        ψz[k] .+= ψztemp
    end
    return obj
end
function _updateH!(H,ΨZ,H0,HΨZ,UVt)
    mul!(HΨZ,H0',ΨZ)
    F = svd!(HΨZ)
    mul!(UVt,F.U,F.Vt)
    mul!(H,H0,UVt)
end

# Core CAOL
function _CAOL(x,H0,R,λ,maxiters,tol)
    @assert H0'H0 ≈ (1/prod(R))*I
    xpad, H, h, Hprev, zlk, ΨZ, ψz, ψztemp, HΨZ, UVt = _initvars(x,H0,R)

    for t in 1:maxiters
        copyto!(Hprev,H)                      # Copy previous filters
        _updateΨZ!(ΨZ,ψz,xpad,h,λ,zlk,ψztemp) # Update ΨZ
        _updateH!(H,ΨZ,H0,HΨZ,UVt)            # Update filters as polar factor

        sqrt(sosdiff(Hprev,H) / (size(H,2)/size(H,1))) <= tol && break
    end

    return H
end
function _CAOLtrace(x,H0,R,λ,maxiters,tol)
    @assert H0'H0 ≈ (1/prod(R))*I
    xpad, H, h, Hprev, zlk, ΨZ, ψz, ψztemp, HΨZ, UVt = _initvars(x,H0,R)

    Htrace     = fill(H0,0)
    objtrace   = fill(NaN,0)
    Hdifftrace = fill(NaN,0)

    for t in 1:maxiters
        copyto!(Hprev,H)                            # Copy previous filters
        obj = _updateΨZ!(ΨZ,ψz,xpad,h,λ,zlk,ψztemp) # Compute objective, update ΨZ
        _updateH!(H,ΨZ,H0,HΨZ,UVt)                  # Update filters as polar factor

        push!(Htrace,copy(H))
        push!(objtrace,obj)
        push!(Hdifftrace, sqrt(sosdiff(Hprev,H) / (size(H,2)/size(H,1))))

        Hdifftrace[end] <= tol && break
    end

    return H, Htrace, objtrace, Hdifftrace
end

# Docstring
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

# Signatures
SignalBank{N}  = AbstractVector{<:AbstractArray{<:Any,N}}    # List of arrays
SignalTuple{N} = Tuple{<:AbstractMatrix,NTuple{N,<:Integer}} # Columns with shape

# x::SignalBank, h0::SignalBank
function CAOL(x::SignalBank{N},λ::Real,h0::SignalBank{N};
        p=0,maxiters=2000,tol=1e-13,trace=false) where N
    @assert p < length(h0)

    H0,R = _filtermatrix(h0[p+1:end])
    if !trace
        H = _CAOL(x,H0,R,λ,maxiters,tol)
        return [h0[1:p]; Array.(_filterlist(H,R))]
    else
        H, Htrace, objtrace, Hdifftrace = _CAOLtrace(x,H0,R,λ,maxiters,tol)
        return [h0[1:p]; Array.(_filterlist(H,R))], _filterlist.(Htrace), objtrace, Hdifftrace
    end
end

# x::SignalBank, (H0,R)::SignalTuple
function CAOL(x::SignalBank{N},λ::Real,(H0,R)::SignalTuple{N};
        p=0,maxiters=2000,tol=1e-13,trace=false) where N
    @assert p < size(H0,2)

    if !trace
        H = _CAOL(x,H0[:,p+1:end],R,λ,maxiters,tol)
        return [H0[:,1:p] H]
    else
        H, Htrace, objtrace, Hdifftrace = _CAOLtrace(x,H0[:,p+1:end],R,λ,maxiters,tol)
        return [H0[:,1:p] H], Htrace, objtrace, Hdifftrace
    end
end

# x::AbstractArray
_extractsignals(X::AbstractArray) = collect(eachslice(X,dims=ndims(X)))
CAOL(X::AbstractArray,λ::Real,h0::SignalBank; p=0,maxiters=2000,tol=1e-13,trace=false) =
    CAOL(_extractsignals(X),λ,h0; p=p,maxiters=maxiters,tol=tol,trace=trace)
CAOL(X::AbstractArray,λ::Real,H0R::SignalTuple; p=0,maxiters=2000,tol=1e-13,trace=false) =
    CAOL(_extractsignals(X),λ,H0R; p=p,maxiters=maxiters,tol=tol,trace=trace)

## Initializations
using LinearAlgebra, FFTW

generatefilters(type,dims;form=:list) = _generatefilters(Val(type),dims,Val(form))

# DCT
function _generatefilters(::Val{:DCT},dims,::Val{:list})
    H = _generatefilters(Val(:DCT),dims,Val(:matrix))
    return Array.(_filterlist(H,dims))
end
function _generatefilters(::Val{:DCT},dims,::Val{:matrix})
    @assert length(dims) == 2           "Only 2D DCT is implemented"      # todo
    @assert all(i->i==first(dims),dims) "Only square DCT is implemented"  # todo

    temp = dct(Matrix(I,dims),1)
    return kron(temp,temp)' / sqrt(prod(dims))
end

end # module
