module ConvolutionalAnalysisOperatorLearning

using OffsetArrays, ImageFiltering, LinearAlgebra

export CAOL, _CAOLnew, _CAOLtracenew

# Utility functions
sosdiff(a::Number,b::Number) = abs2(a-b)
sosdiff(A,B) = sum(ab -> sosdiff(ab...),zip(A,B))

hard(x, beta) = abs(x) < beta ? zero(x) : x
_obj(zlk,λ) = sum(z -> (abs(z) < sqrt(2λ) ? abs2(z)/2 : λ), zlk)

### Work on new implementation ###

# Rough plan
# function _CAOL(x,(H0,R),λ,maxiters,tol)
#     ??, ??, ??, ??, = _init()
#
#     for t in 1:maxiters
#         copyto!(Hprev,H)
#         _updateΨZ!(??,??,??,false)
#         _updateH!(??,??,??)
#         ??? <= tol && break
#     end
#     return H
# end
# function _CAOLtrace(x,(H0,R),λ,maxiters,tol)
#     ??, ??, ??, ??, = _init()
#
#     # Initialize trace
#     ??
#     ??
#     ??
#
#     for t in 1:maxiters
#         copyto!(Hprev,H)
#         obj = _updateΨZ!(??,??,??,true)
#         _updateH!(??,??,??)
#
#         # Store trace
#         ?? = ??
#
#         ??? <= tol && break
#     end
#     return H
# end

function _initvars(x,H0,R)
    K = size(H0,2)

    # Form padded images
    xpad = [padarray(xl,Pad(:circular,ntuple(_->0,ndims(xl)),R)) for xl in x]

    # Initialize filters
    H = copy(H0)
    h = [reshape(view(H,:,k),map(n->1:n,R)) for k in 1:K]
    Hprev = similar(H)  # for stopping condition

    # Initialize temporary variables
    zlk = similar(first(x),map(n->0:n-1,size(first(x))))
    ΨZ = similar(H)
    ψz = [reshape(view(ΨZ,:,k),map(n->1:n,R)) for k in 1:K]
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
function _CAOLnew(x,H0,R,λ,maxiters,tol)
    @assert H0'H0 ≈ (1/prod(R))*I
    xpad, H, h, Hprev, zlk, ΨZ, ψz, ψztemp, HΨZ, UVt = _initvars(x,H0,R)

    for t in 1:maxiters
        copyto!(Hprev,H)                      # Copy previous filters
        _updateΨZ!(ΨZ,ψz,xpad,h,λ,zlk,ψztemp) # Update ΨZ
        _updateH!(H,ΨZ,H0,HΨZ,UVt)            # Update filters as polar factor

        sosdiff(Hprev,H) / (size(H,2)/size(H,1)) <= tol && break
    end

    return H
end
function _CAOLtracenew(x,H0,R,λ,maxiters,tol)
    @assert H0'H0 ≈ (1/prod(R))*I
    xpad, H, h, Hprev, zlk, ΨZ, ψz, ψztemp, HΨZ, UVt = _initvars(x,H0,R)

    Hs = Array{typeof(H0)}(undef,maxiters)
    obj   = OffsetArray(fill(NaN,maxiters),-1)
    Hdiff = fill(NaN,maxiters)

    for t in 1:maxiters
        copyto!(Hprev,H)                                 # Copy previous filters
        obj[t-1] = _updateΨZ!(ΨZ,ψz,xpad,h,λ,zlk,ψztemp) # Compute objective, update ΨZ
        _updateH!(H,ΨZ,H0,HΨZ,UVt)                       # Update filters as polar factor

        Hs[t] = copy(H)

        # Terminate
        Hdiff[t] = sqrt(sosdiff(Hprev,H) / (size(H0,2)/prod(R)))
        Hdiff[t] <= tol && break
    end

    niters = count(o -> !isnan(o),obj)
    return H, (obj[0:niters-1],Hdiff[1:niters]), Hs[1:niters]
end

### Current implementation ###

"""
    CAOL(x, h0, λ; maxiters = 2000, tol = 1e-13, trace = false)

Learn convolutional filters from training data x, initialized by h0, with tuning
parameter λ, and given maximum number of iterations and tolerance for convergence.

# Arguments
- x: vector of training data. The first dimension indexes the training samples.
     x[i] is a single training example (e.g., a 2d image array)
- h0: initialization for filter matrix. The first dimension indexes the number
      of filters. h0[i] is a filter (e.g., 2D DCT approximation).
      These must be orthogonal and have 2-norm of 1/(filter length)
- λ: tuning parameter. Larger values put more emphasis on sparsity; smaller
     values put more emphasis on data-fit.
- maxiters=2000: maximum number of iterations for CAOL to run
- tol=1e-13: tolerance for testing convergence between iterations
- trace: set to true to store a trace of the filter matrix iterates

# Outputs
- h : final learned filters. Same size and arrangement as h0
- (obj,Hdiff) : obj is a vector of all the objective function values and
                Hdiff is a vector of the sequential differences used for
                convergence, i.e., Hdiff[t] = norm(H[t]-H[t-1])/norm(H[t])
                (omitted if trace is false)
- Hs : a vector where Hs[i] is the filter matrix H after the ith iteration
       (omitted if trace is false)

# Examples
```julia-repl
julia> R1 = 3; # filters will be dimension R1 x R1
julia> x = [[i+j for i in 1:10, j in 2:11]./21,
            [i*j for i in 1:10, j in 2:11]./110]; # two example "images"
julia> H0 = dct(Matrix(I,R1*R1,R1*R1),1)'/R1;
julia> h0 = [reshape(H0[:,k], R1, R1) for k in 1:R1*R1];
julia> h = CAOL(x, h0, 1e-4; maxiters=30)
julia> h, (obj,Hdiff), Hs = CAOL(x, h0, 1e-4; maxiters=30, trace=true)
```
"""
function CAOL(x,h0,λ;maxiters=2000,tol=1e-13,trace=false)
    out = _CAOL(x,h0,λ,maxiters,tol,trace)
    return trace ? out : out[1]
end
function _CAOL(x,h0,λ,maxiters,tol,trace)
    R, K = size(first(h0)), length(h0)

    H0 = similar(first(h0),prod(R),K)
    for k in 1:K
        H0[:,k] = vec(h0[k])
    end

    H, (obj,Hdiff), Hs = _CAOL(x,H0,R,λ,maxiters,tol,trace)
    h = [reshape(H[:,k],R) for k in 1:K]

    return h, (obj,Hdiff), Hs
end

"""
    CAOL(x, H0, R, λ; maxiters = 2000, tol = 1e-13, trace = false)

Equivalent to CAOL(x, h0, λ; maxiters = 2000, tol = 1e-13, trace = false) but
with filters represented instead by the matrix H0 with filter size given by R.

# Arguments
- x: vector of training data. The first dimension indexes the training samples.
     x[i] is a single training example (e.g., a 2d image array)
- H0: initialization for filter matrix of size (length of filter) x (# of filters)
- λ: tuning parameter. Larger values put more emphasis on sparsity; smaller
     values put more emphasis on data-fit.
- maxiters=2000: maximum number of iterations for CAOL to run
- tol=1e-13: tolerance for testing convergence between iterations
- trace: set to true to store a trace of the filter matrix iterates

# Outputs
- H : final learned filters. Same size and arrangement as H0
- (obj,Hdiff) : obj is a vector of all the objective function values and
                Hdiff is a vector of the sequential differences used for
                convergence, i.e., Hdiff[t] = norm(H[t]-H[t-1])/norm(H[t])
                (omitted if trace is false)
- Hs : a vector where Hs[i] is the filter matrix H after the ith iteration
       (omitted if trace is false)

# Examples
```julia-repl
julia> R1 = 3; # filters will be dimension R1 x R1
julia> x = [[i+j for i in 1:10, j in 2:11]./21,
            [i*j for i in 1:10, j in 2:11]./110]; # two example "images"
julia> H0 = dct(Matrix(I,R1*R1,R1*R1),1)'/R1;
julia> H = CAOL(x, H0, (R1,R1), 1e-4; maxiters=30)
julia> H, (obj,Hdiff), Hs = CAOL(x, H0, (R1,R1), 1e-4; maxiters=30, trace=true)
```
"""
function CAOL(x,H0,R,λ;maxiters=2000,tol=1e-13,trace=false)
    out = _CAOL(x,H0,R,λ,maxiters,tol,trace)
    return trace ? out : out[1]
end
function _CAOL(x,H0,R,λ,maxiters,tol,trace)
    @assert H0'H0 ≈ (1/prod(R))*I
    K = size(H0,2)

    # Form padded images
    xpad = [padarray(xl,Pad(:circular,ntuple(_->0,ndims(xl)),R)) for xl in x]

    # Initialize filters
    H = copy(H0)
    h = [reshape(view(H,:,k),map(n->1:n,R)) for k in 1:K]
    Hprev = similar(H)  # for stopping condition

    # Initialize temporary variables
    zlk = similar(first(x),map(n->0:n-1,size(first(x))))
    ΨZ = similar(H)
    ψz = [reshape(view(ΨZ,:,k),map(n->1:n,R)) for k in 1:K]
    ψztemp = similar(first(ψz))
    HΨZ = similar(H,K,K)
    UVt = HΨZ

    # Initialize trace of H's, objectives and Hdiff
    Hs = Array{typeof(H0)}(undef,maxiters)
    obj   = OffsetArray(fill(NaN,maxiters),-1)
    Hdiff = fill(NaN,maxiters)

    for t in 1:maxiters
        copyto!(Hprev,H)

        # Compute objective and update ΨZ
        obj[t-1] = zero(obj[t-1])
        fill!(ΨZ,zero(eltype(ΨZ)))
        for xpadl in xpad, k in 1:K
            imfilter!(zlk,xpadl,(h[k],),NoPad(),Algorithm.FIR())
            obj[t-1] += _obj(zlk,λ)
            zlk .= hard.(zlk,sqrt(2λ))
            imfilter!(ψztemp,xpadl,(zlk,),NoPad(),Algorithm.FIR())
            ψz[k] .+= ψztemp
        end

        # Update filter via polar factorization
        mul!(HΨZ,H0',ΨZ)
        F = svd!(HΨZ)
        mul!(UVt,F.U,F.Vt)
        mul!(H,H0,UVt)

        # Store trace of H if debug on
        trace && (Hs[t] = copy(H))

        # Terminate
        Hdiff[t] = sqrt(sosdiff(Hprev,H) / (K/prod(R)))
        Hdiff[t] <= tol && break
    end

    niters = count(o -> !isnan(o),obj)
    return H, (obj[0:niters-1],Hdiff[1:niters]), Hs[1:niters]
end

end # module
