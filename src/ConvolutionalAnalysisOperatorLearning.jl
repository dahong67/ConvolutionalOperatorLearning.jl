module ConvolutionalAnalysisOperatorLearning

using OffsetArrays, ImageFiltering, LinearAlgebra

export CAOL7, CAOL

hard(x, beta) = abs(x) < beta ? zero(x) : x
function procrustesfilter!(H,PsiZ)
    R = size(H,1)

    # Form PsiZ matrix (overwrite in H)
    for k in axes(H,2)
        H[:,k] = vec(PsiZ[k])
    end

    # Compute polar factorization (overwrite in H)
    F = svd!(H)
    mul!(H,F.U,F.Vt)
    H ./= sqrt(R)

    return H
end

normdiff(a::Number,b::Number,p) = abs(a-b)   # norm(a-b,p) = abs(a-b) for numbers
normdiff(A,B,p=2) = norm((normdiff(a,b,p) for (a,b) in zip(A,B)),p)

sos(a::Number) = abs2(a)                   # (recursive) sum of squares
sos(A) = sum(sos,A)
sosdiff(a::Number,b::Number) = abs2(a-b)   # (recursive) sum of square differences
sosdiff(A,B) = sum(ab -> sosdiff(ab...),zip(A,B))

# Version that can handle p > 0 (by passing fewer than R filters)
# todo: maybe form H0 in outer function
function CAOL7(x, h0, λ; maxiters = 2000, tol = 1e-13, debug=false)
    h0c = centered.(h0)
    xpad = [padarray(xl,Pad(:circular)(h0c[1])) for xl in x]

    # TODO: test for (scaled) orthonormality

    return _CAOL7(xpad, h0c, λ, maxiters, tol, debug)
end
function _CAOL7(xpad, h0, λ, maxiters, tol, debug)
    L, K = length(xpad), length(h0)
    R = length(h0[1])

    # Initialize: filters (todo: think about axes...currently assumes h0 centered)
    H = similar(h0[1],R,K)                              # vectorized form
    for k in 1:K
        H[:,k] = vec(h0[k])
    end
    h = [reshape(view(H,:,k),axes(h0[k])) for k in 1:K] # natural form view
    Hprev = similar(H)                                  # for convergence test
    H0 = copy(H)

    # Initialize: temporary variables
    #zlk = similar(Array{Float64}(undef,size(xpad[1])),ImageFiltering.interior(xpad[1],h0[1]))  # h0 -> h
    zlk = similar(xpad[1],ImageFiltering.interior(xpad[1],h0[1])) # TODO: remove undocumented interior()
    ΨZ = similar(H)
    ψz = [reshape(view(ΨZ,:,k),axes(h0[k])) for k in 1:K]
    ψztemp = similar(ψz[1])
    HΨZ = similar(H,K,K)
    UVt = HΨZ  # alias to the same memory

    # initializations if debug is on
    if debug
        xconvh = similar(xpad[1],ImageFiltering.interior(xpad[1],h0[1]))
        niter = 1;
        H_trace = [];
        H_convergence = [];
        obj_fnc_vals = [];
    end

    # Main loop
    for t in 1:maxiters
        obj_fnc = 0

        # Compute ΨZ
        fill!(ΨZ,zero(eltype(ΨZ)))
        for l in 1:L, k in 1:K
            imfilter!(zlk,xpad[l],(h[k],),NoPad(),Algorithm.FIR())
            if debug
                xconvh .= zlk
            end
            zlk .= hard.(zlk,sqrt(2λ))
            imfilter!(ψztemp,xpad[l],(zlk,),NoPad(),Algorithm.FIR())
            ψz[k] .+= ψztemp
            if debug # calculate the objective function
                obj_fnc += 1/2*sosdiff(xconvh,zlk) + λ*norm(zlk,0)
            end
        end

        # Update filter via polar factorization
        copyto!(Hprev,H)
        mul!(HΨZ,H0',ΨZ)   # if we observe drift, may want to use a copy of H0
        F = svd!(HΨZ)
        mul!(UVt,F.U,F.Vt)
        mul!(H,H0,UVt)

        # TODO: implement restart conditions

        # take care of some output
        if debug
            #push!(H_convergence, normdiff(H,Hprev)/norm(H))
            push!(H_convergence, norm( H[:]-Hprev[:] ) / norm( H[:] ))
            push!(H_trace, copy(H))
            push!(obj_fnc_vals, obj_fnc)
        end

        # Check convergence criteria
        if (normdiff(H,Hprev)/norm(H) < tol)
            niter = t
            break
        end
        if t == maxiters
            niter = t
        end
    end

    if debug
        return (h,niter,obj_fnc_vals, H_trace,H_convergence)
    else
        return h
    end
end

# todos
# + stopping criterion
# + parametric types?


import Base: iterate, IteratorSize, IsInfinite, eltype

struct CAOLIterable
    x
    H0
    R
    λ

    CAOLIterable(x,H0,R,λ) = !(H0'H0 ≈ (1/prod(R))*I) ?
        error("Initial filters not orthonormal.") : new(x,H0,R,λ)
end
IteratorSize(::Type{<:CAOLIterable}) = IsInfinite()
eltype(::Type{CAOLIterable}) = CAOLState

mutable struct CAOLState
    xpad   # padded images
    H      # vectorized form
    h      # natural form view
    obj    # objective function

    # Temporary variables
    zlk
    ΨZ
    ψz
    ψztemp
    HΨZ
    UVt
end
function CAOLState(it::CAOLIterable)   # Form initial state from CAOLIterable
    K = size(it.H0,2)

    # Padded images
    xpad = [padarray(xl,Pad(:circular,ntuple(_->0,ndims(xl)),it.R)) for xl in it.x]

    # Initial filters
    H = copy(it.H0)
    h = [reshape(view(H,:,k),map(n->1:n,it.R)) for k in 1:K]

    # Objective function value
    obj = zero(eltype(it.H0))

    # Temporary variables
    zlk = similar(first(it.x),map(n->0:n-1,size(first(it.x))))
    ΨZ = similar(H)
    ψz = [reshape(view(ΨZ,:,k),map(n->1:n,it.R)) for k in 1:K]
    ψztemp = similar(first(ψz))
    HΨZ = similar(H,K,K)
    UVt = HΨZ

    return CAOLState(xpad,H,h,obj,zlk,ΨZ,ψz,ψztemp,HΨZ,UVt)
end

_obj(zlk,λ) = sum(z -> (abs(z) < sqrt(2λ)) ? abs2(z)/2 : λ, zlk)
function iterate(it::CAOLIterable,s::CAOLState=CAOLState(it))
    L, K = length(it.x), length(s.h)
    s.obj = zero(s.obj)

    # Compute ΨZ
    fill!(s.ΨZ,zero(eltype(s.ΨZ)))
    for l in 1:L, k in 1:K
        imfilter!(s.zlk,s.xpad[l],(s.h[k],),NoPad(),Algorithm.FIR())
        s.obj += _obj(s.zlk,it.λ)
        s.zlk .= hard.(s.zlk,sqrt(2*it.λ))
        imfilter!(s.ψztemp,s.xpad[l],(s.zlk,),NoPad(),Algorithm.FIR())
        s.ψz[k] .+= s.ψztemp
    end

    # Update filter via polar factorization
    mul!(s.HΨZ,it.H0',s.ΨZ)
    F = svd!(s.HΨZ)
    mul!(s.UVt,F.U,F.Vt)
    mul!(s.H,it.H0,s.UVt)

    return s,s
end

function CAOL(x,h0::Vector,λ,niters)
    R, K = size(h0[1]), length(h0)

    H0 = similar(h0[1],prod(R),K) # vectorized form
    for k in 1:K
        H0[:,k] = vec(h0[k])
    end

    return CAOL(x,H0,R,λ,niters)
end

function CAOL(x,H0,R,λ,niters)
    h = nothing

    # debug: initialization
    niter = 0
    H_trace = []
    H_convergence = []
    obj_fnc_vals = []
    Hprev = copy(H0)
    # debug

    for s in Iterators.take(CAOLIterable(x,H0,R,λ),niters)
        h = s.h

        # debug: save outputs
        niter += 1
        push!(H_convergence, normdiff(s.H,Hprev)/norm(s.H))
        push!(H_trace, copy(s.H))
        push!(obj_fnc_vals,s.obj)
        # debug

        copyto!(Hprev,s.H)
    end

    return (h,niter,obj_fnc_vals,H_trace,H_convergence)
end

end # module
