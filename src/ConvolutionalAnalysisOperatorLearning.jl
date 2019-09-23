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
end

mutable struct CAOLState
    xpad

    H
    h
    Hprev

    zlk
    ΨZ
    ψz
    ψztemp
    HΨZ
    UVt

    # debug: initialization
    niter
    H_trace
    H_convergence
    obj_fnc_vals
end

IteratorSize(::Type{<:CAOLIterable}) = IsInfinite()
eltype(::Type{CAOLIterable}) = CAOLState

function iterate(it::CAOLIterable)
    K = size(it.H0,2)

    # Initialize: padded versions of images
    xpad = [padarray(xl,Pad(:circular,ntuple(_->0,ndims(xl)),it.R)) for xl in it.x]

    # Initialize: filters
    H = copy(it.H0)
    h = [reshape(view(H,:,k),map(n->1:n,it.R)) for k in 1:K] # natural form view
    Hprev = similar(H)                                       # for convergence test

    # Initialize: temporary variables
    zlk = similar(xpad[1],map(n->0:n-1,size(it.x[1])))
    ΨZ = similar(H)
    ψz = [reshape(view(ΨZ,:,k),axes(h[k])) for k in 1:K]
    ψztemp = similar(ψz[1])
    HΨZ = similar(H,K,K)
    UVt = HΨZ  # alias to the same memory

    # debug: initialization
    niter = 0;
    H_trace = [];
    H_convergence = [];
    obj_fnc_vals = [];
    # debug

    s = CAOLState(xpad,H,h,Hprev,zlk,ΨZ,ψz,ψztemp,HΨZ,UVt,
        niter,H_trace,H_convergence,obj_fnc_vals)
    return s,s
end

objtrick(zlk,λ) = sum(z -> (abs(z) < sqrt(2λ)) ? abs2(z)/2 : λ, zlk)

function iterate(it::CAOLIterable,s::CAOLState)
    L, K = length(it.x), length(s.h)
    obj_fnc = 0

    # Compute ΨZ
    fill!(s.ΨZ,zero(eltype(s.ΨZ)))
    for l in 1:L, k in 1:K
        imfilter!(s.zlk,s.xpad[l],(s.h[k],),NoPad(),Algorithm.FIR())

        # debug: calculate the objective function
        obj_fnc += objtrick(s.zlk,it.λ)
        # debug

        s.zlk .= hard.(s.zlk,sqrt(2*it.λ))
        imfilter!(s.ψztemp,s.xpad[l],(s.zlk,),NoPad(),Algorithm.FIR())
        s.ψz[k] .+= s.ψztemp
    end

    # Update filter via polar factorization
    copyto!(s.Hprev,s.H)
    mul!(s.HΨZ,it.H0',s.ΨZ)
    F = svd!(s.HΨZ)
    mul!(s.UVt,F.U,F.Vt)
    mul!(s.H,it.H0,s.UVt)

    # debug: save outputs
    s.niter += 1
    push!(s.H_convergence, normdiff(s.H,s.Hprev)/norm(s.H))
    push!(s.H_trace, copy(s.H))
    push!(s.obj_fnc_vals, obj_fnc)
    # debug

    return s,s
end

function CAOL(x, h0, λ, niters)
    R, K = size(h0[1]), length(h0)

    H0 = similar(h0[1],prod(R),K) # vectorized form
    for k in 1:K
        H0[:,k] = vec(h0[k])
    end

    # Verify (scaled) orthonormality of H0
    @assert H0'H0 ≈ (1/prod(R))*I

    outs = last(collect(Iterators.take(CAOLIterable(x,H0,R,λ),niters+1)))

    return (outs.h,outs.niter,outs.obj_fnc_vals,outs.H_trace,outs.H_convergence)
end

end # module
