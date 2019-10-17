# 2019.10.06 (dahong): this was an attempt to implement the iterative algorithm
# using iterables to make the code more modular, but this seemed to be less
# memory efficient for some reason, so we are just using loops for now.
# todos: figure out memory inefficiency, make types parametric as appropriate

using IterTools

import Base: iterate, IteratorSize, IsInfinite, SizeUnknown, eltype, tail

struct FilterHaltIterable{T}
    it::T
    tol
end
IteratorSize(::Type{<:FilterHaltIterable}) = SizeUnknown()
eltype(::Type{FilterHaltIterable{I}}) where {I} = eltype(I)

function iterate(fh::FilterHaltIterable{T},state=(false,copy(fh.it.H0),)) where{T}
    halt, Hprev, itstate = state[1], state[2], tail(tail(state))
    halt && return nothing

    itnext = iterate(fh.it,itstate...)
    itnext === nothing && return nothing

    H = itnext[2].H
    haltnext = (normdiff(H,Hprev)/norm(H) <= fh.tol)::Bool

    copyto!(Hprev,H)
    return itnext[1],(haltnext,Hprev,itnext[2])
end

struct CAOLIterable{fmatT}
    x
    H0::fmatT
    R
    λ

    CAOLIterable{fmatT}(x,H0,R,λ) where {fmatT} =
        !(H0'H0 ≈ (1/prod(R))*I) ?
            error("Initial filters not orthonormal.") : new(x,H0,R,λ)
end
CAOLIterable(x,H0::fmatT,R,λ) where {fmatT} = CAOLIterable{fmatT}(x,H0,R,λ)
IteratorSize(::Type{<:CAOLIterable}) = IsInfinite()
eltype(::Type{CAOLIterable{fmatT}}) where {fmatT} = Tuple{fmatT,eltype(fmatT)}

struct CAOLState
    xpad   # padded images
    H      # vectorized form
    h      # natural form view

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

    # Temporary variables
    zlk = similar(first(it.x),map(n->0:n-1,size(first(it.x))))
    ΨZ = similar(H)
    ψz = [reshape(view(ΨZ,:,k),map(n->1:n,it.R)) for k in 1:K]
    ψztemp = similar(first(ψz))
    HΨZ = similar(H,K,K)
    UVt = HΨZ

    return CAOLState(xpad,H,h,zlk,ΨZ,ψz,ψztemp,HΨZ,UVt)
end

_obj(zlk,λ) = sum(z -> (abs(z) < sqrt(2λ)) ? abs2(z)/2 : λ, zlk)
function iterate(it::CAOLIterable{fmatT},s::CAOLState=CAOLState(it)) where {fmatT}
    # Compute objective and ΨZ
    obj = zero(eltype(it.H0))
    fill!(s.ΨZ,zero(eltype(s.ΨZ)))
    for xpadl in s.xpad, k in 1:length(s.h)
        imfilter!(s.zlk,xpadl,(s.h[k],),NoPad(),Algorithm.FIR())
        obj += _obj(s.zlk,it.λ)
        s.zlk .= hard.(s.zlk,sqrt(2*it.λ))
        imfilter!(s.ψztemp,xpadl,(s.zlk,),NoPad(),Algorithm.FIR())
        s.ψz[k] .+= s.ψztemp
    end

    # Update filter via polar factorization
    mul!(s.HΨZ,it.H0',s.ΨZ)
    F = svd!(s.HΨZ)
    mul!(s.UVt,F.U,F.Vt)
    mul!(s.H,it.H0,s.UVt)

    return (s.H,obj)::eltype(it),s
end

function CAOL(x,h0::Vector,λ,niters,tol)
    R, K = size(h0[1]), length(h0)

    H0 = similar(h0[1],prod(R),K) # vectorized form
    for k in 1:K
        H0[:,k] = vec(h0[k])
    end

    return CAOL(x,H0,R,λ,niters,tol)
end
function CAOL(x,H0,R,λ,niters,tol)
    outs = collect(imap(deepcopy,Iterators.take(FilterHaltIterable(CAOLIterable(x,H0,R,λ),tol),niters)))

    H_trace = getindex.(outs,1)
    H_convergence = [normdiff(H_trace[t],t == 1 ? H0 : H_trace[t-1])/norm(H_trace[t]) for t in 1:length(H_trace)]
    obj_fnc_vals = getindex.(outs,2)
    niter = length(outs)

    h = [reshape(view(H_trace[end],:,k),map(n->1:n,R)) for k in 1:size(H_trace[end],2)]

    return (h,niter,obj_fnc_vals,H_trace,H_convergence)
end

function CAOLfast(x,h0::Vector,λ,niters,tol)
    R, K = size(h0[1]), length(h0)

    H0 = similar(h0[1],prod(R),K) # vectorized form
    for k in 1:K
        H0[:,k] = vec(h0[k])
    end

    return CAOLfast(x,H0,R,λ,niters,tol)
end

function CAOLfast(x,H0,R,λ,niters,tol)
    H = H0
    for outs in Iterators.take(FilterHaltIterable(CAOLIterable(x,H0,R,λ),tol),niters)
        H = outs[1]
    end

    return H
end
