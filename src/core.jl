## Core CAOL routines

using OffsetArrays, ImageFiltering, LinearAlgebra

# Allocation and initialization
function _initvars(x,H0,R)
    K = size(H0,2)

    # Form padded images
    xpad = [padarray(xl,Pad(:circular,ntuple(_->0,ndims(xl)),R)) for xl in x]
    # xpad = [view(xl,ntuple(i->vcat(1:size(xl,i),1:R[i]),ndims(xl))...) for xl in x]
    # more memory efficient but seems slower, make a copy for now
    # may be relevant: https://docs.julialang.org/en/v1/manual/performance-tips/#Copying-data-is-not-always-bad-1

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

# Update steps
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

# Iterative algorithms
function _CAOL(x,λ,H0,R,maxiters,tol)
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
function _CAOLtrace(x,λ,H0,R,maxiters,tol)
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
