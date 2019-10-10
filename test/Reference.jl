# Reference implementation

module Reference

# todo: simplify to better serve as a highly trusted reference implementation
#       goal is to make it simple and easy to verify without worrying too much
#       about computational speed or memory use

using OffsetArrays, ImageFiltering, LinearAlgebra

sosdiff(a::Number,b::Number) = abs2(a-b)
sosdiff(A,B) = sum(ab -> sosdiff(ab...),zip(A,B))

hard(x, beta) = abs(x) < beta ? zero(x) : x
_obj(zlk,λ) = sum(z -> (abs(z) < sqrt(2λ) ? abs2(z)/2 : λ), zlk)

function CAOL(x,λ,H0,R,maxiters,tol)
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
    Htrace     = Array{typeof(H0)}(undef,maxiters)
    objtrace   = OffsetArray(fill(NaN,maxiters),-1)
    Hdifftrace = fill(NaN,maxiters)

    for t in 1:maxiters
        copyto!(Hprev,H)

        # Compute objective and update ΨZ
        objtrace[t-1] = zero(objtrace[t-1])
        fill!(ΨZ,zero(eltype(ΨZ)))
        for xpadl in xpad, k in 1:K
            imfilter!(zlk,xpadl,(h[k],),NoPad(),Algorithm.FIR())
            objtrace[t-1] += _obj(zlk,λ)
            zlk .= hard.(zlk,sqrt(2λ))
            imfilter!(ψztemp,xpadl,(zlk,),NoPad(),Algorithm.FIR())
            ψz[k] .+= ψztemp
        end

        # Update filter via polar factorization
        mul!(HΨZ,H0',ΨZ)
        F = svd!(HΨZ)
        mul!(UVt,F.U,F.Vt)
        mul!(H,H0,UVt)

        # Store trace of H
        Htrace[t] = copy(H)

        # Terminate
        Hdifftrace[t] = sqrt(sosdiff(Hprev,H) / (K/prod(R)))
        Hdifftrace[t] <= tol && break
    end

    niters = count(o -> !isnan(o),objtrace)
    return H, Htrace[1:niters], objtrace[0:niters-1], Hdifftrace[1:niters]
end

end
