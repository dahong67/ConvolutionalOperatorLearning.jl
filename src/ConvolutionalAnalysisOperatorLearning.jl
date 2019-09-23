module ConvolutionalAnalysisOperatorLearning

using OffsetArrays, ImageFiltering, LinearAlgebra

export CAOL

sosdiff(a::Number,b::Number) = abs2(a-b)
sosdiff(A,B) = sum(ab -> sosdiff(ab...),zip(A,B))

hard(x, beta) = abs(x) < beta ? zero(x) : x
_obj(zlk,λ) = sum(z -> (abs(z) < sqrt(2λ) ? abs2(z)/2 : λ), zlk)

function CAOL(x,H0,R,λ,niters,tol,trace)
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
    Hs = Array{typeof(H0)}(undef,niters)
    obj   = OffsetArray(fill(NaN,niters),-1)
    Hdiff = fill(NaN,niters)

    for t in 1:niters
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

    niter = count(o -> !isnan(o),obj)
    return H, (obj[0:niter-1],Hdiff[1:niter]), Hs[1:niter]
end

end # module
