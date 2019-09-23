module ConvolutionalAnalysisOperatorLearning

using OffsetArrays, ImageFiltering, LinearAlgebra, IterTools

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

function CAOL(x,h0::Vector,λ,niters,tol,trace)
    R, K = size(h0[1]), length(h0)

    H0 = similar(h0[1],prod(R),K) # vectorized form
    for k in 1:K
        H0[:,k] = vec(h0[k])
    end

    H, (obj,Hdiff), Hs = CAOL(x,H0,R,λ,niters,tol,trace)
    h = [reshape(view(H,:,k),map(n->1:n,R)) for k in 1:K]

    return (h,length(obj),parent(obj),Hs,Hdiff)
end

_obj(zlk,λ) = sum(z -> (abs(z) < sqrt(2λ)) ? abs2(z)/2 : λ, zlk)
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
        Hdiff[t] = normdiff(Hprev,H)/norm(H)
        Hdiff[t] <= tol && break
    end

    niter = count(o -> !isnan(o),obj)
    return H, (obj[0:niter-1],Hdiff[1:niter]), Hs[1:niter]
end

end # module
