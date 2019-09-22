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
            push!(H_trace, H)
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

function CAOL(x, h0, λ; maxiters = 2000, tol = 1e-13, debug=false)
    h0c = centered.(h0)
    xpad = [padarray(xl,Pad(:circular)(h0c[1])) for xl in x]

    # TODO: test for (scaled) orthonormality

    return _CAOL(xpad, h0c, λ, maxiters, tol, debug)
end
function _CAOL(xpad, h0, λ, maxiters, tol, debug)
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
            push!(H_trace, H)
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

end # module
