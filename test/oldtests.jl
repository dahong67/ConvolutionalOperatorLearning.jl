using Revise

using ConvolutionalAnalysisOperatorLearning

# using ConvolutionalAnalysisOperatorLearning.FFTW
using LinearAlgebra

using ConvolutionalAnalysisOperatorLearning.OffsetArrays
using ConvolutionalAnalysisOperatorLearning.ImageFiltering
using ConvolutionalAnalysisOperatorLearning: hard, sosdiff


function CAOL2(x, h0, λ; maxiters = 2000, tol = 1e-13, debug=false)
    h0c = [OffsetArray(h,map(n -> 1:n,size(h))) for h in h0]
    xpad = [padarray(xl,Pad(:circular)(h0c[1])) for xl in x]

    # TODO: test for (scaled) orthonormality

    return _CAOL2(xpad, x, h0c, λ, maxiters, tol, debug)
end
function _CAOL2(xpad, x, h0, λ, maxiters, tol, debug)
    L, K = length(xpad), length(h0)
    R = length(h0[1])

    # Initialize: filters
    H = similar(h0[1],R,K)                              # vectorized form
    for k in 1:K
        H[:,k] = vec(h0[k])
    end
    h = [reshape(view(H,:,k),axes(h0[k])) for k in 1:K] # natural form view
    Hprev = similar(H)                                  # for convergence test
    H0 = copy(H)

    # Initialize: temporary variables
    zlk = similar(xpad[1],map(n -> 0:n-1,size(x[1])))
    ΨZ = similar(H)
    ψz = [reshape(view(ΨZ,:,k),axes(h0[k])) for k in 1:K]
    ψztemp = similar(ψz[1])
    HΨZ = similar(H,K,K)
    UVt = HΨZ  # alias to the same memory

    # initializations if debug is on
    if debug
        xconvh = deepcopy(zlk)
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
        if (sqrt(sosdiff(H,Hprev))/norm(H) < tol)
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

## Tests
N1,N2,L = 128, 128, 62
rr = 3
R = rr^2
K = R

H0 = Matrix(qr!(randn(R,K)).Q) / sqrt(R)
# H0 = dct(Matrix(I,K,K),1)'/sqrt(R)
h0 = [reshape(H0[:,k],rr,rr) for k in 1:K]
x = [randn(N1,N2) for l in 1:L]
λ = 0.0001
maxiters = 30

# Test using p handcrafted filters
p = 3

ph7,niter,obj_fnc_vals, H_trace,H_convergence =
    CAOL2(x,h0[p+1:end],λ,maxiters=maxiters,debug=true)
@time CAOL2(x,h0[p+1:end],λ,maxiters=maxiters,debug=true)
@time CAOL2(x,h0[p+1:end],λ,maxiters=maxiters,debug=false)

h, (obj,Hdiff), Hs = CAOL(x,h0[p+1:end],λ,maxiters=niter,tol=1e-13,trace=true)
@time CAOL(x,h0[p+1:end],λ,maxiters=niter,tol=1e-13,trace=true)
@time CAOL(x,h0[p+1:end],λ,maxiters=niter,tol=1e-13,trace=false)
@show h == ph7
@show length(obj) == niter
@show obj ≈ obj_fnc_vals
@show Hs == H_trace
@show H_convergence ≈ Hdiff

@time CAOL2(x,h0[p+1:end],λ,maxiters=200,debug=true)
@time CAOL(x,h0[p+1:end],λ,maxiters=200,tol=0,trace=true)

@time CAOL2(x,h0[p+1:end],λ,maxiters=200,debug=false);
@time CAOL(x,h0[p+1:end],λ,maxiters=200,tol=0,trace=false);

CAOL(x,H0[:,p+1:end],size(h0[1]),λ,maxiters=30)
