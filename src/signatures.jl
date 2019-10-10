## CAOL signatures

SignalBank{N}  = AbstractVector{<:AbstractArray{<:Any,N}}    # List of arrays
SignalTuple{N} = Tuple{<:AbstractMatrix,NTuple{N,<:Integer}} # Columns with shape

# x::SignalBank, h0::SignalBank
function CAOL(x::SignalBank{N},λ::Real,h0::SignalBank{N};
        p=0,maxiters=2000,tol=1e-13,trace=false) where N
    @assert p < length(h0)

    H0,R = _filtermatrix(h0[p+1:end])
    if !trace
        H = _CAOL(x,λ,H0,R,maxiters,tol)
        return [h0[1:p]; Array.(_filterlist(H,R))]
    else
        H, Htrace, objtrace, Hdifftrace = _CAOLtrace(x,λ,H0,R,maxiters,tol)
        return [h0[1:p]; Array.(_filterlist(H,R))], _filterlist.(Htrace), objtrace, Hdifftrace
    end
end

# x::SignalBank, (H0,R)::SignalTuple
function CAOL(x::SignalBank{N},λ::Real,(H0,R)::SignalTuple{N};
        p=0,maxiters=2000,tol=1e-13,trace=false) where N
    @assert p < size(H0,2)

    if !trace
        H = _CAOL(x,λ,H0[:,p+1:end],R,maxiters,tol)
        return [H0[:,1:p] H]
    else
        H, Htrace, objtrace, Hdifftrace = _CAOLtrace(x,λ,H0[:,p+1:end],R,maxiters,tol)
        return [H0[:,1:p] H], Htrace, objtrace, Hdifftrace
    end
end

# X::AbstractArray
_extractsignals(X::AbstractArray) = collect(eachslice(X,dims=ndims(X)))
CAOL(X::AbstractArray,λ::Real,h0::SignalBank; p=0,maxiters=2000,tol=1e-13,trace=false) =
    CAOL(_extractsignals(X),λ,h0; p=p,maxiters=maxiters,tol=tol,trace=trace)
CAOL(X::AbstractArray,λ::Real,H0R::SignalTuple; p=0,maxiters=2000,tol=1e-13,trace=false) =
    CAOL(_extractsignals(X),λ,H0R; p=p,maxiters=maxiters,tol=tol,trace=trace)
