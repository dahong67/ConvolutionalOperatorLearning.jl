## Initialization Filters
using LinearAlgebra, FFTW

"""
dims is the size of the filters
e.g., dims=(3,2) will give you 2 dimensional filters, each of size 3x2

These routines always return prod(dims) number of filters

:DCT filters are ordered from low to high frequency
"""
generatefilters(type,dims;form=:list) = _generatefilters(Val(type),dims,Val(form))

# DCT
_dctmatrix(n) = dct(Matrix(I,(n,n)),1)'
function _generatefilters(::Val{:DCT},dims,::Val{:list})
    H = _generatefilters(Val(:DCT),dims,Val(:matrix))
    return Array.(_filterlist(H,dims))
end
function _generatefilters(::Val{:DCT},dims,::Val{:matrix})
    @assert length(dims) >= 1  "Must provide at least one dimention for filter initialization."
    @assert all(dims .>= 1) "All filter dimensions must have positive size"
    H = _dctmatrix(dims[1])
    for i in 2:length(dims)
        H = kron(_dctmatrix(dims[i]), H)
    end
    @assert H'H ≈ I "DCT filter generation led to non-orthogonal filters"
    h = _reordersnake(Array.(_filterlist(H,dims)), dims)
    return  _filtermatrix(h)[1] ./ sqrt(prod(dims))
end

# Helper functions
function _reordersnake(h, dims)
    freqmeasure = zeros(dims)
    for i in 1:length(dims)
        orderdims = [i; 1:(i-1); (i+1):length(dims)]
        reorderdims = [2:i; 1; (i+1):length(dims)]
        repdims = [1; collect(dims[orderdims][2:end])]
        temp = repeat(1:dims[i], outer=repdims)
        freqmeasure += permutedims(temp, reorderdims)
    end
    perminds = sortperm(vec(freqmeasure))
    return h[perminds]
end
