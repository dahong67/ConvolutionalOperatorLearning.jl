## Initialization Filters

using LinearAlgebra, FFTW

generatefilters(type,dims;form=:list) = _generatefilters(Val(type),dims,Val(form))

# DCT
function _generatefilters(::Val{:DCT},dims,::Val{:list})
    H = _generatefilters(Val(:DCT),dims,Val(:matrix))
    return Array.(_filterlist(H,dims))
end
function _generatefilters(::Val{:DCT},dims,::Val{:matrix})
    @assert length(dims) == 2           "Only 2D DCT is implemented"      # todo
    @assert all(i->i==first(dims),dims) "Only square DCT is implemented"  # todo

    temp = dct(Matrix(I,dims),1)
    return kron(temp,temp)' / sqrt(prod(dims))
end
