## Utility functions

# Sum of square differences
sosdiff(a::Number,b::Number) = abs2(a-b)
sosdiff(A,B) = sum(ab -> sosdiff(ab...),zip(A,B))

# Hard thresholding
hard(x, beta) = abs(x) < beta ? zero(x) : x

# Partial objective
_obj(zlk,λ) = sum(z -> (abs(z) < sqrt(2λ) ? abs2(z)/2 : λ), zlk)

# Conversions between filter forms
_filtermatrix(hlist) =
    (hcat([vec(h) for h in hlist]...)::Matrix{eltype(first(hlist))},
     size(first(hlist)))
_filterlist(Hmatrix,R) = [reshape(h,map(n->1:n,R)) for h in eachcol(Hmatrix)]
