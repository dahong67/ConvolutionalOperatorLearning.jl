using BenchmarkTools, ConvolutionalOperatorLearning
using Random

const SUITE = BenchmarkGroup()

## CAOL 2D benchmarks: 62 images (128 x 128)
SUITE["CAOL 2D"] = BenchmarkGroup()

rng = MersenneTwister(1)
x = [randn(rng,128,128) for l in 1:62]

R = (3,3)
H0 = generatefilters(:DCT,R,form=:matrix)
λ = 0.0001

maxiters = 30

SUITE["CAOL 2D"]["notrace,30"]  = @benchmarkable CAOL($x,$λ,($H0,$R),maxiters=30)
SUITE["CAOL 2D"]["trace,30"]    = @benchmarkable CAOL($x,$λ,($H0,$R),maxiters=30,trace=true)

SUITE["CAOL 2D"]["notrace,200"] = @benchmarkable CAOL($x,$λ,($H0,$R),maxiters=200)
SUITE["CAOL 2D"]["trace,200"]   = @benchmarkable CAOL($x,$λ,($H0,$R),maxiters=200,trace=true)

for p in 2:2:size(H0,2)-1
    SUITE["CAOL 2D"]["p=$p"] = @benchmarkable CAOL($x,$λ,($H0,$R),p=$p,maxiters=30)
end


# todo: add more benchmarks, incorporate in package ci appropriately?
