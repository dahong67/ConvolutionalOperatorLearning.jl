using Test, ConvolutionalAnalysisOperatorLearning

# Load reference implementation
include(joinpath(@__DIR__,"reference.jl"))

# Random gaussian images (2D)
@testset "Random gaussian images (2D)" begin
	x = [randn(128,128) for _ in 1:62]

	# 3 x 3 filters
	@testset "3 x 3 filters" begin
		R = (3,3)
		H0 = generatefilters(:DCT,R)

		λ, iters = 1e-4, 30
		for p in 0:2:8
			refH, (refobjtrace, refHdifftrace), refHtrace =
				CAOLprev(x,H0[:,p+1:end],R,λ,maxiters=iters,tol=1e-13,trace=true)
			H, Htrace, objtrace, Hdifftrace =
				CAOL(x,λ,(H0[:,p+1:end],R),maxiters=iters,tol=1e-13,trace=true)

			@test H          == refH
			@test Htrace     == refHtrace
			@test objtrace   == refobjtrace
			@test Hdifftrace == refHdifftrace

			H = CAOL(x,λ,(H0[:,p+1:end],R),maxiters=iters,tol=1e-13)
			@test H == refH
		end
	end
end

# need to test:
# + all signatures
# + termination condition
# + rectangular filters
# + even-sized filters
# + 1d filters/data
# + 3d filters/data
# + filter/data dimension mismatch, e.g., 1d filter but 2d data
# + something related to speed / memory use
