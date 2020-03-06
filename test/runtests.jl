using Test, ConvolutionalOperatorLearning
using Random

# Load reference implementation
include(joinpath(@__DIR__,"Reference.jl"))

# Random gaussian images (2D)
@testset "Random gaussian images (2D): vectorized x" begin
	rng = MersenneTwister(1)
	x = [randn(rng,16,16) for _ in 1:4] # vector form

	# 3 x 3 filters
	R = (3,3)
	H0 = generatefilters(:DCT,R,form=:matrix)
	λ, iters, tol = 1e-4, 4, 0.0
	for p in 0:2:8
		@testset "3 x 3 filters (p = $p)" begin
			refH, refHtrace, refobjtrace, refHdifftrace =
				Reference.CAOL(x,λ,H0[:,p+1:size(H0,2)],R,iters,tol)
			H, Htrace, objtrace, Hdifftrace =
				CAOL(x,λ,(H0[:,p+1:size(H0,2)],R),maxiters=iters,tol=tol,trace=true)

			@test H          == refH
			@test Htrace     == refHtrace
			@test objtrace   == refobjtrace
			@test Hdifftrace == refHdifftrace

			H = CAOL(x,λ,(H0[:,p+1:size(H0,2)],R),maxiters=iters,tol=tol)
			@test H == refH
		end
	end

	# 4 x 4 filters in list form
	R = (4,4)
	H0 = generatefilters(:DCT,R,form=:matrix)
	h0 = generatefilters(:DCT,R,form=:list)
	λ, iters, tol = 1e-4, 4, 0.0
	for p in 0:5:15
		@testset "4 x 4 filters (p = $p)" begin
			refH, refHtrace, refobjtrace, refHdifftrace =
				Reference.CAOL(x,λ,H0[:,p+1:size(H0,2)],R,iters,tol)
			h, htrace, objtrace, Hdifftrace =
				CAOL(x,λ,h0[p+1:end],maxiters=iters,tol=tol,trace=true)

			@test ConvolutionalOperatorLearning._filtermatrix(h)[1]          				== refH
			@test [ConvolutionalOperatorLearning._filtermatrix(ht)[1] for ht in htrace]   == refHtrace
			@test objtrace   == refobjtrace
			@test Hdifftrace == refHdifftrace

			H = CAOL(x,λ,(H0[:,p+1:size(H0,2)],R),maxiters=iters,tol=tol)
			@test H == refH
		end
	end

	# 2 x 3 filters: Reference test does not support rectangular filters, so
	# test that cost function is decending
	R = (2,3)
	H0 = generatefilters(:DCT,R,form=:matrix)
	λ, iters, tol = 1e-4, 100, 0.0
	for p in 0:2:4
		@testset "2 x 3 filters (p = $p)" begin
			H, Htrace, objtrace, Hdifftrace =
				CAOL(x,λ,(H0[:,p+1:size(H0,2)],R),maxiters=iters,tol=tol,trace=true)
			@test all(diff(objtrace) .<= 100*eps(objtrace[1]))

			H2 = CAOL(x,λ,(H0[:,p+1:size(H0,2)],R),maxiters=iters,tol=tol)
			@test H == H2
		end
	end

	# 4 x 3 filters in vectorized form
	R = (4,3)
	H0 = generatefilters(:DCT,R,form=:matrix)
	h0 = generatefilters(:DCT,R,form=:list)
	λ, iters, tol = 1e-4, 100, 0.0
	for p in 0:2:4
		@testset "4 x 3 filters in vectorized form (p = $p)" begin
			h, htrace, objtrace, Hdifftrace =
				CAOL(x,λ,h0[p+1:end],maxiters=iters,tol=tol,trace=true)
			@test all(diff(objtrace) .<= 100*eps(objtrace[1]))

			H2 = CAOL(x,λ,(H0[:,p+1:size(H0,2)],R),maxiters=iters,tol=tol)
			@test ConvolutionalOperatorLearning._filtermatrix(h)[1] == H2
		end
	end

	# 4 x 1 filters - verify error is thrown for 1d filters with 2d signals
	R = (4,)
	H0 = generatefilters(:DCT,R,form=:matrix)
	h0 = generatefilters(:DCT,R,form=:list)
	λ, iters, tol = 1e-4, 4, 0.0
	for p in 0:2:4
		@testset "1d filters with 2d sigmals (p = $p)" begin
			@test_throws AssertionError CAOL(x,λ,(H0[:,p+1:size(H0,2)],R),maxiters=iters,tol=tol,trace=true)
			@test_throws AssertionError CAOL(x,λ,(H0[:,p+1:size(H0,2)],R),maxiters=iters,tol=tol)
			@test_throws AssertionError CAOL(x,λ,h0[p+1:end],maxiters=iters,tol=tol,trace=true)
			@test_throws AssertionError CAOL(x,λ,h0[p+1:end],maxiters=iters,tol=tol)
		end
	end
end

# Random overlaid squares (2D); all X's are now in matrix form
@testset "Random overlaid squares (2D): matrix X" begin
	rng = MersenneTwister(1)
	square((x,y),w,n) = [y-w<i<y+w && x-w<j<x+w ? 1.0 : 0.0 for i in 1:n, j in 1:n]
	randsquare(rng,wrange,n,T) = square(
		rand(rng,1+first(wrange):n-first(wrange),2),
		rand(rng,wrange), n)
	x = [sum(randsquare(rng,2:3,16,1) for _ in 1:5) for _ in 1:4]
	X = cat(x..., dims=ndims(x[1])+1) # in matrix form

	# 3 x 3 filters
	R = (3,3)
	H0 = generatefilters(:DCT,R,form=:matrix)
	λ, iters, tol = 1e-2, 4, 0.0
	for p in 0:2:8
		@testset "3 x 3 filters (p = $p)" begin
			refH, refHtrace, refobjtrace, refHdifftrace =
				Reference.CAOL(x,λ,H0[:,p+1:size(H0,2)],R,iters,tol)
			H, Htrace, objtrace, Hdifftrace =
				CAOL(X,λ,(H0[:,p+1:size(H0,2)],R),maxiters=iters,tol=tol,trace=true)
			@test H          == refH
			@test Htrace     == refHtrace
			@test objtrace   == refobjtrace
			@test Hdifftrace == refHdifftrace

			H = CAOL(X,λ,(H0[:,p+1:size(H0,2)],R),maxiters=iters,tol=tol)
			@test H == refH
		end
	end

	# 4 x 4 filters in list form
	R = (4,4)
	H0 = generatefilters(:DCT,R,form=:matrix)
	h0 = generatefilters(:DCT,R,form=:list)
	λ, iters, tol = 1e-2, 4, 0.0
	for p in 0:4:15
		@testset "4 x 4 filters (p = $p)" begin
			refH, refHtrace, refobjtrace, refHdifftrace =
				Reference.CAOL(x,λ,H0[:,p+1:size(H0,2)],R,iters,tol)
			h, htrace, objtrace, Hdifftrace =
				CAOL(X,λ,h0[p+1:end],maxiters=iters,tol=tol,trace=true)

			@test ConvolutionalOperatorLearning._filtermatrix(h)[1]          				== refH
			@test [ConvolutionalOperatorLearning._filtermatrix(ht)[1] for ht in htrace]   == refHtrace
			@test objtrace   == refobjtrace
			@test Hdifftrace == refHdifftrace

			H = CAOL(X,λ,(H0[:,p+1:size(H0,2)],R),maxiters=iters,tol=tol)
			@test H == refH
		end
	end

	# 4 x 2 filters
	R = (4,2)
	H0 = generatefilters(:DCT,R,form=:matrix)
	λ, iters, tol = 1e-2, 100, 0.0
	for p in 0:2:6
		@testset "4 x 2 filters (p = $p)" begin
			H, Htrace, objtrace, Hdifftrace =
				CAOL(X,λ,(H0[:,p+1:size(H0,2)],R),maxiters=iters,tol=tol,trace=true)
			@test all(diff(objtrace) .<= 100*eps(objtrace[1]))

			H2 = CAOL(X,λ,(H0[:,p+1:size(H0,2)],R),maxiters=iters,tol=tol)
			@test H == H2
		end
	end

	# 3 x 4 filters in vectorized form
	R = (3,4)
	H0 = generatefilters(:DCT,R,form=:matrix)
	h0 = generatefilters(:DCT,R,form=:list)
	λ, iters, tol = 1e-4, 100, 0.0
	for p in 0:2:4
		@testset "3 x 4 filters in vectorized form (p = $p)" begin
			h, htrace, objtrace, Hdifftrace =
				CAOL(X,λ,h0[p+1:end],maxiters=iters,tol=tol,trace=true)
			@test all(diff(objtrace) .<= 100*eps(objtrace[1]))

			H2 = CAOL(X,λ,(H0[:,p+1:size(H0,2)],R),maxiters=iters,tol=tol)
			@test ConvolutionalOperatorLearning._filtermatrix(h)[1] == H2
		end
	end

	# 4 x 1 filters - verify error is thrown for 1d filters with 2d signals
	R = (4,)
	H0 = generatefilters(:DCT,R,form=:matrix)
	h0 = generatefilters(:DCT,R,form=:list)
	λ, iters, tol = 1e-4, 4, 0.0
	for p in 0:2:4
		@testset "1d filters with 2d signals (p = $p)" begin
			@test_throws AssertionError CAOL(X,λ,(H0[:,p+1:size(H0,2)],R),maxiters=iters,tol=tol,trace=true)
			@test_throws AssertionError CAOL(X,λ,(H0[:,p+1:size(H0,2)],R),maxiters=iters,tol=tol)
			@test_throws AssertionError CAOL(X,λ,h0,maxiters=iters,tol=tol,trace=true)
			@test_throws AssertionError CAOL(X,λ,h0,maxiters=iters,tol=tol)
		end
	end

	# 2 x 2 x 2 filters - verify error is thrown for 3d filters with 2d signals
	R = (2,2,2)
	H0 = generatefilters(:DCT,R,form=:matrix)
	h0 = generatefilters(:DCT,R,form=:list)
	λ, iters, tol = 1e-4, 4, 0.0
	for p in 0:2:4
		@testset "2 x 3 filters (p = $p)" begin
			@test_throws AssertionError CAOL(X,λ,(H0[:,p+1:size(H0,2)],R),maxiters=iters,tol=tol,trace=true)
			@test_throws AssertionError CAOL(X,λ,(H0[:,p+1:size(H0,2)],R))
			@test_throws AssertionError CAOL(X,λ,h0,trace=true)
			@test_throws AssertionError CAOL(X,λ,h0,maxiters=iters,tol=tol)
		end
	end
end

# Random gaussian signals (1D)
@testset "Random gaussian signals (1D)" begin
	rng = MersenneTwister(1)
	x = [randn(rng,16) for _ in 1:4]
	X = cat(x..., dims=ndims(x[1])+1) # in matrix form

	# length 3 filters
	R = (3,)
	H0 = generatefilters(:DCT,R,form=:matrix)
	λ, iters, tol = 1e-4, 4, 0.0
	for p in 0:1:2
		@testset "3 x 1 filters (p = $p)" begin
			refH, refHtrace, refobjtrace, refHdifftrace =
				Reference.CAOL(x,λ,H0[:,p+1:size(H0,2)],R,iters,tol)
			H, Htrace, objtrace, Hdifftrace =
				CAOL(x,λ,(H0[:,p+1:size(H0,2)],R),maxiters=iters,tol=tol,trace=true)

			@test H          == refH
			@test Htrace     == refHtrace
			@test objtrace   == refobjtrace
			@test Hdifftrace == refHdifftrace

			H = CAOL(x,λ,(H0[:,p+1:size(H0,2)],R),maxiters=iters,tol=tol)
			@test H == refH
		end
	end

	# length 3 filters with data in matrix form
	R = (3,)
	H0 = generatefilters(:DCT,R,form=:matrix)
	λ, iters, tol = 1e-4, 4, 0.0
	for p in 0:1:2
		@testset "3 x 1 filters, data in matrix form (p = $p)" begin
			refH, refHtrace, refobjtrace, refHdifftrace =
				Reference.CAOL(x,λ,H0[:,p+1:size(H0,2)],R,iters,tol)
			H, Htrace, objtrace, Hdifftrace =
				CAOL(X,λ,(H0[:,p+1:size(H0,2)],R),maxiters=iters,tol=tol,trace=true)

			@test H          == refH
			@test Htrace     == refHtrace
			@test objtrace   == refobjtrace
			@test Hdifftrace == refHdifftrace

			H = CAOL(X,λ,(H0[:,p+1:size(H0,2)],R),maxiters=iters,tol=tol)
			@test H == refH
		end
	end

	# 6 x 1 filters
	R = (6,)
	H0 = generatefilters(:DCT,R,form=:matrix)
	λ, iters, tol = 1e-4, 4, 0.0
	for p in 0:2:4
		@testset "6 x 1 filters (p = $p)" begin
			refH, refHtrace, refobjtrace, refHdifftrace =
				Reference.CAOL(x,λ,H0[:,p+1:size(H0,2)],R,iters,tol)
			H, Htrace, objtrace, Hdifftrace =
				CAOL(x,λ,(H0[:,p+1:size(H0,2)],R),maxiters=iters,tol=tol,trace=true)
			@test H          == refH
			@test Htrace     == refHtrace
			@test objtrace   == refobjtrace
			@test Hdifftrace == refHdifftrace

			H = CAOL(x,λ,(H0[:,p+1:size(H0,2)],R),maxiters=iters,tol=tol)
			@test H == refH
		end
	end

	# 6 x 1 filters in vectorized form
	R = (6,)
	H0 = generatefilters(:DCT,R,form=:matrix)
	h0 = generatefilters(:DCT,R,form=:list)
	λ, iters, tol = 1e-4, 4, 0.0
	for p in 0:2:4
		@testset "6 x 1 vectorized filters (p = $p)" begin
			refH, refHtrace, refobjtrace, refHdifftrace =
				Reference.CAOL(x,λ,H0[:,p+1:size(H0,2)],R,iters,tol)
			h, htrace, objtrace, Hdifftrace =
				CAOL(x,λ,h0[p+1:end],maxiters=iters,tol=tol,trace=true)

			@test ConvolutionalOperatorLearning._filtermatrix(h)[1]          				== refH
			@test [ConvolutionalOperatorLearning._filtermatrix(ht)[1] for ht in htrace]   == refHtrace
			@test objtrace   == refobjtrace
			@test Hdifftrace == refHdifftrace

			H = CAOL(x,λ,(H0[:,p+1:size(H0,2)],R),maxiters=iters,tol=tol)
			@test H == refH
		end
	end

	# 6 x 1 filters in vectorized form and data in matrix form
	R = (6,)
	H0 = generatefilters(:DCT,R,form=:matrix)
	h0 = generatefilters(:DCT,R,form=:list)
	λ, iters, tol = 1e-4, 4, 0.0
	for p in 0:2:4
		@testset "6 x 1 vectorized filters with data in matrix form (p = $p)" begin
			refH, refHtrace, refobjtrace, refHdifftrace =
				Reference.CAOL(x,λ,H0[:,p+1:size(H0,2)],R,iters,tol)
			h, htrace, objtrace, Hdifftrace =
				CAOL(X,λ,h0[p+1:end],maxiters=iters,tol=tol,trace=true)

			@test ConvolutionalOperatorLearning._filtermatrix(h)[1]          				== refH
			@test [ConvolutionalOperatorLearning._filtermatrix(ht)[1] for ht in htrace]   == refHtrace
			@test objtrace   == refobjtrace
			@test Hdifftrace == refHdifftrace

			H = CAOL(X,λ,(H0[:,p+1:size(H0,2)],R),maxiters=iters,tol=tol)
			@test H == refH
		end
	end

	# 2 x 2 filters - verify error is thrown for 2d filters with 1d signals
	R = (2,2)
	H0 = generatefilters(:DCT,R,form=:matrix)
	h0 = generatefilters(:DCT,R,form=:list)
	λ, iters, tol = 1e-4, 4, 0.0
	for p in 0:2:4
		@testset "2d filters with 1d signal (p = $p)" begin
			@test_throws AssertionError CAOL(x,λ,(H0[:,p+1:size(H0,2)],R),maxiters=iters,tol=tol,trace=true)
			@test_throws AssertionError CAOL(x,λ,(H0[:,p+1:size(H0,2)],R),maxiters=iters,tol=tol)
			@test_throws AssertionError CAOL(X,λ,(H0[:,p+1:size(H0,2)],R),maxiters=iters,tol=tol,trace=true)
			@test_throws AssertionError CAOL(X,λ,(H0[:,p+1:size(H0,2)],R),maxiters=iters,tol=tol)

			@test_throws AssertionError CAOL(x,λ,h0,maxiters=iters,tol=tol,trace=true)
			@test_throws AssertionError CAOL(x,λ,h0,maxiters=iters,tol=tol)
			@test_throws AssertionError CAOL(X,λ,h0,maxiters=iters,tol=tol,trace=true)
			@test_throws AssertionError CAOL(X,λ,h0,maxiters=iters,tol=tol)
		end
	end
end

# Random gaussian signals (3D)
@testset "Random gaussian signals (3D)" begin
	rng = MersenneTwister(1)
	x = [randn(rng,16,16,16) for _ in 1:4]

	# 2x3x4 filters in matrix form
	R = (2,3,4)
	H0 = generatefilters(:DCT,R,form=:matrix)
	λ, iters, tol = 1e-4, 4, 0.0
	for p in 0:5:20
		@testset "2x3x4 filters (p = $p)" begin
			refH, refHtrace, refobjtrace, refHdifftrace =
				Reference.CAOL(x,λ,H0[:,p+1:size(H0,2)],R,iters,tol)
			H, Htrace, objtrace, Hdifftrace =
				CAOL(x,λ,(H0[:,p+1:size(H0,2)],R),maxiters=iters,tol=tol,trace=true)
			@test H          == refH
			@test Htrace     == refHtrace
			@test objtrace   == refobjtrace
			@test Hdifftrace == refHdifftrace

			H = CAOL(x,λ,(H0[:,p+1:size(H0,2)],R),maxiters=iters,tol=tol)
			@test H == refH
		end
	end

	# 2 x 2 x 2 filters in list form
	R = (2,2,2)
	H0 = generatefilters(:DCT,R,form=:matrix)
	h0 = generatefilters(:DCT,R,form=:list)
	λ, iters, tol = 1e-4, 4, 0.0
	for p in 0:2:4
		@testset "4 x 4 filters (p = $p)" begin
			refH, refHtrace, refobjtrace, refHdifftrace =
				Reference.CAOL(x,λ,H0[:,p+1:size(H0,2)],R,iters,tol)
			h, htrace, objtrace, Hdifftrace =
				CAOL(x,λ,h0[p+1:end],maxiters=iters,tol=tol,trace=true)

			@test ConvolutionalOperatorLearning._filtermatrix(h)[1]          				== refH
			@test [ConvolutionalOperatorLearning._filtermatrix(ht)[1] for ht in htrace]   == refHtrace
			@test objtrace   == refobjtrace
			@test Hdifftrace == refHdifftrace

			H = CAOL(x,λ,(H0[:,p+1:size(H0,2)],R),maxiters=iters,tol=tol)
			@test H == refH
		end
	end

	# 2 x 2 filters - verify error is thrown for 2d filters with 3d signals
	R = (2,2)
	H0 = generatefilters(:DCT,R,form=:matrix)
	λ, iters, tol = 1e-4, 4, 0.0
	for p in 0:2:4
		@testset "2 x 3 filters (p = $p)" begin
			@test_throws AssertionError CAOL(x,λ,(H0[:,p+1:size(H0,2)],R),maxiters=iters,tol=tol,trace=true)
			@test_throws AssertionError CAOL(x,λ,(H0[:,p+1:size(H0,2)],R),maxiters=iters,tol=tol)
		end
	end

	# 2x2x2x2 filters - verify error is thrown for 4d filters with 3d signals
	R = (2,2,2,2)
	H0 = generatefilters(:DCT,R,form=:matrix)
	λ, iters, tol = 1e-4, 4, 0.0
	for p in 0:2:4
		@testset "2 x 3 filters (p = $p)" begin
			@test_throws AssertionError CAOL(x,λ,(H0[:,p+1:size(H0,2)],R),maxiters=iters,tol=tol,trace=true)
			@test_throws AssertionError CAOL(x,λ,(H0[:,p+1:size(H0,2)],R),maxiters=iters,tol=tol)
		end
	end
end

# need to test:
# + termination condition (make sure all agree)
@testset "Termination condition test" begin
	rng = MersenneTwister(1)
	square((x,y),w,n) = [y-w<i<y+w && x-w<j<x+w ? 1.0 : 0.0 for i in 1:n, j in 1:n]
	randsquare(rng,wrange,n,T) = square(
		rand(rng,1+first(wrange):n-first(wrange),2),
		rand(rng,wrange), n)
	x = [sum(randsquare(rng,2:3,16,1) for _ in 1:5) for _ in 1:4]
	X = cat(x..., dims=ndims(x[1])+1) # in matrix form

	# 3 x 3 filters
	R = (3,3)
	H0 = generatefilters(:DCT,R,form=:matrix)
	h0 = generatefilters(:DCT,R,form=:list)
	λ, iters, tol = 1e-2, 10000, 1e-8
	for p in 0:2:8
		@testset "3 x 3 filters (p = $p)" begin
			refH, refHtrace, refobjtrace, refHdifftrace =
				Reference.CAOL(x,λ,H0[:,p+1:size(H0,2)],R,iters,tol)

			Hx, Hx_Htrace, Hx_objtrace, Hx_Hdifftrace =
				CAOL(x,λ,(H0[:,p+1:size(H0,2)],R),maxiters=iters,tol=tol,trace=true)

			HX, HX_Htrace, HX_objtrace, HX_Hdifftrace =
				CAOL(X,λ,(H0[:,p+1:size(H0,2)],R),maxiters=iters,tol=tol,trace=true)

			hx, hx_Htrace, hx_objtrace, hx_Hdifftrace =
				CAOL(x,λ,h0[p+1:end],maxiters=iters,tol=tol,trace=true)
			hx = ConvolutionalOperatorLearning._filtermatrix(hx)[1]
			hx_Htrace = [ConvolutionalOperatorLearning._filtermatrix(ht)[1] for ht in hx_Htrace]

			hX, hX_Htrace, hX_objtrace, hX_Hdifftrace =
				CAOL(X,λ,h0[p+1:end],maxiters=iters,tol=tol,trace=true)
			hX = ConvolutionalOperatorLearning._filtermatrix(hX)[1]
			hX_Htrace = [ConvolutionalOperatorLearning._filtermatrix(ht)[1] for ht in hX_Htrace]

			@test length(hX_Hdifftrace) < iters # otherwise not testing convergence!

			@test refH 		    == Hx 			 == HX            == hX 		   == hx
			@test refHtrace     == Hx_Htrace     == HX_Htrace     == hX_Htrace     == hx_Htrace
			@test refobjtrace   == Hx_objtrace   == HX_objtrace   == hX_objtrace   == hx_objtrace
			@test refHdifftrace == Hx_Hdifftrace == HX_Hdifftrace == hX_Hdifftrace == hx_Hdifftrace

			H = CAOL(X,λ,(H0[:,p+1:size(H0,2)],R),maxiters=iters,tol=tol)
			@test H == refH
		end
	end
end
