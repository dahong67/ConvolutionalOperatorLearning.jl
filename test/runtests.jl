using Test, ConvolutionalAnalysisOperatorLearning, FFTW, LinearAlgebra

@testset "Dummy tests" begin
	@test true == true
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

# Temporary tests that just compare against existing version
@testset "Regression tests" begin
	N1,N2,L = 128, 128, 62
	rr = 3
	R = rr^2
	K = R

	# H0 = Matrix(qr!(randn(R,K)).Q) / sqrt(R)
	H0 = dct(Matrix(I,K,K),1)'/sqrt(R)
	h0 = [reshape(H0[:,k],rr,rr) for k in 1:K]
	x = [randn(N1,N2) for l in 1:L]
	λ = 0.0001
	maxiters = 30

	# for p in 0:K-1
	for p in [3,7]
		pH, (pobj,pHdiff), pHs = CAOLprev(x,H0[:,p+1:end],(rr,rr),λ,maxiters=maxiters,tol=1e-13,trace=true)
		H, Hs, obj, Hdiff = CAOL(x,λ,(H0[:,p+1:end],(rr,rr)),maxiters=maxiters,tol=1e-13,trace=true)
		@test pH == H
		@test pobj == obj
		@test pHdiff == Hdiff
		@test pHs == Hs

		H = CAOL(x,λ,(H0[:,p+1:end],(rr,rr)),maxiters=maxiters,tol=1e-13,trace=false)
		@test pH == H
	end
end
