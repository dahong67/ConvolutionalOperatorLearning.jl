using Test, ConvolutionalAnalysisOperatorLearning, FFTW, LinearAlgebra

@testset "Dummy tests" begin
	@test true == true
end
# todo put actual tests

# need to test: various signatures, termination condition

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
	println("# Trace = false")
	@time CAOLprev(x,H0[:,3+1:end],(rr,rr),λ,maxiters=maxiters,tol=1e-13,trace=false)
	println("--->")
	@time CAOL(x,λ,(H0[:,3+1:end],(rr,rr)),maxiters=maxiters,tol=1e-13,trace=false)

	println("# Trace = true")
	@time CAOLprev(x,H0[:,3+1:end],(rr,rr),λ,maxiters=maxiters,tol=1e-13,trace=true)
	println("--->")
	@time CAOL(x,λ,(H0[:,3+1:end],(rr,rr)),maxiters=maxiters,tol=1e-13,trace=true)

	println("# Longer run, trace = false")
	@time CAOLprev(x,H0[:,3+1:end],(rr,rr),λ,maxiters=200,tol=1e-13,trace=false)
	println("--->")
	@time CAOL(x,λ,(H0[:,3+1:end],(rr,rr)),maxiters=200,tol=1e-13,trace=false)

	println("# Longer run, trace = true")
	@time CAOLprev(x,H0[:,3+1:end],(rr,rr),λ,maxiters=200,tol=1e-13,trace=true)
	println("--->")
	@time CAOL(x,λ,(H0[:,3+1:end],(rr,rr)),maxiters=200,tol=1e-13,trace=true)
end
