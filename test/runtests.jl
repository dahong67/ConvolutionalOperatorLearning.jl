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
		H, (obj,Hdiff), Hs = CAOL(x,H0[:,p+1:end],(rr,rr),λ,maxiters=maxiters,tol=1e-13,trace=true)
		Hnew, Hsnew, objnew, Hdiffnew = CAOLnew(x,λ,(H0[:,p+1:end],(rr,rr)),maxiters=maxiters,tol=1e-13,trace=true)
		@test H == Hnew
		@test obj == objnew
		@test Hdiff == Hdiffnew
		@test Hs == Hsnew

		Hnew = CAOLnew(x,λ,(H0[:,p+1:end],(rr,rr)),maxiters=maxiters,tol=1e-13,trace=false)
		@test H == Hnew
	end
	println("# Trace = false")
	@time CAOL(x,H0[:,3+1:end],(rr,rr),λ,maxiters=maxiters,tol=1e-13,trace=false)
	println("--->")
	@time CAOLnew(x,λ,(H0[:,3+1:end],(rr,rr)),maxiters=maxiters,tol=1e-13,trace=false)

	println("# Trace = true")
	@time CAOL(x,H0[:,3+1:end],(rr,rr),λ,maxiters=maxiters,tol=1e-13,trace=true)
	println("--->")
	@time CAOLnew(x,λ,(H0[:,3+1:end],(rr,rr)),maxiters=maxiters,tol=1e-13,trace=true)

	println("# Longer run, trace = false")
	@time CAOL(x,H0[:,3+1:end],(rr,rr),λ,maxiters=200,tol=1e-13,trace=false)
	println("--->")
	@time CAOLnew(x,λ,(H0[:,3+1:end],(rr,rr)),maxiters=200,tol=1e-13,trace=false)

	println("# Longer run, trace = true")
	@time CAOL(x,H0[:,3+1:end],(rr,rr),λ,maxiters=200,tol=1e-13,trace=true)
	println("--->")
	@time CAOLnew(x,λ,(H0[:,3+1:end],(rr,rr)),maxiters=200,tol=1e-13,trace=true)
end
