Further extensions:
+ handle complex-valued data (check for conjugate needed in imfilter calls!)
+ handle color images
+ signature that uses a default initialization, e.g., something like
```
CAOL(x ::Vector of arrays, λ; filt_type=:DCT, filt_size=get_default_for_filt_type(filt_type).*dimensions of x, p=0, maxiters=2000, tol=1e-13, trace=false)
```

Longer term questions:
+ good defaults for lambda - maybe based on some norm of x and the size of the filters?
+ handle image input stream instead of a vector, e.g., something that generates signals on the fly (maybe apply a mini-batch approach?)
+ (super long term) compile package and provide matlab/python wrappers?
+ we currently precompute padded images since it seems that imfilter! would create them otherwise and those allocations add up with all the calls to imfilter!. maybe contribute to ImageFiltering to have a version that carefully indexes to avoid creating padded copies?
+ handle images with different sizes? could probably work in principle
+ handle images with offset axes

**Some general notes on imfilter! (17 May 2019):**
I tried setting imfilter algorithms to FFT - seemed to yield more allocation
and slower runtime, didn't investigate more.
Also tried FIRTiled in hopes of getting multi-threading but it didn't seem to
perform much better than setting to FIR even though Julia was started with
`export JULIA_NUM_THREADS=4`, didn't investigate more.
Would be nice to understand the tradeoffs with these options more.
Some rough profiling seems to suggest that most of the time is spent
on the two imfilter! calls.
