# PT3D 

## Description

PT3D is code for the simulation of non-linear thermo-mechanical processes in 3D. It is designed to run on both CPU (parallelised with Base.Threads) and on the GPU. 
PT3D relies on the package [ParallelStencil](https://github.com/omlins/ParallelStencil.jl).

The code can be used to simulate the coupled mechanical processed that occur during host-inclusion decompression (e.g., [Luisier et al, 20203](https://www.nature.com/articles/s41467-023-41310-w) ) in 3D.
The couplings between processes such as phases transformation and frictional plasticity are treated fully implicitely.

## Applications

Frictional strain localisation: reproduction of a model in [Duretz et al., 2019](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2019GC008531):

<img src="https://github.com/tduretz/PT3D/blob/main/gif/_Duretz19.gif" width="500" height="300" />

Decompression of a coesite inclusion in an elliptical shape host:

<img src="https://github.com/tduretz/PT3D/blob/main/gif/_QuartzCoesiteJulia_res4.gif" width="500" height="300" />

3D model of rhombic dodecahedron host (i.e. garnet) with spherical silica inclusions:

<img src="https://github.com/tduretz/PT3D/blob/main/gif/Results1.png" width="500" height="300" />
