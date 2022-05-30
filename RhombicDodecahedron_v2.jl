# using Plots, WriteVTK
using LinearAlgebra

```Set of vertices and vertex 2 face connectivity ```
function RhombicDodecahedronGeometry( )
    nfac  = 12
    p     = zeros(14,3)             # vertices 
    f2v   = zeros(Int64,nfac,3)     # face 2 vertices
    # Vertices set 1
    p[1,:] = [1,1,1] 
    p[2,:] = [1,1,−1]
    p[3,:] = [1,−1,1]
    p[4,:] = [1,−1,−1]
    p[5,:] = [−1,1,1]
    p[6,:] = [−1,1,−1]
    p[7,:] = [−1,−1,1]
    p[8,:] = [−1,−1,−1]
    # Vertices set 2
    p[9,:]  = [0,0,2]
    p[10,:] = [0,0,−2]
    p[11,:] = [0,2,0]
    p[12,:] = [0,−2,0]
    p[13,:] = [2,0,0]
    p[14,:] = [−2,0,0]
    # Face connectivity
    f2v[ 1,:] = [14, 11, 5] 
    f2v[ 2,:] = [9, 14, 5] 
    f2v[ 3,:] = [7, 9, 12] 
    f2v[ 4,:] = [13, 9, 1 ]  
    f2v[ 5,:] = [11, 13, 1 ] 
    f2v[ 6,:] = [11, 1, 5 ] 
    f2v[ 7,:] = [11, 10, 2 ] 
    f2v[ 8,:] = [10, 13, 2 ] 
    f2v[ 9,:] = [10, 6, 14 ] 
    f2v[10,:] = [10, 8, 12 ] 
    f2v[11,:] = [4, 12, 13 ] 
    f2v[12,:] = [14, 12, 8 ] 
    return p, f2v
end

```Compute fit```
function FitRhombicDocdecahedron( p, f2v )
    nfac = size(f2v,1)
    f    = zeros(nfac,4)           # fit for each face
    for ifac=1:nfac
        P1, P2, P3 = p[f2v[ifac,1],:], p[f2v[ifac,2],:], p[f2v[ifac,3],:]
        v1 = P2 .- P1
        v2 = P2 .- P3
        X  = cross(v1,v2)
        k  = -P1'*X
        f[ifac,1], f[ifac,2], f[ifac,3], f[ifac,4] = X[1], X[2], X[3], k
    end
    return f
end

```Identify if a point is inside the rhombic dodecahedron```
function IsInRhombicDodecahedron( X, Y, Z, f )
    nfac = size(f,1) 
    F    = true
    for ifac=1:nfac
        F = F && (f[ifac,1]*X +  f[ifac,2]*Y + f[ifac,3]*Z + f[ifac,4]) < 0.0
    end
    return F
end

function main( n )
    # Domain
    xmin, xmax = -2.5e-2, 2.5e-2
    ymin, ymax = -2.5e-2, 2.5e-2
    zmin, zmax = -2.5e-2, 2.5e-2
    # Centroid location and embedding radius
    x0, y0, z0 = 1.e-3, -2e-3, 3.0e-3
    r          = 2.0e-2
    # Spatial discretisation
    nx, ny, nz = n*8+1, n*8+1, n*8+1
    dx = (xmax-xmin)/nx
    dy = (ymax-ymin)/ny
    dz = (zmax-zmin)/nz
    xc = LinRange(xmin+dx/2, xmax-dx/2, nx)
    yc = LinRange(ymin+dy/2, ymax-dy/2, ny)
    zc = LinRange(zmin+dz/2, zmax-dz/2, nz)
    # Arrays
    phase = zeros(Float64,nx,ny,nz) # phase type
    # Geometric data for rhombic dodecahedron
    p, f2v = RhombicDodecahedronGeometry( )
    # Scale to desired radius
    p    .*= r/2.0
    # Call fitting function
    f  = FitRhombicDocdecahedron( p, f2v )
    # Loop through cells and check if in or out
    @Threads.threads for k=1:nz
        @inbounds for j=1:ny
            for i=1:nx
                if IsInRhombicDodecahedron( xc[i]-x0, yc[j]-y0, zc[k]-z0, f )
                    phase[i,j,k] = 1.0
                end
            end
        end
    end
    # Quick check with Plots
    xmid, ymid, zmid = nx÷2+1, ny÷2+1, nz÷2+1
    p1 = heatmap(xc,zc,phase[:,ymid,:]') # xz
    p2 = heatmap(zc,yc,phase[xmid,:,:]') # yz
    p3 = heatmap(xc,yc,phase[:,:,zmid]') # xy
    display(plot(p1,p2,p3))
    # Output for Paraview
    vtk_grid("RhombicDodecahedron", xc, yc, zc) do vtk
        vtk["phase"] = phase
    end
end

# main( 12 )

