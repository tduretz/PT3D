using Plots, WriteVTK

n = 8
nx, ny, nz = n*8+1, n*8+1, n*8+1

phase = zeros(Float64,nx,ny,nz)

xmid = nx÷2+1
ymid = ny÷2+1
if ymid==0 ymid=1 end
zmid = nz÷2+1
phase[xmid,ymid,zmid] = 1.0

# xz plane
phase[xmid+1:end-1, ymid, zmid] .= 1.0
phase[2:xmid-1,   ymid, zmid]   .= 1.0

phase[xmid,   ymid, zmid+1:end-1] .= 1.0
phase[xmid,   ymid, 2:zmid-1  ]   .= 1.0

# yz plane
phase[xmid, ymid+1:end-1, zmid]  .= 1.0
phase[xmid, 2:ymid-1,     zmid]  .= 1.0

phase[xmid,  ymid, zmid+1:end-1] .= 1.0
phase[xmid,  ymid, 2:zmid-1    ] .= 1.0

####################
##### XZ

# loop in E
for i=xmid+1:nx
    for k=2:nz-1
        if (phase[i-1,ymid,k+1]!=0) && (phase[i-1,ymid,k-1]!=0)
            phase[i,ymid,k]=1.0 
        end
    end
end

# loop in W
for i=xmid-1:-1:1
    for k=2:nz-1
        if (phase[i+1,ymid,k+1]!=0) && (phase[i+1,ymid,k-1]!=0)
            phase[i,ymid,k]=1.0 
        end
    end
end

####################
##### YZ

# loop in back
for k=zmid+1:nz
    for j=2:ny-1
        if (phase[xmid,j+1,k-1]!=0) && (phase[xmid,j-1,k-1]!=0)
            phase[xmid,j,k]=1.0
        end
    end
end

# loop in front
for k=zmid-1:-1:1
    for j=2:ny-1
        if (phase[xmid,j+1,k+1]!=0) && (phase[xmid,j-1,k+1]!=0)
            phase[xmid,j,k]=1.0 
        end
    end
end

####################
##### XY

# loop in N
for i=xmid+1:nx
    for j=2:ny-1
        if (phase[i-1,j+1,zmid]!=0) && (phase[i-1,j-1,zmid]!=0)
            phase[i,j,zmid]=1.0 
        end
    end
end

# loop in S
for i=xmid-1:-1:1
    for j=2:ny-1
        if (phase[i+1,j+1,zmid]!=0) && (phase[i+1,j-1,zmid]!=0)
            phase[i,j,zmid]=1.0 
        end
    end
end


# loop in E
for i=xmid+1:nx
    for j=2:ny-1
        for k=2:nz-1
            if ( phase[i-1,j+1,k+1]==1 && phase[i-1,j-1,k-1]==1 && phase[i-1,j-1,k+1]==1 && phase[i-1,j+1,k-1]==1 )
                phase[i,j,k]=1.0 
            end
        end
    end
end

p1 = heatmap(phase[:,ymid,:]') #xz
p2 = heatmap(phase[xmid,:,:]') #yz
p3 = heatmap(phase[:,:,zmid]') #xy
# display(p1)
display(plot(p1,p2,p3))

x = LinRange(0.0,1.0,nx)
y = LinRange(0.0,1.0,ny)
z = LinRange(0.0,1.0,nz)

vtk_grid("RhombicDodecahedron", x, y, z) do vtk
    vtk["phase"] = phase
end

