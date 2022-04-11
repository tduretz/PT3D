using MAT, GLMakie

# Reading file
file = matopen( string(@__DIR__,"/Stokes3D.mat"), "r")
P    = read(file, "P")
n    = read(file, "res_fact")
close(file)

# Generate domain
Lx,  Ly,  Lz  =  1.0,  1.0,  1.0 
ncx, ncy, ncz = n*32, n*32, n*32
xv = LinRange(-Lx/2, Lx/2, ncx+1)
yv = LinRange(-Ly/2, Ly/2, ncy+1)
zv = LinRange(-Lz/2, Lz/2, ncz+1)
Δx, Δy, Δz = Lx/ncx, Ly/ncy, Lz/ncz
xce = LinRange(-Lx/2-Δx/2, Lx/2+Δx/2, ncx+2)
yce = LinRange(-Ly/2-Δy/2, Ly/2+Δy/2, ncy+2)
zce = LinRange(-Lz/2-Δz/2, Lz/2+Δz/2, ncz+2)

#####

# 3D slices, positions controlled by sliders 
# adapted from: 
# https://makie.juliaplots.org/v0.15.2/examples/plotting_functions/volumeslices/index.html

fig = Figure()
ax = LScene(fig[1, 1], scenekw=(show_axis=false,))

lsgrid = labelslidergrid!(
  fig,
  ["yz plane - x axis", "xz plane - y axis", "xy plane - z axis"],
  [1:length(xce), 1:length(yce), 1:length(zce)]
)
fig[2, 1] = lsgrid.layout

plt = volumeslices!(ax, xce, yce, zce, P)

# connect sliders to `volumeslices` update methods
sl_yz, sl_xz, sl_xy = lsgrid.sliders

on(sl_yz.value) do v; plt[:update_yz][](v) end
on(sl_xz.value) do v; plt[:update_xz][](v) end
on(sl_xy.value) do v; plt[:update_xy][](v) end

set_close_to!(sl_yz, .5length(xce))
set_close_to!(sl_xz, .5length(yce))
set_close_to!(sl_xy, .5length(zce))

display(fig)