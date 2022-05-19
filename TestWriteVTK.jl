using  WriteVTK

x = 0:0.1:1
y = 0:0.2:1
z = -1:0.05:1

vtk_grid("fields", x, y, z) do vtk
    vtk["temperature"] = rand(length(x), length(y), length(z))
end