# V-E-P model
# rheology on centers ONLY
# viscosity with extended stencil (useful?)
# dt for 3D
using Printf, HDF5, LightXML
# using Plots
import Statistics: mean
import LinearAlgebra: norm
import SpecialFunctions: erfc
#-----------
const USE_GPU = true
const GPU_ID  = 1
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3);
else
    @init_parallel_stencil(Threads, Float64, 3);
end

include(joinpath(@__DIR__,"data_io.jl"))

#-----------
function main( n )
show_visu    = 0
write_out    = 1
write_nout   = 5
restart_from = 000
#----------- BENCHMARK 
nt            = 200
Δtr           = 0.125e11/2
ε_BG          = 0.0e-16
∇V_BG         = 1.0e-14
r             = 2/3*2e-3
βr            = 1.0/(17e10)
Gr            = 9e10
ρr            = 3300
ηr            = 1e22
Lx,  Ly,  Lz  = 5e-3,  5e-3,  5e-3 
ncx, ncy, ncz = n*32-2, n*32-2, n*32-2
BCtype        = :PureShear_xz
Pini          = 3.45e9
Pr            = Pini
dPr           = 5e7
Pt            = 2.77e9
dρinc         = 300.0
Ptens         = -2e7
#----------- for the 1D plot (post-processing only!) 
Pmin   = 1e8
Pmax   = Pini 
P_1d   = LinRange( Pmin, Pmax, 200 )
ρr_1d  = 0.8788ρr .- dρinc*1.0/2.0 .* erfc.( (P_1d.-Pt)./dPr ) 
ρ_1d   = ρr_1d .*exp.(βr.*(P_1d.-Pr))
ϕ      = 35.0
ψ      = 5.0
C      = 1e7
η_vp   = 1e18
τy_1d  = C.*cosd(ϕ) .+ P_1d.*sind(ϕ)
time   = 0.0
#-----------
dρdP     = 0.5641895835477563*dρinc.*exp.( .-((P_1d.-Pt)./dPr).^2 ) ./ dPr
max_dρdP = maximum(dρdP)
Δt_1d    = Δtr*(1.0 .- dρdP ./ max_dρdP ./1.1)
@printf("min Δt = %2.2e --- max Δt = %2.2e\n", minimum(Δt_1d), maximum(Δt_1d))
#-----------
Lc = Lx
tc = ηr*βr
εc = 1.0/tc
σc = Pini
ρc = σc*tc^2/Lc^2
μc = σc*tc
Vc = Lc/tc
#-----------
Lx,  Ly,  Lz = Lx/Lc,  Ly/Lc,  Lz/Lc
ε_BG, ∇V_BG  = ε_BG/εc, ∇V_BG/εc 
r        /= Lc
println(r)
βr       /= (1.0/σc)
Gr       /= σc
ρr       /= ρc
Pr       /= σc
Pini     /= σc
ηr       /= μc
Δtr      /= tc
dPr      /= σc
Pt       /= σc
dρinc    /= ρc
Ptens    /= σc
C        /= σc
η_vp     /= μc
max_dρdP /= (ρc/σc)
#-----------
xv  = LinRange(-Lx/2, Lx/2, ncx+1)
yv  = LinRange(-Ly/2, Ly/2, ncy+1)
zv  = LinRange(-Lz/2, Lz/2, ncz+1)
Δx, Δy, Δz = Lx/ncx, Ly/ncy, Lz/ncz
xce = LinRange(-Lx/2-Δx/2, Lx/2+Δx/2, ncx+2)
yce = LinRange(-Ly/2-Δy/2, Ly/2+Δy/2, ncy+2)
zce = LinRange(-Lz/2-Δz/2, Lz/2+Δz/2, ncz+2)
xc  = LinRange(-Lx/2+Δx/2, Lx/2-Δx/2, ncx+2)
yc  = LinRange(-Ly/2+Δy/2, Ly/2-Δy/2, ncy+2)
zc  = LinRange(-Lz/2+Δz/2, Lz/2-Δz/2, ncz+2)
# saving example -----------------------------------------------------------------------------------------------
if show_visu ==1 anim   = Animation() end
out_path = @sprintf("%s/out_visu", @__DIR__)
!ispath(out_path) && mkdir(out_path)
dim_g  = (ncx, ncy, ncz)
timev = Float64[]; h5_names = String[]; isave = 1
# saving example -----------------------------------------------------------------------------------------------
Ft    = @zeros(ncx+0, ncy+0, ncz+0)
Fs    = @zeros(ncx+0, ncy+0, ncz+0)
Y     = @zeros(ncx+0, ncy+0, ncz+0)
P0    = @zeros(ncx+2, ncy+2, ncz+2)
P1    = @zeros(ncx+2, ncy+2, ncz+2)
dρ    = @zeros(ncx+0, ncy+0, ncz+0)
ρ0    = @zeros(ncx+0, ncy+0, ncz+0)
ρref  = @zeros(ncx+0, ncy+0, ncz+0)
βc    = @zeros(ncx+0, ncy+0, ncz+0)
λc    = @zeros(ncx+0, ncy+0, ncz+0)
λxy   = @zeros(ncx+1, ncy+1, ncz+2)
λxz   = @zeros(ncx+1, ncy+2, ncz+1)
λyz   = @zeros(ncx+2, ncy+1, ncz+1)
τxx0  = @zeros(ncx+2, ncy+2, ncz+2)
τyy0  = @zeros(ncx+2, ncy+2, ncz+2)
τzz0  = @zeros(ncx+2, ncy+2, ncz+2)
τxy0  = @zeros(ncx+1, ncy+1, ncz+2)
τxz0  = @zeros(ncx+1, ncy+2, ncz+1)
τyz0  = @zeros(ncx+2, ncy+1, ncz+1)
τii   = @zeros(ncx+0, ncy+0, ncz+0)
εii   = @zeros(ncx+0, ncy+0, ncz+0)
ηc    = @zeros(ncx+2, ncy+2, ncz+2)
ηv    = @zeros(ncx+1, ncy+1, ncz+1)
Gc    = @zeros(ncx+0, ncy+0, ncz+0)
Gv    = @zeros(ncx+1, ncy+1, ncz+1)
βv    = @zeros(ncx+1, ncy+1, ncz+1)
∇V    = @zeros(ncx+2, ncy+2, ncz+2) 
εxx   = @zeros(ncx+2, ncy+2, ncz+2) 
εyy   = @zeros(ncx+2, ncy+2, ncz+2)
εzz   = @zeros(ncx+2, ncy+2, ncz+2)
εxy   = @zeros(ncx+1, ncy+1, ncz+2)
εxz   = @zeros(ncx+1, ncy+2, ncz+1)
εyz   = @zeros(ncx+2, ncy+1, ncz+1)
Fx    = @zeros(ncx+1, ncy+0, ncz+0)
Fy    = @zeros(ncx+0, ncy+1, ncz+0)
Fz    = @zeros(ncx+0, ncy+0, ncz+1)
Fp    = @zeros(ncx+0, ncy+0, ncz+0)
if restart_from == 0
    Vx    = @zeros(ncx+1, ncy+2, ncz+2)
    Vy    = @zeros(ncx+2, ncy+1, ncz+2)
    Vz    = @zeros(ncx+2, ncy+2, ncz+1)
    P     = @zeros(ncx+2, ncy+2, ncz+2)
    ρ     = @zeros(ncx+0, ncy+0, ncz+0)
    τxx   = @zeros(ncx+2, ncy+2, ncz+2)
    τyy   = @zeros(ncx+2, ncy+2, ncz+2)
    τzz   = @zeros(ncx+2, ncy+2, ncz+2)
    τxy   = @zeros(ncx+1, ncy+1, ncz+2)
    τxz   = @zeros(ncx+1, ncy+2, ncz+1)
    τyz   = @zeros(ncx+2, ncy+1, ncz+1)
    dVxdτ = @zeros(ncx+1, ncy+0, ncz+0)
    dVydτ = @zeros(ncx+0, ncy+1, ncz+0)
    dVzdτ = @zeros(ncx+0, ncy+0, ncz+1)
    #-----------
    @parallel InitialCondition( Vx, Vy, Vz, ηv, Gv, βv, ε_BG, ∇V_BG, xv, yv, zv, xce, yce, zce, r, ηr, dρ, dρinc, βc, βr, Gr, ρref, ρr )
    P .= Pini
    #-----------
    @parallel UpdateDensity( ρ, ρref, βc, P, Pr, dρ, Pt, dPr )
    @parallel InterpV2Ce( ηc, ηv )
    @parallel (1:size(ηc,2), 1:size(ηc,3)) bc_x!(ηc)
    @parallel (1:size(ηc,1), 1:size(ηc,3)) bc_y!(ηc)
    @parallel (1:size(ηc,1), 1:size(ηc,2)) bc_z!(ηc)
    @parallel InterpV2C( ηv, ηc, 0 )
 else# Breakpoint business
    fname = @sprintf("./Breakpoint%05d.h5", restart_from)
    @printf("Reading file %s\n", fname)
    h5open(fname, "r") do file
        dρ    = Data.Array(read(file, "drho"))
        ρref  = Data.Array(read(file, "rho_ref"))
        ηv    = Data.Array(read(file, "ev"))
        Gv    = Data.Array(read(file, "Gv"))
        βv    = Data.Array(read(file, "Bv"))
        P     = Data.Array(read(file, "P"))
        Vx    = Data.Array(read(file, "Vx"))
        Vy    = Data.Array(read(file, "Vy"))
        Vz    = Data.Array(read(file, "Vz"))
        ρ     = Data.Array(read(file, "rho"))
        τxx   = Data.Array(read(file, "Txx"))
        τyy   = Data.Array(read(file, "Tyy"))
        τzz   = Data.Array(read(file, "Tzz"))
        τxy   = Data.Array(read(file, "Txy"))
        τxz   = Data.Array(read(file, "Txz"))
        τyz   = Data.Array(read(file, "Tyz"))
        dVxdτ = Data.Array(read(file, "dVxdt"))
        dVydτ = Data.Array(read(file, "dVydt"))
        dVzdτ = Data.Array(read(file, "dVzdt"))
        timev = Data.Array(read(file, "timev"))
        time  = read(file, "time")
    end
end
#-----------
@parallel InterpV2C( βc, βv, 1 )
#-----------
niter  = 1e5
nout   = 500
Reopt  = 0.5*pi
cfl    = 0.62
ρnum   = cfl*Reopt/max(ncx,ncy,ncz)
λrel   = .5  
tol    = 1e-6
#-----------
for it=restart_from+1:nt
    #----------- Adaptive Δt
    # dρdP   = 0.5641895835477563*dρinc.*exp.( .-((P[(ncx+2)÷2,(ncy+2)÷2,(ncz+2)÷2].-Pt)./dPr).^2 ) ./ dPr 
    Δt     = Δtr#*(1.0 .-  dρdP ./ max_dρdP./1.1)
    time  += Δt
    #----------- Adaptive PT parameters
    η_ve   = 1.0/(1.0/maximum(ηv) + 1.0/(Gr*Δt)) 
    Δτ     = ρnum*min(Δx, Δy, Δz).^2 / η_ve /6.1  /1.1
    ΚΔτ    = min(η_ve, Δt/βr) * min(Δx, Δz) / sqrt(Lx^2+Ly^2+Lz^2) * cfl*1.0 /1.1
    @printf("###################################################################################\n")
    @printf("#### Time step %04d --- Δt = %2.2e  ####\n", it, Δt*tc)
    @printf("###################################################################################\n")
    P0   .= P
    ρ0   .= ρ
    τxx0 .= τxx
    τyy0 .= τyy
    τzz0 .= τzz
    τxy0 .= τxy
    τxz0 .= τxz
    τyz0 .= τyz
    λc   .= 0.0
    λxy  .= 0.0
    λxz  .= 0.0
    λyz  .= 0.0
    ##-----------
    for iter=1:niter
        @parallel (1:size(Vy,2), 1:size(Vy,3)) bc_x!(Vy)
        @parallel (1:size(Vz,2), 1:size(Vz,3)) bc_x!(Vz)
        @parallel (1:size(Vx,1), 1:size(Vx,3)) bc_y!(Vx)
        @parallel (1:size(Vz,1), 1:size(Vz,3)) bc_y!(Vz)
        @parallel (1:size(Vx,1), 1:size(Vx,2)) bc_z!(Vx)
        @parallel (1:size(Vy,1), 1:size(Vy,2)) bc_z!(Vy)
        @parallel UpdateDensity( ρ, ρref, βc, P, Pr, dρ, Pt, dPr )
        @parallel ComputeStrainRates( ∇V, εxx, εyy, εzz, εxy, εxz, εyz, Vx, Vy, Vz, Δx, Δy, Δz )
        @parallel StressEverywhere( P1, P, τxx, τyy, τzz, τxy, τxz, τyz, εxx, εyy, εzz, εxy, εxz, εyz, τxx0, τyy0, τzz0, τxy0, τxz0, τyz0, τii, ηv, βv, Gv, Δt, λc, λxy, λxz, λyz, C, cosd(ϕ), sind(ϕ), sind(ψ), η_vp, λrel )
        @parallel ComputeResiduals( Fx, Fy, Fz, Fp, τxx, τyy, τzz, τxy, τxz, τyz, P1, ∇V, ρ, ρ0, Δx, Δy, Δz, Δt )
        @parallel UpdateRates( dVxdτ, dVydτ, dVzdτ, ρnum, Fx, Fy, Fz, ncx, ncy, ncz )
        @parallel UpdateVP( dVxdτ, dVydτ, dVzdτ, Fp, Vx, Vy, Vz, P, ρnum, Δτ, ΚΔτ,  ncx, ncy, ncz )
        if mod(iter,nout) == 0 || iter==1
            nFx = norm(Fx)/sqrt(length(Fx))
            nFy = norm(Fy)/sqrt(length(Fy))
            nFz = norm(Fz)/sqrt(length(Fz))
            nFp = norm(Fp)/sqrt(length(Fp))
            @printf("Iter. %05d\n", iter) 
            @printf("Fx = %2.4e\n", nFx) 
            @printf("Fy = %2.4e\n", nFy) 
            @printf("Fz = %2.4e\n", nFz) 
            @printf("Fp = %2.4e\n", nFp) 
            max(nFx, nFy, nFz, nFp)<tol && break # short circuiting operations
            isnan(nFx) && error("NaN emergency!") 
            nFx>1e8    && error("Blow up!") 
        end
    end
    P .= P1
    @parallel UpdateDensity( ρ, ρref, βc, P, Pr, dρ, Pt, dPr )
    #-----------
    @printf("τxx : min = %2.4e --- max = %2.4e\n", minimum(τxx[2:end-1,2:end-1,2:end-1])*σc, maximum(τxx[2:end-1,2:end-1,2:end-1])*σc/1e9)
    @printf("τyy : min = %2.4e --- max = %2.4e\n", minimum(τyy[2:end-1,2:end-1,2:end-1])*σc, maximum(τyy[2:end-1,2:end-1,2:end-1])*σc/1e9)
    @printf("τzz : min = %2.4e --- max = %2.4e\n", minimum(τzz[2:end-1,2:end-1,2:end-1])*σc, maximum(τzz[2:end-1,2:end-1,2:end-1])*σc/1e9)
    @printf("P0  : min = %2.4e --- max = %2.4e\n", minimum( P0[2:end-1,2:end-1,2:end-1])*σc, maximum( P0[2:end-1,2:end-1,2:end-1])*σc/1e9)
    @printf("P   : min = %2.4e --- max = %2.4e\n", minimum(  P[2:end-1,2:end-1,2:end-1])*σc, maximum(  P[2:end-1,2:end-1,2:end-1])*σc/1e9)
    @printf("ρ0  : min = %2.4e --- max = %2.4e\n", minimum( ρ0)*ρc, maximum( ρ0)*ρc)
    @printf("ρ   : min = %2.4e --- max = %2.4e\n", minimum(  ρ)*ρc, maximum(  ρ)*ρc)
    #----------- 
    if show_visu ==1 
        Pin  = P[2:end-1,2:end-1,2:end-1]
        ∇Vin = ∇V[2:end-1,2:end-1,2:end-1]
        # Fs .= τii .- C.*cosd(ϕ) .- Pin.*sind(ϕ)
        # Ft .= Ptens  .- Pin
        # @printf("max Ft = %2.2e\n", maximum(Ft))
        # Y .= 0
        # Y[Fs.>0.0 .&& Ft.>0.0 .&& Fs.>Ft] .= 1
        # Y[Fs.>0.0 .&& Ft.>0.0 .&& Fs.<Ft] .= 2
        # Y[Fs.>0.0 .&& Ft.<0.0           ] .= 2
        # Y[Ft.>0.0 .&& Fs.<0.0           ] .= 1
        # p1  = heatmap(xce[2:end-1].*Lc*1e2, zce[2:end-1].*Lc*1e2, ∇Vin[:, (size(∇Vin,2))÷2, :]'.*εc, title="∇V [1/s]", aspect_ratio=1, xlims=(-Lx/2*Lc*1e2, Lx/2*Lc*1e2), c=:jet1 )
        p1  = heatmap(xce[2:end-1].*Lc*1e2, zce[2:end-1].*Lc*1e2, Pin[:, (size(Pin,2))÷2, :]'.*σc./1e9, title="P [GPa]", aspect_ratio=1, xlims=(-Lx/2*Lc*1e2, Lx/2*Lc*1e2), c=:jet1 )
        # p3  = heatmap(xce[2:end-1].*Lc*1e2, zce[2:end-1].*Lc*1e2, ρ[:, (size(ρ,2))÷2, :]'.*ρc, title="ρ [kg/m³]", aspect_ratio=1, xlims=(-Lx/2*Lc*1e2, Lx/2*Lc*1e2), c=:jet1 )
        # p3  = heatmap(xce[2:end-1].*Lc*1e2, zce[2:end-1].*Lc*1e2, τii[:, (size(τii,2))÷2, :]'.*σc./1e9, title="τii [GPa]", aspect_ratio=1, xlims=(-Lx/2*Lc*1e2, Lx/2*Lc*1e2), c=:jet1 )
        # p3  = heatmap(xce[2:end-1].*Lc*1e2, zce[2:end-1].*Lc*1e2, Y[:, (size(Y,2))÷2, :]', title="Yield mode", aspect_ratio=1, xlims=(-Lx/2, Lx/2), c=:jet1 )
        # p1  = heatmap(xv.*Lc*1e2, zv.*Lc*1e2, log10.(ηv[:, (size(ηv,2))÷2, :].*μc)', title="log10 ηv [Pa.s]", aspect_ratio=1, xlims=(-Lx/2*Lc*1e2, Lx/2*Lc*1e2), c=:jet1 )
        # p1  = heatmap(xv.*Lc*1e2, zv.*Lc*1e2, Gv[:, (size(Gv,2))÷2, :]'.*σc, title="Gv [Pa]", aspect_ratio=1, xlims=(-Lx/2*Lc*1e2, Lx/2*Lc*1e2), c=:jet1 )

        # p3  = heatmap(xce[:].*Lc*1e2, zce[:].*Lc*1e2, τxzc[:, (size(τxzc,2))÷2, :]'.*σc./1e9, title="τxz [GPa]", aspect_ratio=1, xlims=(-Lx/2, Lx/2), c=:jet1 )
        # p3  = heatmap(xce[2:end-1].*Lc*1e2, zce[2:end-1].*Lc*1e2, Y[:, (size(Y,2))÷2, :]', title="Yield mode", aspect_ratio=1, xlims=(-Lx/2, Lx/2), c=:jet1 )
        p3  = heatmap(xce[2:end-1].*Lc*1e2, zce[2:end-1].*Lc*1e2, λc[:, (size(λc,2))÷2, :]'.*(1.0/tc), title="λ [1/s]", aspect_ratio=1, xlims=(-Lx/2*Lc*1e2, Lx/2*Lc*1e2), c=:jet1 )
        # p3  = heatmap(xv.*Lc*1e2, zv.*Lc*1e2, λxz[:, (size(λxz,2))÷2, :]'.*(1.0/tc), title="λxz [1/s]", aspect_ratio=1, xlims=(-Lx/2*Lc*1e2, Lx/2*Lc*1e2), c=:jet1 )
        # p1  = heatmap(ηv[:, (size(ηv,2))÷2, :]'.*μc)
        p2  = plot(P_1d./1e9, ρ_1d,legend=false)
        p2  = scatter!(Pin[:].*σc./1e9, ρ[:].*ρc, xlabel="P [GPa]", ylabel="ρ [kg / m³]")
        p4  = plot(P_1d./1e9, τy_1d./1e9,legend=false)
        p4  = scatter!(Pin[:].*σc./1e9, τii[:].*σc./1e9, xlabel="P [GPa]", ylabel="τii [GPa]")
        p   = plot(p1,p2,p3,p4)
        frame(anim)
        display(p) 
    end
    # Breakpoint business
    if write_out==1 && (it==1 || mod(it, write_nout)==0)
        fname = @sprintf("%s/Breakpoint%05d.h5", @__DIR__, it)
        @printf("Writing file %s\n", fname)
        h5open(fname, "w") do file
            write(file, "drho", Array(dρ)) 
            write(file, "rho_ref", Array(ρref))
            write(file, "ev", Array(ηv)) 
            write(file, "Gv", Array(Gv)) 
            write(file, "Bv", Array(βv)) 
            write(file, "P", Array(P)) 
            write(file, "Vx", Array(Vx)) 
            write(file, "Vy", Array(Vy)) 
            write(file, "Vz", Array(Vz))
            write(file, "rho", Array(ρ))
            write(file, "Txx", Array(τxx))
            write(file, "Tyy", Array(τyy))
            write(file, "Tzz", Array(τzz))
            write(file, "Txy", Array(τxy))
            write(file, "Txz", Array(τxz))
            write(file, "Tyz", Array(τyz))
            write(file, "dVxdt", Array(dVxdτ))
            write(file, "dVydt", Array(dVydτ))
            write(file, "dVzdt", Array(dVzdτ))
            write(file, "timev", Array(timev))
            write(file, "time", time)
        end
        # Interpolate some data
        # VxC = 0.5*(Vx[2:end,:,:]+Vx[1:end-1,:,:])
        # VyC = 0.5*(Vy[:,2:end,:]+Vy[:,1:end-1,:])
        # VzC = 0.5*(Vz[:,:,2:end]+Vz[:,:,1:end-1])
        @parallel ComputeStressInvariant( εii, εxx, εyy, εzz, εxy, εxz, εyz ) 
        # XML save from Ludo and Ivan ==> 🚀
        out_name = @sprintf("PT3DOutput%05d", isave) #it
        out_h5 = joinpath(out_path, out_name)*".h5"
        I = CartesianIndices(( 1:ncx, 1:ncy, 1:ncz ))
        fields = Dict("ηc"=>Array(ηc[2:end-1,2:end-1,2:end-1].*μc), "ρ"=>Array(ρ.*ρc), "τii"=>Array(τii.*σc), "P"=>Array(P[2:end-1,2:end-1,2:end-1].*σc), "εii"=>Array(εii.*εc), "λc"=>Array(λc.*εc), "∇V"=>Array(∇V[2:end-1,2:end-1,2:end-1].*εc))
        push!(timev, time); push!(h5_names, out_name*".h5")
        write_h5(out_h5, fields, dim_g, I)
        write_xdmf( joinpath(out_path,out_name)*".xdmf3", h5_names,fields, (xc[2],yc[2],zc[2]), (Δx,Δy,Δz), dim_g, timev.*tc )
        isave += 1
        # @printf("Writing file %s\n", fname)
    end
end
if show_visu ==1  gif(anim, "QuartzCoesiteJulia.gif", fps = 6) end
#-----------
return nothing
end

@parallel_indices (i,j,k) function InitialCondition( Vx, Vy, Vz, ηv, Gv, βv, ε_BG, ∇V_BG, xv, yv, zv, xce, yce, zce, r, ηr, dρ, dρinc, βc, βr, Gr, ρref, ρr )
    ri, ro, ar, ari = 0.25*r, r, 1.6, 1.2
    # Vertices
    if i<=size(Vx,1) Vx[i,j,k] = (-ε_BG + 1.0/3.0*∇V_BG)*xv[i] end
    if j<=size(Vy,2) Vy[i,j,k] = (        1.0/3.0*∇V_BG)*yv[j] end
    if k<=size(Vz,3) Vz[i,j,k] = ( ε_BG + 1.0/3.0*∇V_BG)*zv[k] end
    if i<=size(ηv,1) && j<=size(ηv,2) && k<=size(ηv,3) 
        Gv[i,j,k] = Gr
        βv[i,j,k] = βr
        ηv[i,j,k] = ηr 
        if (xv[i]^2/(ar*ro)^2 + yv[j]^2/ro^2 + zv[k]^2/ro^2) < 1.0  ηv[i,j,k] = ηr*100     end 
        if (xv[i]^2/(ar*ro)^2 + yv[j]^2/ro^2 + zv[k]^2/ro^2) < 1.0  βv[i,j,k] = βr*1.4166  end 
        if (xv[i]^2/(ar*ro)^2 + yv[j]^2/ro^2 + zv[k]^2/ro^2) < 1.0  Gv[i,j,k] = Gr*1.5*(1.0 + 0.2*(0.5-rand()))     end 
        if ((xv[i]-0.12)^2/(ari*ri)^2 + (yv[j]-0.02)^2/ri^2 + (zv[k]-0.05)^2/ri^2) < 1.0  βv[i,j,k] = βr end  
        if ((xv[i]-0.12)^2/(ari*ri)^2 + (yv[j]-0.02)^2/ri^2 + (zv[k]-0.05)^2/ri^2) < 1.0  ηv[i,j,k] = ηr/10.0 end  
    end
    # Centroids
    if i<=size(dρ,1) && j<=size(dρ,2) && k<=size(dρ,3)
        ρref[i,j,k] = 0.8788*ρr
        if (xv[i]^2/(ar*ro)^2 + zv[k]^2/ro^2) < 1.0  ρref[i,j,k] = ρr    end
        if ((xce[i+1]-0.12)^2/(ari*ri)^2 + (yce[j+1]-0.02)^2/ri^2 + (zce[k+1]-0.05)^2/ri^2) < 1.0  
            ρref[i,j,k] = 0.8788*ρr
            dρ[i,j,k]   = dρinc  
        end  
    end
    return nothing
end

@parallel_indices (i,j,k) function UpdateDensity(ρ, ρref, βc, P, Pr, dρ, Pt, dPr)
    if i<=size(ρ, 1) && j<=size(ρ, 2) && k<=size(ρ, 3) ρref1    = ρref[i,j,k] - dρ[i,j,k] * 1.0/2.0 * erfc( (P[i+1,j+1,k+1]-Pt)/dPr )  end
    if i<=size(ρ, 1) && j<=size(ρ, 2) && k<=size(ρ, 3) ρ[i,j,k] = ρref1 * exp( βc[i,j,k]*(P[i+1,j+1,k+1] - Pr) ) end
    return nothing
end

@parallel_indices (j,k) function bc_x!(A::Data.Array)
    A[  1, j,  k] = A[    2,   j,   k]
    A[end, j,  k] = A[end-1,   j,   k]
    return
end

@parallel_indices (i,k) function bc_y!(A::Data.Array)
    A[ i,  1,  k] = A[   i,    2,   k]
    A[ i,end,  k] = A[   i,end-1,   k]
    return
end

@parallel_indices (i,j) function bc_z!(A::Data.Array)
    A[ i,  j,  1] = A[   i,   j,    2]
    A[ i,  j,end] = A[   i,   j,end-1]
    return
end

@parallel_indices (i,j,k) function InterpV2C( ηc, ηv, type )
    if type == 0 && i<=size(ηc,1) && j<=size(ηc,2) && k<=size(ηc,3)
        ηc[i,j,k]  = 1.0/8.0*( ηv[i,  j,  k] + ηv[i+1,j,k  ] + ηv[i,j+1,k  ] + ηv[i,  j,k+1  ] )
        ηc[i,j,k] += 1.0/8.0*( ηv[i+1,j+1,k] + ηv[i+1,j,k+1] + ηv[i,j+1,k+1] + ηv[i+1,j+1,k+1] )
    end
    if type == 1 && i<=size(ηc,1) && j<=size(ηc,2) && k<=size(ηc,3)
        a  = 1.0/8.0*( 1.0/ηv[i,  j,  k] + 1.0/ηv[i+1,j,k  ] + 1.0/ηv[i,j+1,k  ] + 1.0/ηv[i,  j,k+1  ] )
        a += 1.0/8.0*( 1.0/ηv[i+1,j+1,k] + 1.0/ηv[i+1,j,k+1] + 1.0/ηv[i,j+1,k+1] + 1.0/ηv[i+1,j+1,k+1] )
        ηc[i,j,k] = 1.0/a
    end
    return nothing
end

@parallel_indices (i,j,k) function InterpV2Ce( ηc, ηv )
    if i<=size(ηc,1)-2 && j<=size(ηc,2)-2 && k<=size(ηc,3)-2
        ηc[i+1,j+1,k+1]  = 1.0/8.0*( ηv[i,  j,  k] + ηv[i+1,j,k  ] + ηv[i,j+1,k  ] + ηv[i,  j,k+1  ] )
        ηc[i+1,j+1,k+1] += 1.0/8.0*( ηv[i+1,j+1,k] + ηv[i+1,j,k+1] + ηv[i,j+1,k+1] + ηv[i+1,j+1,k+1] )
    end
    return nothing
end

@parallel_indices (i,j,k) function InterpV2xyz( ηxy, ηxz, ηyz, ηv )
    if k<=size(ηxy,3) ηxy[i,j,k]  = 1.0/2.0*( ηv[i,j,k] + ηv[i,j,k+1]) end
    if j<=size(ηxz,2) ηxz[i,j,k]  = 1.0/2.0*( ηv[i,j,k] + ηv[i,j+1,k]) end
    if i<=size(ηyz,1) ηyz[i,j,k]  = 1.0/2.0*( ηv[i,j,k] + ηv[i+1,j,k]) end
    return nothing
end

@parallel_indices (i,j,k) function ComputeStrainRates( ∇V, εxx, εyy, εzz, εxy, εxz, εyz, Vx, Vy, Vz, Δx, Δy, Δz )
    if i<=size(εxx,1)-2 && j<=size(εxx,2)-2 && k<=size(εxx,3)-2
        dVxΔx      = (Vx[i+1,j+1,k+1] - Vx[i,j+1,k+1]) / Δx
        dVyΔy      = (Vy[i+1,j+1,k+1] - Vy[i+1,j,k+1]) / Δy
        dVzΔz      = (Vz[i+1,j+1,k+1] - Vz[i+1,j+1,k]) / Δz
        ∇V[i+1,j+1,k+1]  = dVxΔx + dVyΔy + dVzΔz
        εxx[i+1,j+1,k+1] = dVxΔx - 1.0/3.0 * ∇V[i+1,j+1,k+1]
        εyy[i+1,j+1,k+1] = dVyΔy - 1.0/3.0 * ∇V[i+1,j+1,k+1]
        εzz[i+1,j+1,k+1] = dVzΔz - 1.0/3.0 * ∇V[i+1,j+1,k+1]
    end
    if i<=size(εxy,1) && j<=size(εxy,2) && k<=size(εxy,3)-2
        dVxΔy      = (Vx[i,j+1,k+1] - Vx[i,j,k+1]) / Δy 
        dVyΔx      = (Vy[i+1,j,k+1] - Vy[i,j,k+1]) / Δx 
        εxy[i,j,k+1] =  1.0/2.0*(dVxΔy + dVyΔx)
    end
    if i<=size(εxz,1) && j<=size(εxz,2)-2 && k<=size(εxz,3)
        dVxΔz      = (Vx[i  ,j+1,k+1] - Vx[i,j+1,k]) / Δz                     
        dVzΔx      = (Vz[i+1,j+1,k  ] - Vz[i,j+1,k]) / Δx 
        εxz[i,j+1,k] = 1.0/2.0*(dVxΔz + dVzΔx)
    end
    if i<=size(εyz,1)-2 && j<=size(εyz,2) && k<=size(εyz,3)
        dVyΔz      = (Vy[i+1,j,k+1] - Vy[i+1,j,k]) / Δz 
        dVzΔy      = (Vz[i+1,j+1,k] - Vz[i+1,j,k]) / Δy 
        εyz[i+1,j,k] = 1.0/2.0*(dVyΔz + dVzΔy)
    end
    return nothing
end

@parallel_indices (i,j,k) function ComputeResiduals( Fx, Fy, Fz, Fp, τxx, τyy, τzz, τxy, τxz, τyz, P, ∇V, ρ, ρ0, Δx, Δy, Δz, Δt )
    if i<=size(Fx,1) && j<=size(Fx,2) && k<=size(Fx,3)
        if i>1 && i<size(Fx,1) # avoid Dirichlets
            Fx[i,j,k]  = (τxx[i+1,j+1,k+1] - τxx[i,j+1,k+1]) / Δx
            Fx[i,j,k] -= (  P[i+1,j+1,k+1] -   P[i,j+1,k+1]) / Δx
            Fx[i,j,k] += (τxy[i,j+1,k+1] - τxy[i,j,k+1]) / Δy
            Fx[i,j,k] += (τxz[i,j+1,k+1] - τxz[i,j+1,k]) / Δz
        end
    end
    if i<=size(Fy,1) && j<=size(Fy,2) && k<=size(Fy,3)
        if j>1 && j<size(Fy,2) # avoid Dirichlets
            Fy[i,j,k]  = (τyy[i+1,j+1,k+1] - τyy[i+1,j,k+1]) / Δy
            Fy[i,j,k] -= (  P[i+1,j+1,k+1] -   P[i+1,j,k+1]) / Δy
            Fy[i,j,k] += (τxy[i+1,j,k+1] - τxy[i,j,k+1]) / Δx
            Fy[i,j,k] += (τyz[i+1,j,k+1] - τyz[i+1,j,k]) / Δz
        end
    end
    if i<=size(Fz,1) && j<=size(Fz,2) && k<=size(Fz,3)
        if k>1 && k<size(Fz,3) # avoid Dirichlets
            Fz[i,j,k]  = (τzz[i+1,j+1,k+1] - τzz[i+1,j+1,k]) / Δz
            Fz[i,j,k] -= (  P[i+1,j+1,k+1] -   P[i+1,j+1,k]) / Δz
            Fz[i,j,k] += (τxz[i+1,j+1,k] - τxz[i,j+1,k]) / Δx
            Fz[i,j,k] += (τyz[i+1,j+1,k] - τyz[i+1,j,k]) / Δy
        end
    end
    if i<=size(Fp,1) && j<=size(Fp,2) && k<=size(Fp,3)
        Fp[i,j,k] = -∇V[i+1,j+1,k+1] - (log( ρ[i,j,k] ) - log( ρ0[i,j,k])) / Δt
    end
    return nothing
end

@parallel_indices (i,j,k) function UpdateRates( dVxdτ, dVydτ, dVzdτ, ρnum, Fx, Fy, Fz, ncx, ncy, ncz )
    if i<=size(Fx,1) && j<=size(Fx,2) && k<=size(Fx,3)
        dVxdτ[i,j,k] = (1.0-ρnum)*dVxdτ[i,j,k] + Fx[i,j,k]
    end
    if i<=size(Fy,1) && j<=size(Fy,2) && k<=size(Fy,3)
        dVydτ[i,j,k] = (1.0-ρnum)*dVydτ[i,j,k] + Fy[i,j,k]
    end
    if i<=size(Fz,1) && j<=size(Fz,2) && k<=size(Fz,3)
        dVzdτ[i,j,k] = (1.0-ρnum)*dVzdτ[i,j,k] + Fz[i,j,k]
    end
    return nothing
end

@parallel_indices (i,j,k) function UpdateVP( dVxdτ, dVydτ, dVzdτ, dPdτ, Vx, Vy, Vz, P, ρnum, Δτ, ΚΔτ,  ncx, ncy, ncz )
    if i<=size(dVxdτ,1) && j<=size(dVxdτ,2) && k<=size(dVxdτ,3)
        Vx[i,j+1,k+1] += Δτ/ρnum*dVxdτ[i,j,k]
    end
    if i<=size(dVydτ,1) && j<=size(dVydτ,2) && k<=size(dVydτ,3)
        Vy[i+1,j,k+1] += Δτ/ρnum*dVydτ[i,j,k]
    end
    if i<=size(dVzdτ,1) && j<=size(dVzdτ,2) && k<=size(dVzdτ,3)
        Vz[i+1,j+1,k] += Δτ/ρnum*dVzdτ[i,j,k]
    end
    if i<=size(dPdτ,1) && j<=size(dPdτ,2) && k<=size(dPdτ,3)
        P[i+1,j+1,k+1] += ΚΔτ*dPdτ[i,j,k] 
    end
    return nothing
end

@parallel_indices (i,j,k) function StressEverywhere( P1, P, τxx, τyy, τzz, τxy, τxz, τyz, εxx, εyy, εzz, εxy, εxz, εyz, τxx0, τyy0, τzz0, τxy0, τxz0, τyz0, τii, ηv, βv, Gv, Δt, λc, λxy, λxz, λyz, C, cosϕ, sinϕ, sinψ, η_vp, λrel )
    # CENTROIDS    
    if i<=size(τxx,1)-2 && j<=size(τxx,2)-2 && k<=size(τxx,3)-2
        # Centroid viscosity
        η  = 1.0/8.0*( ηv[i,  j,  k] + ηv[i+1,j,k  ] + ηv[i,j+1,k  ] + ηv[i,  j,k+1  ] )
        η += 1.0/8.0*( ηv[i+1,j+1,k] + ηv[i+1,j,k+1] + ηv[i,j+1,k+1] + ηv[i+1,j+1,k+1] )
        # Centroid shear modulus
        G  = 1.0/8.0*( Gv[i,  j,  k] + Gv[i+1,j,k  ] + Gv[i,j+1,k  ] + Gv[i,  j,k+1  ] )
        G += 1.0/8.0*( Gv[i+1,j+1,k] + Gv[i+1,j,k+1] + Gv[i,j+1,k+1] + Gv[i+1,j+1,k+1] )
        # Centroid compressibility
        β  = 1.0/8.0*( βv[i,  j,  k] + βv[i+1,j,k  ] + βv[i,j+1,k  ] + βv[i,  j,k+1  ] )
        β += 1.0/8.0*( βv[i+1,j+1,k] + βv[i+1,j,k+1] + βv[i,j+1,k+1] + βv[i+1,j+1,k+1] )
        # Visco-elastic rheology
        η_e  = G*Δt
        η_ve = 1.0 / ( 1.0/η + 1.0/η_e )
        # Trial deviatoric normal stress
        τxx[i+1,j+1,k+1] = 2η_ve*( εxx[i+1,j+1,k+1] + τxx0[i+1,j+1,k+1]/(2η_e) )
        τyy[i+1,j+1,k+1] = 2η_ve*( εyy[i+1,j+1,k+1] + τyy0[i+1,j+1,k+1]/(2η_e) )
        τzz[i+1,j+1,k+1] = 2η_ve*( εzz[i+1,j+1,k+1] + τzz0[i+1,j+1,k+1]/(2η_e) )
        # Trial deviatoric shear stress
        εxyc  = 0.25*( εxy[i,j,k+1] +  εxy[i+1,j,k+1] +  εxy[i,j+1,k+1] +  εxy[i+1,j+1,k+1])
        εxzc  = 0.25*( εxz[i,j+1,k] +  εxz[i+1,j+1,k] +  εxz[i,j+1,k+1] +  εxz[i+1,j+1,k+1])
        εyzc  = 0.25*( εyz[i+1,j,k] +  εyz[i+1,j+1,k] +  εyz[i+1,j,k+1] +  εyz[i+1,j+1,k+1])
        τxyc0 = 0.25*(τxy0[i,j,k+1] + τxy0[i+1,j,k+1] + τxy0[i,j+1,k+1] + τxy0[i+1,j+1,k+1])
        τxzc0 = 0.25*(τxz0[i,j+1,k] + τxz0[i+1,j+1,k] + τxz0[i,j+1,k+1] + τxz0[i+1,j+1,k+1])
        τyzc0 = 0.25*(τyz0[i+1,j,k] + τyz0[i+1,j+1,k] + τyz0[i+1,j,k+1] + τyz0[i+1,j+1,k+1])
        τxyc  = 2η_ve*(εxyc + τxyc0/(2η_e) )
        τxzc  = 2η_ve*(εxzc + τxzc0/(2η_e) )
        τyzc  = 2η_ve*(εyzc + τyzc0/(2η_e) )
        # Plasticity
        F, λ1, τii1, p1, τxx1, τyy1, τzz1, τxy1, τxz1, τyz1 = PlasticCorrection( λc[i,j,k], εxx[i+1,j+1,k+1], εyy[i+1,j+1,k+1], εzz[i+1,j+1,k+1], εxyc, εxzc, εyzc, τxx[i+1,j+1,k+1], τyy[i+1,j+1,k+1], τzz[i+1,j+1,k+1], τxyc, τxzc, τyzc, τxx0[i+1,j+1,k+1], τyy0[i+1,j+1,k+1], τzz0[i+1,j+1,k+1], τxyc0, τxzc0, τyzc0, P[i+1,j+1,k+1], C, cosϕ, sinϕ, sinψ, η_vp, η_ve, η_e, β, Δt, λrel)
        λc[i,j,k]        = λ1
        τii[i,j,k]       = τii1
        P1[i+1,j+1,k+1]  = p1
        τxx[i+1,j+1,k+1] = τxx1
        τyy[i+1,j+1,k+1] = τyy1
        τzz[i+1,j+1,k+1] = τzz1
    end
    # XY
    if i<=size(τxy,1)-0 && j<=size(τxy,2)-0 && k>1 && k<=size(τxy,3)-1
        # Centroid viscosity
        η     = 1.0/2.0*( ηv[i,  j,  k-1] + ηv[i,j,k] )
        # Centroid shear modulus
        G     = 1.0/2.0*( Gv[i,  j,  k-1] + Gv[i,j,k] )
        # Centroid compressibility
        β     = 1.0/2.0*( βv[i,  j,  k-1] + βv[i,j,k] )
        # Trial pressure
        Pv    = 1.0/4.0*( P[i,  j,  k] + P[i+1,j,k] + P[i,j+1,k] + P[i+1,j+1,k ] )
        # Visco-elastic rheology
        η_e   = G*Δt
        η_ve  = 1.0 / ( 1.0/η + 1.0/η_e )
        # Trial deviatoric normal stress
        εxxv  = 1.0/4.0*(  εxx[i,  j,  k] +  εxx[i+1,j,k] +  εxx[i,j+1,k] +  εxx[i+1,j+1,k ] )
        τxxv0 = 1.0/4.0*( τxx0[i,  j,  k] + τxx0[i+1,j,k] + τxx0[i,j+1,k] + τxx0[i+1,j+1,k ] )
        εyyv  = 1.0/4.0*(  εyy[i,  j,  k] +  εyy[i+1,j,k] +  εyy[i,j+1,k] +  εyy[i+1,j+1,k ] )
        τyyv0 = 1.0/4.0*( τyy0[i,  j,  k] + τyy0[i+1,j,k] + τyy0[i,j+1,k] + τyy0[i+1,j+1,k ] )
        εzzv  = 1.0/4.0*(  εzz[i,  j,  k] +  εzz[i+1,j,k] +  εzz[i,j+1,k] +  εzz[i+1,j+1,k ] )
        τzzv0 = 1.0/4.0*( τzz0[i,  j,  k] + τzz0[i+1,j,k] + τzz0[i,j+1,k] + τzz0[i+1,j+1,k ] )
        τxxv  = 2η_ve*( εxxv + τxxv0/(2η_e) )
        τyyv  = 2η_ve*( εyyv + τyyv0/(2η_e) )
        τzzv  = 2η_ve*( εzzv + τzzv0/(2η_e) )
        # Trial deviatoric shear stress
        εxzv  = 1.0/4.0*(  εxz[i,  j,  k-1] +  εxz[i,j+1,k-1] +  εxz[i,j,k] +  εxz[i,j+1,k ] )
        τxzv0 = 1.0/4.0*( τxz0[i,  j,  k-1] + τxz0[i,j+1,k-1] + τxz0[i,j,k] + τxz0[i,j+1,k ] )
        εyzv  = 1.0/4.0*(  εyz[i,  j,  k-1] +  εyz[i+1,j,k-1] +  εyz[i,j,k] +  εyz[i+1,j,k ] )
        τyzv0 = 1.0/4.0*( τyz0[i,  j,  k-1] + τyz0[i+1,j,k-1] + τyz0[i,j,k] + τyz0[i+1,j,k ] )
        τxy[i,j,k] = 2η_ve*( εxy[i,j,k] + τxy0[i,j,k]/(2η_e) )
        τxzv       = 2η_ve*( εxzv       + τxzv0/(2η_e) )
        τyzv       = 2η_ve*( εyzv       + τyzv0/(2η_e) )
        # Plasticity
        F, λ1, τii1, p1, τxx1, τyy1, τzz1, τxy1, τxz1, τyz1 = PlasticCorrection( λxy[i,j,k], εxxv, εyyv, εzzv, εxy[i,j,k], εxzv, εyzv, τxxv, τyyv, τzzv, τxy[i,j,k], τxzv, τyzv, τxxv0, τyyv0, τzzv0, τxy0[i,j,k], τxzv0, τyzv0, Pv, C, cosϕ, sinϕ, sinψ, η_vp, η_ve, η_e, β, Δt, λrel)
        λxy[i,j,k] = λ1
        τxy[i,j,k] = τxy1
    end
    # XZ
    if i<=size(τxz,1)-0 && j>1 && j<=size(τxz,2)-1 && k<=size(τxz,3)-0
        # Centroid viscosity
        η     = 1.0/2.0*( ηv[i,  j-1,  k] + ηv[i,j,k] )
        # Centroid shear modulus
        G     = 1.0/2.0*( Gv[i,  j-1,  k] + Gv[i,j,k] )
        # Centroid compressibility
        β     = 1.0/2.0*( βv[i,  j-1,  k] + βv[i,j,k] )
        # Trial pressure
        Pv    = 1.0/4.0*( P[i,  j,  k] + P[i+1,j,k] + P[i,j,k+1] + P[i+1,j,k+1] )
        # Visco-elastic rheology
        η_e   = G*Δt
        η_ve  = 1.0 / ( 1.0/η + 1.0/η_e )
        # Trial deviatoric normal stress
        εxxv  = 1.0/4.0*(   εxx[i,  j,  k] +  εxx[i+1,j,k] +  εxx[i,j,k+1] +  εxx[i+1,j,k+1] )
        τxxv0 = 1.0/4.0*(  τxx0[i,  j,  k] + τxx0[i+1,j,k] + τxx0[i,j,k+1] + τxx0[i+1,j,k+1] )
        εyyv  = 1.0/4.0*(   εyy[i,  j,  k] +  εyy[i+1,j,k] +  εyy[i,j,k+1] +  εyy[i+1,j,k+1] )
        τyyv0 = 1.0/4.0*(  τyy0[i,  j,  k] + τyy0[i+1,j,k] + τyy0[i,j,k+1] + τyy0[i+1,j,k+1] )
        εzzv  = 1.0/4.0*(   εzz[i,  j,  k] +  εzz[i+1,j,k] +  εzz[i,j,k+1] +  εzz[i+1,j,k+1] )
        τzzv0 = 1.0/4.0*(  τzz0[i,  j,  k] + τzz0[i+1,j,k] + τzz0[i,j,k+1] + τzz0[i+1,j,k+1] )
        τxxv  = 2η_ve*( εxxv + τxxv0/(2η_e) )
        τyyv  = 2η_ve*( εyyv + τyyv0/(2η_e) )
        τzzv  = 2η_ve*( εzzv + τzzv0/(2η_e) )
        # Trial deviatoric shear stress
        εxyv  = 1.0/4.0*(  εxy[i,  j-1,  k] +  εxy[i,j-1,k+1] +  εxy[i,j,k] +  εxy[i  ,j,k+1] )
        τxyv0 = 1.0/4.0*( τxy0[i,  j-1,  k] + τxy0[i,j-1,k+1] + τxy0[i,j,k] + τxy0[i  ,j,k+1] )
        εyzv  = 1.0/4.0*(  εyz[i,  j-1,  k] +  εyz[i+1,j-1,k] +  εyz[i,j,k] +  εyz[i+1,j,k  ] )
        τyzv0 = 1.0/4.0*( τyz0[i,  j-1,  k] + τyz0[i+1,j-1,k] + τyz0[i,j,k] + τyz0[i+1,j,k  ] )
        τxyv       = 2η_ve*( εxyv       + τxyv0/(2η_e) )
        τxz[i,j,k] = 2η_ve*( εxz[i,j,k] + τxz0[i,j,k]/(2η_e) )
        τyzv       = 2η_ve*( εyzv       + τyzv0/(2η_e) )
        # Plasticity
        F, λ1, τii1, p1, τxx1, τyy1, τzz1, τxy1, τxz1, τyz1 = PlasticCorrection( λxz[i,j,k], εxxv, εyyv, εzzv, εxyv, εxz[i,j,k], εyzv, τxxv, τyyv, τzzv, τxyv, τxz[i,j,k], τyzv, τxxv0, τyyv0, τzzv0, τxyv0, τxz0[i,j,k], τyzv0, Pv, C, cosϕ, sinϕ, sinψ, η_vp, η_ve, η_e, β, Δt, λrel)
        λxz[i,j,k] = λ1
        τxz[i,j,k] = τxz1
    end
    # YZ
    if i>1 && i<=size(τyz,1)-1 && j<=size(τyz,2)-0 && k<=size(τyz,3)-0
        # Centroid viscosity
        η    = 1.0/2.0*( ηv[i-1,  j,  k] + ηv[i,j,k] )  
        # Centroid shear modulus
        G    = 1.0/2.0*( Gv[i-1,  j,  k] + Gv[i,j,k] )
        # Centroid compressibility
        β    = 1.0/2.0*( βv[i-1,  j,  k] + βv[i,j,k] )
        # # Trial pressure
        Pv   = 1.0/4.0*( P[i,  j,  k] + P[i,j+1,k] + P[i,j,k+1] + P[i,j+1,k+1] )
        # Visco-elastic rheology
        η_e   = G*Δt
        η_ve  = 1.0 / ( 1.0/η + 1.0/η_e )
        # Trial deviatoric normal stress
        εxxv  = 1.0/4.0*(   εxx[i,  j,  k] +  εxx[i,j+1,k] +  εxx[i,j,k+1] +  εxx[i,j+1,k+1] )
        τxxv0 = 1.0/4.0*(  τxx0[i,  j,  k] + τxx0[i,j+1,k] + τxx0[i,j,k+1] + τxx0[i,j+1,k+1] )
        εyyv  = 1.0/4.0*(   εyy[i,  j,  k] +  εyy[i,j+1,k] +  εyy[i,j,k+1] +  εyy[i,j+1,k+1] )
        τyyv0 = 1.0/4.0*(  τyy0[i,  j,  k] + τyy0[i,j+1,k] + τyy0[i,j,k+1] + τyy0[i,j+1,k+1] )
        εzzv  = 1.0/4.0*(   εzz[i,  j,  k] +  εzz[i,j+1,k] +  εzz[i,j,k+1] +  εzz[i,j+1,k+1] )
        τzzv0 = 1.0/4.0*(  τzz0[i,  j,  k] + τzz0[i,j+1,k] + τzz0[i,j,k+1] + τzz0[i,j+1,k+1] )
        τxxv  = 2η_ve*( εxxv + τxxv0/(2η_e) )
        τyyv  = 2η_ve*( εyyv + τyyv0/(2η_e) )
        τzzv  = 2η_ve*( εzzv + τzzv0/(2η_e) )
        # Trial deviatoric shear stress
        εxyv  = 1.0/4.0*(  εxy[i-1,  j,  k] +  εxy[i-1,j,k+1] +  εxy[i,j,k] +  εxy[i  ,j,k+1] )
        τxyv0 = 1.0/4.0*( τxy0[i-1,  j,  k] + τxy0[i-1,j,k+1] + τxy0[i,j,k] + τxy0[i  ,j,k+1] )
        εxzv  = 1.0/4.0*(  εxz[i-1,  j,  k] +  εxz[i-1,j+1,k] +  εxz[i,j,k] +  εxz[i,j+1,k  ] )
        τxzv0 = 1.0/4.0*( τxz0[i-1,  j,  k] + τxz0[i-1,j+1,k] + τxz0[i,j,k] + τxz0[i,j+1,k  ] )
        τxyv       = 2η_ve*( εxyv       + τxyv0/(2η_e) )
        τxzv       = 2η_ve*( εxzv       + τxzv0/(2η_e) )
        τyz[i,j,k] = 2η_ve*( εyz[i,j,k] + τyz0[i,j,k]/(2η_e) )
        # Plasticity
        F, λ1, τii1, p1, τxx1, τyy1, τzz1, τxy1, τxz1, τyz1 = PlasticCorrection( λyz[i,j,k], εxxv, εyyv, εzzv, εxyv, εxzv, εyz[i,j,k], τxxv, τyyv, τzzv, τxyv, τxzv, τyz[i,j,k], τxxv0, τyyv0, τzzv0, τxyv0, τxzv0, τyz0[i,j,k], Pv, C, cosϕ, sinϕ, sinψ, η_vp, η_ve, η_e, β, Δt, λrel)
        λyz[i,j,k] = λ1
        τyz[i,j,k] = τyz1
    end
    return nothing
end

@views function PlasticCorrection( λ, εxx, εyy, εzz, εxy, εxz, εyz, τxx, τyy, τzz, τxy, τxz, τyz, τxx0, τyy0, τzz0, τxy0, τxz0, τyz0, P, C, cosϕ, sinϕ, sinψ, η_vp, η_ve, η_e, β, Δt, λrel)
    # Plasticity
    τii = sqrt(0.5*(τxx^2 + τyy^2 + τzz^2) + τxy^2 + τxz^2 + τyz^2)
    F   = τii - C*cosϕ - P*sinϕ
    if F>0
        λ1    = F / ( η_ve + η_vp + Δt/β*sinϕ*sinψ )
        λ     = (1.0-λrel)*λ + λrel*λ1
        P1    = P   + λ/β*Δt*sinψ
        τii1  = τii - η_ve*λ
        Eii1  = 0.5*(εxx + τxx0/(2η_e) )^2
        Eii1 += 0.5*(εyy + τyy0/(2η_e) )^2
        Eii1 += 0.5*(εzz + τzz0/(2η_e) )^2
        Eii1 +=     (εxy + τxy0/(2η_e) )^2
        Eii1 +=     (εxz + τxz0/(2η_e) )^2
        Eii1 +=     (εyz + τyz0/(2η_e) )^2
        Eii1  = sqrt(Eii1)
        η_vep = τii1/2.0/Eii1
        τxx   = 2η_vep*( εxx + τxx0/(2η_e) )
        τyy   = 2η_vep*( εyy + τyy0/(2η_e) )
        τzz   = 2η_vep*( εzz + τzz0/(2η_e) )
        τxy   = 2η_vep*( εxy + τxy0/(2η_e) )
        τxz   = 2η_vep*( εxz + τxz0/(2η_e) )
        τyz   = 2η_vep*( εyz + τyz0/(2η_e) )
        τii   = sqrt(0.5*(τxx^2 + τyy^2 + τzz^2) + τxy^2 + τxz^2 + τyz^2)
        F     = τii1 - C*cosϕ - P1*sinϕ - λ1*η_vp
    else
        λ     = 0.0
        τii1  = τii
        P1    = P
    end
    return F, λ, τii1, P1, τxx, τyy, τzz, τxy, τxz, τyz  
end

@parallel_indices (i,j,k) function ComputeStressInvariant( τII, τxx, τyy, τzz, τxy, τxz, τyz )
    if i<=size(τII,1) && j<=size(τII,2) && k<=size(τII,3)
        Jii      = 0.5*τxx[i+1,j+1,k+1]^2
        Jii     += 0.5*τyy[i+1,j+1,k+1]^2
        Jii     += 0.5*τzz[i+1,j+1,k+1]^2
        Jii     += (0.25*(τxy[i,j,k] + τxy[i+1,j,k] + τxy[i,j+1,k] + τxy[i+1,j+1,k]) )^2
        Jii     += (0.25*(τxz[i,j,k] + τxz[i+1,j,k] + τxz[i,j,k+1] + τxz[i+1,j,k+1]) )^2
        Jii     += (0.25*(τyz[i,j,k] + τyz[i,j+1,k] + τyz[i,j,k+1] + τyz[i,j+1,k+1]) )^2
        τII[i,j,k] = sqrt(Jii)
    end
    return nothing
end

@time main(4)
