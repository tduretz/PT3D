# V-E-P model
# rheology on centers ONLY
# viscosity with extended stencil (useful?)
using  Printf, Plots
import Statistics: mean
import LinearAlgebra: norm
import SpecialFunctions: erfc

const USE_GPU = false
const GPU_ID  = 0
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3);
else
    @init_parallel_stencil(Threads, Float64, 3);
end

function main( n )
nt            = 100
Δtr           = 5e11
ε_BG          = 1.0e-16
∇V_BG         = 1.0e-14
r             = 1e-3
βr            = 1.0e-10
Gr            = 3e30
ρr            = 3000
ηr            = 1e25
Lx,  Ly,  Lz  =  1.0e-2,  (3.0/32)*1e-2,  1.0e-2 
ncx, ncy, ncz = n*32, 2, n*32
BCtype        = :PureShear_xz
# HP
Pini   = 4e9
Pr     = Pini
dPr    = 5e7
Pt     = 3.5e9
dρinc  = 300.0
Ptens  = -2e7
#-----------
Pmin   = 2e9
Pmax   = Pini 
P_1d   = LinRange( Pmin, Pmax, 200 )
ρr_1d  = ρr .- dρinc*1//2 .* erfc.( (P_1d.-Pt)./dPr ) 
ρ_1d   = ρr_1d .*exp.(βr.*(P_1d.-Pr))
ϕ      = 20.0
C      = 1e7
τy_1d  = C.*cosd(ϕ) .+ P_1d.*sind(ϕ)
#-----------
dρdP     = 0.5641895835477563*dρinc.*exp.( .-((P_1d.-Pt)./dPr).^2 ) ./ dPr
max_dρdP = maximum(dρdP)
Δt_1d    = Δtr*(1.0 .- dρdP ./ max_dρdP ./1.1)
@printf("min Δt = %2.2e --- max Δt = %2.2e\n", minimum(Δt_1d), maximum(Δt_1d))
# # LP
# Pini   = 3e8
# Pr     = Pini
# dPr    = 2e7
# Pt     = 1.5e8
# dρinc  = 300.0
# Ptens  = -1e7
# #-----------
# Pmin   = -5e7
# Pmax   = Pini 
# P_1d   = LinRange( Pmin, Pmax, 200 )
# ρr_1d  = ρr .- dρinc*1//2 .* erfc.( (P_1d.-Pt)./dPr ) 
# ρ_1d   = ρr_1d .*exp.(βr.*(P_1d.-Pr))
# ϕ      = 20.0
# C      = 1e7
# τy_1d  = C.*cosd(ϕ) .+ P_1d.*sind(ϕ)
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
max_dρdP /= (ρc/σc)
#-----------
Ft    = @zeros(ncx+0, ncy+0, ncz+0)
Fs    = @zeros(ncx+0, ncy+0, ncz+0)
Y     = @zeros(ncx+0, ncy+0, ncz+0)
P0    = @zeros(ncx+2, ncy+2, ncz+2)
P     = @zeros(ncx+2, ncy+2, ncz+2)
dρ    = @zeros(ncx+0, ncy+0, ncz+0)
Vx    = @zeros(ncx+1, ncy+2, ncz+2)
Vy    = @zeros(ncx+2, ncy+1, ncz+2)
Vz    = @zeros(ncx+2, ncy+2, ncz+1)
ρ0    = @zeros(ncx+0, ncy+0, ncz+0)
ρ     = @zeros(ncx+0, ncy+0, ncz+0)
β     = @zeros(ncx+0, ncy+0, ncz+0)
τxx   = @zeros(ncx+2, ncy+2, ncz+2)
τyy   = @zeros(ncx+2, ncy+2, ncz+2)
τzz   = @zeros(ncx+2, ncy+2, ncz+2)
τxy   = @zeros(ncx+1, ncy+1, ncz+0)
τxz   = @zeros(ncx+1, ncy+0, ncz+1)
τyz   = @zeros(ncx+0, ncy+1, ncz+1)
τxyc  = @zeros(ncx+2, ncy+2, ncz+2)
τxzc  = @zeros(ncx+2, ncy+2, ncz+2)
τyzc  = @zeros(ncx+2, ncy+2, ncz+2)
τxx0  = @zeros(ncx+2, ncy+2, ncz+2)
τyy0  = @zeros(ncx+2, ncy+2, ncz+2)
τzz0  = @zeros(ncx+2, ncy+2, ncz+2)
τxy0  = @zeros(ncx+1, ncy+1, ncz+0)
τxz0  = @zeros(ncx+1, ncy+0, ncz+1)
τyz0  = @zeros(ncx+0, ncy+1, ncz+1)
τii   = @zeros(ncx+0, ncy+0, ncz+0)
ηc    = @zeros(ncx+2, ncy+2, ncz+2)
ηv    = @zeros(ncx+1, ncy+1, ncz+1)
ηxy   = @zeros(ncx+1, ncy+1, ncz+0)
ηxz   = @zeros(ncx+1, ncy+0, ncz+1)
ηyz   = @zeros(ncx+0, ncy+1, ncz+1)
Gc    = @zeros(ncx+0, ncy+0, ncz+0)
Gv    = @zeros(ncx+1, ncy+1, ncz+1)
Gxy   = @zeros(ncx+1, ncy+1, ncz+0)
Gxz   = @zeros(ncx+1, ncy+0, ncz+1)
Gyz   = @zeros(ncx+0, ncy+1, ncz+1)
∇V    = @zeros(ncx+0, ncy+0, ncz+0) 
εxx   = @zeros(ncx+0, ncy+0, ncz+0) 
εyy   = @zeros(ncx+0, ncy+0, ncz+0)
εzz   = @zeros(ncx+0, ncy+0, ncz+0)
εxy   = @zeros(ncx+1, ncy+1, ncz+0)
εxz   = @zeros(ncx+1, ncy+0, ncz+1)
εyz   = @zeros(ncx+0, ncy+1, ncz+1)
Fx    = @zeros(ncx+1, ncy+0, ncz+0)
Fy    = @zeros(ncx+0, ncy+1, ncz+0)
Fz    = @zeros(ncx+0, ncy+0, ncz+1)
Fp    = @zeros(ncx+0, ncy+0, ncz+0)
dVxdτ = @zeros(ncx+1, ncy+0, ncz+0)
dVydτ = @zeros(ncx+0, ncy+1, ncz+0)
dVzdτ = @zeros(ncx+0, ncy+0, ncz+1)
#-----------
xv  = LinRange(-Lx/2, Lx/2, ncx+1)
yv  = LinRange(-Ly/2, Ly/2, ncy+1)
zv  = LinRange(-Lz/2, Lz/2, ncz+1)
Δx, Δy, Δz = Lx/ncx, Ly/ncy, Lz/ncz
xce = LinRange(-Lx/2-Δx/2, Lx/2+Δx/2, ncx+2)
yce = LinRange(-Ly/2-Δy/2, Ly/2+Δy/2, ncy+2)
zce = LinRange(-Lz/2-Δz/2, Lz/2+Δz/2, ncz+2)
##########
@parallel InitialCondition( Vx, Vy, Vz, ηv, Gv, ε_BG, ∇V_BG, xv, yv, zv, xce, yce, zce, r, ηr, dρ, dρinc, β, βr, Gr )
P .= Pini
@parallel UpdateDensity( ρ, ρr, β, P, Pr, dρ, Pt, dPr )
# @parallel InterpV2C( ηc, ηv )
# @parallel InterpV2C( ηv[2:end-1,2:end-1,2:end-1], ηc )
# @parallel InterpV2C( ηc, ηv )
# @parallel InterpV2xyz( ηxy, ηxz, ηyz, ηv )
@parallel InterpV2Ce( ηc, ηv )
@parallel InterpV2C( ηv, ηc )
@parallel InterpV2Ce( ηc, ηv )
@parallel InterpV2xyz( ηxy, ηxz, ηyz, ηv )
#
@parallel InterpV2C( Gc, Gv )
@parallel InterpV2C( Gv[2:end-1,2:end-1,2:end-1], Gc )
@parallel InterpV2C( Gc, Gv )
@parallel InterpV2xyz( Gxy, Gxz, Gyz, Gv )
##########
niter  = 1e5
nout   = 500
Reopt  = 1*pi
cfl    = 0.5
ρnum   = cfl*Reopt/max(ncx,ncy,ncz)
tol    = 1e-8
η_ve   = 1.0/(1.0/maximum(ηc) + 1.0/(Gr*Δtr))
Δτ     = ρnum*Δy^2 / η_ve /6.1 * cfl
ΚΔτ    = cfl * Δtr/βr * Δx / Lx  * 10.0 
@printf("ρnum = %2.2e, Δτ = %2.2e, ΚΔτ = %2.2e %2.2e\n", ρnum, Δτ, ΚΔτ, maximum(ηc))
# P_1d2 = (P_1d/σc)
# dρdP  = 0.5641895835477563*dρinc.*exp.( .-(( P_1d2.-Pt)./dPr).^2 ) ./ dPr
# Δt_1d2 = Δtr*(1.0 .-  dρdP ./ max_dρdP./1.5)
##########
for it=1:nt
    ###
    dρdP   = 0.5641895835477563*dρinc.*exp.( .-((P[(ncx+2)÷2,(ncy+2)÷2,(ncz+2)÷2].-Pt)./dPr).^2 ) ./ dPr 
    Δt     = Δtr*(1.0 .-  dρdP ./ max_dρdP./1.1)
    η_ve   = 1.0/(1.0/maximum(ηc) + 1.0/(Gr*Δt))
    Δτ     = ρnum*Δy^2 / η_ve /6.1 * cfl
    ΚΔτ    = cfl * Δt/βr * Δx / Lx  * 5.0 
    @printf("##########################################\n")
    @printf("#### Time step %04d --- Δt = %2.2e P = %2.2e ####\n", it, Δt*tc, P[(ncx+2)÷2,(ncy+2)÷2,(ncz+2)÷2]*σc/1e9)
    @printf("##########################################\n")
    P0   .= P
    ρ0   .= ρ
    τxx0 .= τxx
    τyy0 .= τyy
    τzz0 .= τzz
    τxy0 .= τxy
    τxz0 .= τxz
    τyz0 .= τyz
    ###
    for iter=1:niter
        @parallel (1:size(Vy,2), 1:size(Vy,3)) bc_x!(Vy)
        @parallel (1:size(Vz,2), 1:size(Vz,3)) bc_x!(Vz)
        @parallel (1:size(Vx,1), 1:size(Vx,3)) bc_y!(Vx)
        @parallel (1:size(Vz,1), 1:size(Vz,3)) bc_y!(Vz)
        @parallel (1:size(Vx,1), 1:size(Vx,2)) bc_z!(Vx)
        @parallel (1:size(Vy,1), 1:size(Vy,2)) bc_z!(Vy)
        @parallel UpdateDensity( ρ, ρr, β, P, Pr, dρ, Pt, dPr )
        @parallel ComputeStrainRates( ∇V, εxx, εyy, εzz, εxy, εxz, εyz, Vx, Vy, Vz, Δx, Δy, Δz )
        
        @parallel StressOnCentroids( τxx, τyy, τzz, τxyc, τxzc, τyzc, εxx, εyy, εzz, εxy, εxz, εyz, τxx0, τyy0, τzz0, τxy0, τxz0, τyz0, ηc, Gc, Δt )
        @parallel ShearStressFromCentroids( τxy, τxz, τyz, τxyc, τxzc, τyzc )
        
        # @parallel ComputeStress( τxx, τyy, τzz, τxy, τxz, τyz, τxx0, τyy0, τzz0, τxy0, τxz0, τyz0, εxx, εyy, εzz, εxy, εxz, εyz, ηc, ηxy, ηxz, ηyz, Gc, Gxy, Gxz, Gyz, Δt )
        @parallel ComputeResiduals( Fx, Fy, Fz, Fp, τxx, τyy, τzz, τxy, τxz, τyz, P, ∇V, ρ, ρ0, Δx, Δy, Δz, Δt )
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
    ##########
    @printf("τxx : min = %2.4e --- max = %2.4e\n", minimum(τxx[2:end-1,2:end-1,2:end-1])*σc, maximum(τxx[2:end-1,2:end-1,2:end-1])*σc/1e9)
    @printf("τyy : min = %2.4e --- max = %2.4e\n", minimum(τyy[2:end-1,2:end-1,2:end-1])*σc, maximum(τyy[2:end-1,2:end-1,2:end-1])*σc/1e9)
    @printf("τzz : min = %2.4e --- max = %2.4e\n", minimum(τzz[2:end-1,2:end-1,2:end-1])*σc, maximum(τzz[2:end-1,2:end-1,2:end-1])*σc/1e9)
    @printf("τxzc: min = %2.4e --- max = %2.4e\n", minimum(τxzc[2:end-1,2:end-1,2:end-1])*σc, maximum(τxzc[2:end-1,2:end-1,2:end-1])*σc/1e9)
    @printf("P0  : min = %2.4e --- max = %2.4e\n", minimum( P0[2:end-1,2:end-1,2:end-1])*σc, maximum( P0[2:end-1,2:end-1,2:end-1])*σc/1e9)
    @printf("P   : min = %2.4e --- max = %2.4e\n", minimum(  P[2:end-1,2:end-1,2:end-1])*σc, maximum(  P[2:end-1,2:end-1,2:end-1])*σc/1e9)
    @printf("ρ0  : min = %2.4e --- max = %2.4e\n", minimum( ρ0)*ρc, maximum( ρ0)*ρc)
    @printf("ρ   : min = %2.4e --- max = %2.4e\n", minimum(  ρ)*ρc, maximum(  ρ)*ρc)
    ##########
    # @parallel ComputeStressFromCentroids( τxx, τyy, τzz, τxy, τxz, τyz, τxx0, τyy0, τzz0, εxx, εyy, εzz, εxy, εxz, εyz, ηc, Gc, Δt, τxyc, τxzc, τyzc )
    
    Pin = P[2:end-1,2:end-1,2:end-1]
    @parallel ComputeStressInvariant( τii, τxx, τyy, τzz, τxy, τxz, τyz )
    Fs .= τii .- C.*cosd(ϕ) .- Pin.*sind(ϕ)
    Ft .= Ptens  .- Pin
    @printf("max Ft = %2.2e\n", maximum(Ft))
    Y .= 0
    Y[Fs.>0.0 .&& Ft.>0.0 .&& Fs.>Ft] .= 1
    Y[Fs.>0.0 .&& Ft.>0.0 .&& Fs.<Ft] .= 2
    Y[Fs.>0.0 .&& Ft.<0.0           ] .= 2
    Y[Ft.>0.0 .&& Fs.<0.0           ] .= 1
    # p = heatmap(Vz[:, (ncy+1)÷2,:]'.*Vc)
    # p1 = heatmap(dρ[:, (size(dρ,2))÷2, :]'.*ρc)
    p1  = heatmap(xce[2:end-1].*Lc*1e2, zce[2:end-1].*Lc*1e2, Pin[:, (size(Pin,2))÷2, :]'.*σc./1e9, title="P [GPa]", aspect_ratio=1, xlims=(-Lx/2, Lx/2), c=:jet1 )
    #p3  = heatmap(xce[2:end-1].*Lc*1e2, zce[2:end-1].*Lc*1e2, τii[:, (size(τii,2))÷2, :]'.*σc./1e9, title="τii [GPa]", aspect_ratio=1, xlims=(-Lx/2, Lx/2) )
    # p3  = heatmap(xce[2:end-1].*Lc*1e2, zce[2:end-1].*Lc*1e2, Y[:, (size(Y,2))÷2, :]', title="Yield mode", aspect_ratio=1, xlims=(-Lx/2, Lx/2), c=:jet1 )

    # p1  = heatmap(xv.*Lc*1e2, zv.*Lc*1e2, τxz[:, (size(τxz,2))÷2, :]'.*σc./1e9, title="τxz [GPa]", aspect_ratio=1, xlims=(-Lx/2, Lx/2), c=:jet1 )
    # p3  = heatmap(xce[:].*Lc*1e2, zce[:].*Lc*1e2, τxzc[:, (size(τxzc,2))÷2, :]'.*σc./1e9, title="τxz [GPa]", aspect_ratio=1, xlims=(-Lx/2, Lx/2), c=:jet1 )
    p3  = heatmap(xce[2:end-1].*Lc*1e2, zce[2:end-1].*Lc*1e2, Y[:, (size(Y,2))÷2, :]', title="Yield mode", aspect_ratio=1, xlims=(-Lx/2, Lx/2), c=:jet1 )
    # p1  = heatmap(ηv[:, (size(ηv,2))÷2, :]'.*μc)
    p2  = plot(P_1d./1e9, ρ_1d,legend=false)
    p2  = scatter!(Pin[:].*σc./1e9, ρ[:].*ρc, xlabel="P [GPa]", ylabel="ρ [kg / m³]")
    p4  = plot(P_1d./1e9, τy_1d./1e9,legend=false)
    p4  = scatter!(Pin[:].*σc./1e9, τii[:].*σc./1e9, xlabel="P [GPa]", ylabel="τii [GPa]")
    display(plot(p1,p2,p3,p4))
# p1 = plot(P_1d./1e9, Δt_1d,legend=false)
# p1 = plot!(P_1d2*σc./1e9, Δt_1d2*tc,legend=false)
# display(p1)
end
#-----------
return nothing
end

@parallel_indices (i,j,k) function InitialCondition( Vx, Vy, Vz, ηv, Gv, ε_BG, ∇V_BG, xv, yv, zv, xce, yce, zce, r, ηr, dρ, dρinc, β, βr, Gr )
    ri, ro, ar = r, 2r, 2
    # Vertices
    if i<=size(Vx,1) Vx[i,j,k] = (-ε_BG + 1//3*∇V_BG)*xv[i] end
    if j<=size(Vy,2) Vy[i,j,k] = (        1//3*∇V_BG)*yv[j] end
    if k<=size(Vz,3) Vz[i,j,k] = ( ε_BG + 1//3*∇V_BG)*zv[k] end
    if i<=size(ηv,1) && j<=size(ηv,2) && k<=size(ηv,3) 
        Gv[i,j,k] = Gr 
        ηv[i,j,k] = ηr/100 
        if (xv[i]*xv[i]/(ar*ro)^2 + zv[k]*zv[k]/ro^2) < 1.0  ηv[i,j,k] = ηr      end 
        if (xv[i]*xv[i]/(ar*ri)^2 + zv[k]*zv[k]/ri^2) < 1.0  ηv[i,j,k] = ηr/100.0 end  
    end
    # Centroids
    if i<=size(dρ,1) && j<=size(dρ,2) && k<=size(dρ,3)
        β[i,j,k] = βr
        if (xce[i+1]*xce[i+1]/(ar*ri)^2 + zce[k+1]*zce[k+1]/ri^2) < 1.0  dρ[i,j,k] = dρinc  end  
        if (xce[i+1]*xce[i+1]/(ar*ro)^2 + zce[k+1]*zce[k+1]/ro^2) < 1.0  β[i,j,k]  = βr/1.5 end
        if (xce[i+1]*xce[i+1]/(ar*ri)^2 + zce[k+1]*zce[k+1]/ri^2) < 1.0  β[i,j,k]  = βr     end  
    end
    return nothing
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

@parallel_indices (i,j,k) function UpdateDensity(ρ, ρr, β, P, Pr, dρ, Pt, dPr)
    if i<=size(ρ, 1) && j<=size(ρ, 2) && k<=size(ρ, 3) ρr1 = ρr - dρ[i,j,k] * 1//2 * erfc( (P[i+1,j+1,k+1]-Pt)/dPr )  end
    if i<=size(ρ, 1) && j<=size(ρ, 2) && k<=size(ρ, 3) ρ[i,j,k] = ρr1 * exp( β[i,j,k]*(P[i+1,j+1,k+1] - Pr) ) end
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

@parallel_indices (i,j,k) function InterpV2C( ηc, ηv )
    if i<=size(ηc,1) && j<=size(ηc,2) && k<=size(ηc,3)
        ηc[i,j,k]  = 1.0/8.0*( ηv[i,  j,  k] + ηv[i+1,j,k  ] + ηv[i,j+1,k  ] + ηv[i,  j,k+1  ] )
        ηc[i,j,k] += 1.0/8.0*( ηv[i+1,j+1,k] + ηv[i+1,j,k+1] + ηv[i,j+1,k+1] + ηv[i+1,j+1,k+1] )
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
    if i<=size(εxx,1) && j<=size(εxx,2) && k<=size(εxx,3)
        dVxΔx      = (Vx[i+1,j+1,k+1] - Vx[i,j+1,k+1]) / Δx
        dVyΔy      = (Vy[i+1,j+1,k+1] - Vy[i+1,j,k+1]) / Δy
        dVzΔz      = (Vz[i+1,j+1,k+1] - Vz[i+1,j+1,k]) / Δz
        ∇V[i,j,k]  = dVxΔx + dVyΔy + dVzΔz
        εxx[i,j,k] = dVxΔx - 1//3 * ∇V[i,j,k]
        εyy[i,j,k] = dVyΔy - 1//3 * ∇V[i,j,k]
        εzz[i,j,k] = dVzΔz - 1//3 * ∇V[i,j,k]
    end
    if i<=size(εxy,1) && j<=size(εxy,2) && k<=size(εxy,3)
        dVxΔy      = (Vx[i,j+1,k+1] - Vx[i,j,k+1]) / Δy 
        dVyΔx      = (Vy[i+1,j,k+1] - Vy[i,j,k+1]) / Δx 
        εxy[i,j,k] = 1//2*(dVxΔy + dVyΔx)
    end
    if i<=size(εxz,1) && j<=size(εxz,2) && k<=size(εxz,3)
        dVxΔz      = (Vx[i  ,j+1,k+1] - Vx[i,j+1,k]) / Δz                     
        dVzΔx      = (Vz[i+1,j+1,k  ] - Vz[i,j+1,k]) / Δx 
        εxz[i,j,k] = 1//2*(dVxΔz + dVzΔx)
    end
    if i<=size(εyz,1) && j<=size(εyz,2) && k<=size(εyz,3)
        dVyΔz      = (Vy[i+1,j,k+1] - Vy[i+1,j,k]) / Δz 
        dVzΔy      = (Vz[i+1,j+1,k] - Vz[i+1,j,k]) / Δy 
        εyz[i,j,k] = 1//2*(dVyΔz + dVzΔy)
    end
    return nothing
end

@parallel_indices (i,j,k) function ComputeStress( τxx, τyy, τzz, τxy, τxz, τyz, τxx0, τyy0, τzz0, τxy0, τxz0, τyz0, εxx, εyy, εzz, εxy, εxz, εyz, ηc, ηxy, ηxz, ηyz, Gc, Gxy, Gxz, Gyz, Δt )
    if i<=size(εxx,1) && j<=size(εxx,2) && k<=size(εxx,3)
        η_e  = Gc[i,j,k]*Δt
        η_ve = 1.0 / ( 1.0/ηc[i+1,j+1,k+1] + 1.0/η_e )
        τxx[i+1,j+1,k+1] = 2*η_ve*( εxx[i,j,k] + τxx0[i+1,j+1,k+1]/(2η_e) )
        τyy[i+1,j+1,k+1] = 2*η_ve*( εyy[i,j,k] + τyy0[i+1,j+1,k+1]/(2η_e) )
        τzz[i+1,j+1,k+1] = 2*η_ve*( εzz[i,j,k] + τzz0[i+1,j+1,k+1]/(2η_e) )
    end
    if i<=size(εxy,1) && j<=size(εxy,2) && k<=size(εxy,3)
        η_e  = Gxy[i,j,k]*Δt
        η_ve = 1.0 / ( 1.0/ηxy[i,j,k] + 1.0/η_e )
        τxy[i,j,k] = 2*η_ve*( εxy[i,j,k] + τxy0[i,j,k]/(2η_e) )
    end
    if i<=size(εxz,1) && j<=size(εxz,2) && k<=size(εxz,3)
        η_e  = Gxz[i,j,k]*Δt
        η_ve = 1.0 / ( 1.0/ηxz[i,j,k] + 1.0/η_e )
        τxz[i,j,k] = 2*η_ve*( εxz[i,j,k] + τxz0[i,j,k]/(2η_e) )
    end
    if i<=size(εyz,1) && j<=size(εyz,2) && k<=size(εyz,3)
        η_e  = Gyz[i,j,k]*Δt
        η_ve = 1.0 / ( 1.0/ηyz[i,j,k] + 1.0/η_e )
        τyz[i,j,k] = 2*η_ve*( εyz[i,j,k] + τyz0[i,j,k]/(2η_e) ) 
    end
    return nothing
end

@parallel_indices (i,j,k) function ComputeResiduals( Fx, Fy, Fz, Fp, τxx, τyy, τzz, τxy, τxz, τyz, P, ∇V, ρ, ρ0, Δx, Δy, Δz, Δt )
    if i<=size(Fx,1) && j<=size(Fx,2) && k<=size(Fx,3)
        if i>1 && i<size(Fx,1) # avoid Dirichlets
            Fx[i,j,k]  = (τxx[i+1,j+1,k+1] - τxx[i,j+1,k+1]) / Δx
            Fx[i,j,k] -= (  P[i+1,j+1,k+1] -   P[i,j+1,k+1]) / Δx
            Fx[i,j,k] += (τxy[i,j+1,k] - τxy[i,j,k]) / Δy
            Fx[i,j,k] += (τxz[i,j,k+1] - τxz[i,j,k]) / Δz
        end
    end
    if i<=size(Fy,1) && j<=size(Fy,2) && k<=size(Fy,3)
        if j>1 && j<size(Fy,2) # avoid Dirichlets
            Fy[i,j,k]  = (τyy[i+1,j+1,k+1] - τyy[i+1,j,k+1]) / Δy
            Fy[i,j,k] -= (  P[i+1,j+1,k+1] -   P[i+1,j,k+1]) / Δy
            Fy[i,j,k] += (τxy[i+1,j,k] - τxy[i,j,k]) / Δx
            Fy[i,j,k] += (τyz[i,j,k+1] - τyz[i,j,k]) / Δz
        end
    end
    if i<=size(Fz,1) && j<=size(Fz,2) && k<=size(Fz,3)
        if k>1 && k<size(Fz,3) # avoid Dirichlets
            Fz[i,j,k]  = (τzz[i+1,j+1,k+1] - τzz[i+1,j+1,k]) / Δz
            Fz[i,j,k] -= (  P[i+1,j+1,k+1] -   P[i+1,j+1,k]) / Δz
            Fz[i,j,k] += (τxz[i+1,j,k] - τxz[i,j,k]) / Δx
            Fz[i,j,k] += (τyz[i,j+1,k] - τyz[i,j,k]) / Δy
        end
    end
    if i<=size(Fp,1) && j<=size(Fp,2) && k<=size(Fp,3)
        Fp[i,j,k] = -∇V[i,j,k] - (log( ρ[i,j,k] ) - log( ρ0[i,j,k])) / Δt
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

@parallel_indices (i,j,k) function StressOnCentroids( τxx, τyy, τzz, τxyc, τxzc, τyzc, εxx, εyy, εzz, εxy, εxz, εyz, τxx0, τyy0, τzz0, τxy0, τxz0, τyz0, ηc, Gc, Δt )
    if i<=size(τxx,1)-2 && j<=size(τxx,2)-2 && k<=size(τxx,3)-2
        # Normal
        η_e  = Gc[i,j,k]*Δt
        η_ve = 1.0 / ( 1.0/ηc[i+1,j+1,k+1] + 1.0/η_e )
        τxx[i+1,j+1,k+1] = 2η_ve*( εxx[i,j,k] + τxx0[i+1,j+1,k+1]/(2η_e) )
        τyy[i+1,j+1,k+1] = 2η_ve*( εyy[i,j,k] + τyy0[i+1,j+1,k+1]/(2η_e) )
        τzz[i+1,j+1,k+1] = 2η_ve*( εzz[i,j,k] + τzz0[i+1,j+1,k+1]/(2η_e) )
        # Shear
        εxyc = 0.25*(εxy[i,j,k] + εxy[i+1,j,k] + εxy[i,j+1,k] + εxy[i+1,j+1,k])
        εxzc = 0.25*(εxz[i,j,k] + εxz[i+1,j,k] + εxz[i,j,k+1] + εxz[i+1,j,k+1])
        εyzc = 0.25*(εyz[i,j,k] + εyz[i,j+1,k] + εyz[i,j,k+1] + εyz[i,j+1,k+1])
        τxyc0 = 0.25*(τxy0[i,j,k] + τxy0[i+1,j,k] + τxy0[i,j+1,k] + τxy0[i+1,j+1,k])
        τxzc0 = 0.25*(τxz0[i,j,k] + τxz0[i+1,j,k] + τxz0[i,j,k+1] + τxz0[i+1,j,k+1])
        τyzc0 = 0.25*(τyz0[i,j,k] + τyz0[i,j+1,k] + τyz0[i,j,k+1] + τyz0[i,j+1,k+1])
        τxyc[i+1,j+1,k+1] = 2η_ve*(εxyc + τxyc0/(2η_e) )
        τxzc[i+1,j+1,k+1] = 2η_ve*(εxzc + τxzc0/(2η_e) )
        τyzc[i+1,j+1,k+1] = 2η_ve*(εyzc + τyzc0/(2η_e) )
    end
    return nothing
end

@parallel_indices (i,j,k) function ShearStressFromCentroids( τxy, τxz, τyz, τxyc, τxzc, τyzc )
    if i<=size(τxy,1) && j<=size(τxy,2) && k<=size(τxy,3)
        # if i>1 && j>1 && i<=size(τxy,1)-1 && j<=size(τxy,2)-1
            τxy[i,j,k] = 0.25*(τxyc[i,j,k+1] + τxyc[i+1,j,k+1] + τxyc[i,j+1,k+1] + τxyc[i+1,j+1,k+1])
        # end
    end
    if i<=size(τxz,1) && j<=size(τxz,2) && k<=size(τxz,3)
        τxz[i,j,k] = 0.25*(τxzc[i,j+1,k] + τxzc[i+1,j+1,k] + τxzc[i,j+1,k+1] + τxzc[i+1,j+1,k+1])
    end
    if i<=size(τyz,1) && j<=size(τyz,2) && k<=size(τyz,3)
        τyz[i,j,k] = 0.25*(τyzc[i+1,j,k] + τyzc[i+1,j+1,k] + τyzc[i+1,j,k+1] + τyzc[i+1,j+1,k+1])
    end
    return nothing
end

@time main( 1 )
# @time main( 2 )