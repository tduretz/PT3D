clear


step = 1; 

Lx   = 5e-3;
betr = 1.0/(17e10);
Pini = 3.45e9;
etar = 1e22;

Lc = Lx;
tc = etar*betr;
Ec = 1.0/tc;
sigc = Pini;
rhoc = sigc*tc^2/Lc^2;
muc = sigc*tc;
Vc = Lc/tc;
% 
fname = ['./Breakpoint', num2str(step, '%05d'), '.h5'];

for iy=1:5:190

figure(1), clf
subplot(221)
P     = hdf5read(fname,'P');
P1 = reshape( P(2:end-1, iy, 2:end-1), size(P,1)-2,  size(P,3)-2);
imagesc(P1'*sigc/1e6); colorbar, axis xy image
subplot(222)
rho     = hdf5read(fname,'rho');
rho1 = reshape( rho(:, iy, :), size(rho,1)-0,  size(rho,3)-0);
imagesc(rho1'*rhoc); colorbar, axis xy image
subplot(223)
lam     = hdf5read(fname,'lam');
lam1 = reshape( lam(:, iy, :), size(lam,1)-0,  size(lam,3)-0);
imagesc(lam1'*Ec); colorbar, axis xy image
subplot(224)
% Eii     = hdf5read(fname,'Eii');
% Eii1 = reshape( Eii(:, iy, :), size(Eii,1)-0,  size(Eii,3)-0);
% imagesc(Eii1'*Ec); colorbar, axis xy image
Gv     = hdf5read(fname,'Gv');
Gv1 = reshape( Gv(:, iy, :), size(Gv,1)-0,  size(Gv,3)-0);
imagesc(Gv1'*sigc); colorbar, axis xy image

drawnow
end
