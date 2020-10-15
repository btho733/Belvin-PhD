function [Jxx, Jxy, Jxz, Jyy, Jyz, Jzz]=StructureTensor3D(ux,uy,uz,rho)

Jxx = single(ux.^2);
Jxy = single(ux.*uy);
Jxz = single(ux.*uz);
Jyy = single(uy.^2);
Jyz = single(uy.*uz);
Jzz = single(uz.^2);

% Do the gaussian smoothing
Jxx = imgaussian(Jxx,rho,5*rho);
Jxy = imgaussian(Jxy,rho,5*rho);
Jxz = imgaussian(Jxz,rho,5*rho);
Jyy = imgaussian(Jyy,rho,5*rho);
Jyz = imgaussian(Jyz,rho,5*rho);
Jzz = imgaussian(Jzz,rho,5*rho);
