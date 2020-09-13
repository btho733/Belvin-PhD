clear all; close all; clc
%% Small block of fibers & Angles to vtk binary file (uses vtkwrite function)
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI;
load PVblock/HighResPVsmallblocksmoothedVF2.mat;
load Angles.mat;
[X,Y,Z] = meshgrid(1:200,1:200,1:200);
px=VectorF(1:200,1:200,1:200,1);
py=VectorF(1:200,1:200,1:200,2);
pz=VectorF(1:200,1:200,1:200,3);
cd /hpc/btho733/ABI/JZ/Fiber_DTI/normalisevec
[u1x, u1y, u1z] = normvec(px, py, pz); 
vtkwrite('Small_Norm_FiberAndAngles.vtk', 'structured_grid',X,Y,Z, 'vectors','fibre_field',u1x, u1y, u1z,'scalars','Angle_values',MeasureAngles(1:200,1:200,1:200));
%% Small block Geometry to vtk ascii file (uses /hpc_atog/btho733/ABI/JZ/Fiber_DTI/marthawritevtkfun2fil.m function)
clear all; close all; clc;
load HighResPVsmallblockfibers.mat;
Geo_block=BBori(1:200,1:200,1:200);
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI;
marthawritevtkfun2fil(Geo_block,'geo_ascii.vtk');


