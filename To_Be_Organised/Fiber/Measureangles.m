clear all; close all; clc
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI;
load PVblock/HighResPVsmallblocksmoothedVF2.mat;




[Ny, Nx, Nz,k] = size(VectorF)

%VectorF = VectorF(end:-1:1,:,:,:);
%load LowResWholeAtriaUpdat6FAROIV2.mat;
%Angles = zeros([Ny Nx Nz],'single');
MeasureAngles = single(zeros([Ny Nx Nz],'single') - 0.1);
%AnglesTheta = single(zeros([Ny Nx Nz],'single') - pi);


for plane=1:1:Nz,
    for Yi=1:1:Ny,
        for Xi=1:1:Nx,
           % if V(Yi, Xi, plane) > 100,
                % only work on tissue region
                y = VectorF(Yi,Xi,plane,1);
                x = VectorF(Yi,Xi,plane,2);
                z = VectorF(Yi,Xi,plane,3);
                %theta = atan2(y,x);
                phi = atan2(z,sqrt(y.^2 + x.^2));
                r = sqrt(x.^2 + y.^2 + z.^2);
                %Angles(Yi,Xi,plane) =  phi;
                %if theta < 0,
                %    AnglesTheta(Yi,Xi,plane) =  theta + pi;
                %else
                %    AnglesTheta(Yi,Xi,plane) =  theta;
                %end;
                if (abs(phi) >= pi/4),
                    MeasureAngles(Yi,Xi,plane) = (abs(phi) - pi/4)/(pi/4)*0.5+0.5;
                else
                    MeasureAngles(Yi,Xi,plane) = abs(phi)/(pi/4)*0.5;
                end;
            end;
        end;
    end;


save -v7.3 Angles.mat MeasureAngles;