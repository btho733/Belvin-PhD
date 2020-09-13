function [coh,l1,l2,l3,VectorF] = Prop_Parallel_testStructureFiber3D1(u,Dxx, Dxy, Dxz, Dyy, Dyz, Dzz,parameters)
%function [FA,VectorF]=testStructureFiber3D1(u,Dxx, Dxy, Dxz, Dyy, Dyz, Dzz,parameters)
%function [FA,VectorF,DifT]=StructureFiber3D(u,Dxx, Dxy, Dxz, Dyy, Dyz, Dzz,parameters)
[nx,ny,nz] = size(u);

% Create a matrix to store the fractional anistropy (FA)
%FA=zeros(nx,ny,nz,'single');
% Create a maxtrix to store the (main) fiber direction in each pixel
VectorF=zeros([nx ny nz 3],'single');
%EV=zeros([nx ny nz 3],'single');
coh=zeros([nx ny nz],'single');
%DifT=zeros([ny nx nz 6],'single');
l1=zeros([nx ny nz],'single');
l2=zeros([nx ny nz],'single');
l3=zeros([nx ny nz],'single');
% Loop through all voxel coordinates
parfor x=1:nx
    for y=1:ny
        for z=1:nz
            % Only process a pixel if it isn't background
            if(u(x,y,z)>parameters.BackgroundTreshold)
                
                % The DiffusionTensor (Remember it is a symetric matrix,
                % thus for instance Dxy == Dyx)
                DiffusionTensor= [Dxx(x,y,z) Dxy(x,y,z) Dxz(x,y,z); Dxy(x,y,z) Dyy(x,y,z) Dyz(x,y,z); Dxz(x,y,z) Dyz(x,y,z) Dzz(x,y,z)];
                % Calculate the eigenvalues and vectors, and sort the 
                % eigenvalues from small to large
                % Ascending order (default)
                [EigenVectors,D]=eig(DiffusionTensor); EigenValues=diag(D);
                [t,index]=sort(abs(EigenValues)); 
                EigenValues=EigenValues(index); EigenVectors=EigenVectors(:,index);
                EigenValues_old=EigenValues;
                
                % Regulating of the eigen values (negative eigenvalues are
                % due to noise and other non-idealities of MRI)
                if((EigenValues(1)<0)&&(EigenValues(2)<0)&&(EigenValues(3)<0)), EigenValues=abs(EigenValues);end
                if(EigenValues(1)<=0), EigenValues(1)=eps; end
                if(EigenValues(2)<=0), EigenValues(2)=eps; end
                
                % Apparent Diffuse Coefficient
                ADCv=(EigenValues(1)+EigenValues(2)+EigenValues(3))/3;
                
                % Fractional Anistropy (2 different definitions exist)
                % First FA definition:
              %FAv=(1/sqrt(2))*( sqrt((EigenValues(1)-EigenValues(2)).^2+(EigenValues(2)-EigenValues(3)).^2+(EigenValues(1)-EigenValues(3)).^2)./sqrt(EigenValues(1).^2+EigenValues(2).^2+EigenValues(3).^2) );
                % Second FA definition:
                FAv=sqrt(1.5)*( sqrt((EigenValues(1)-ADCv).^2+(EigenValues(2)-ADCv).^2+(EigenValues(3)-ADCv).^2)./sqrt(EigenValues(1).^2+EigenValues(2).^2+EigenValues(3).^2) );
                %defined by jichao zhao
                %FA(x,y,z,1)=(EigenValues(3) - EigenValues(2))./EigenValues(3);  
                %FA(x,y,z,2)=(EigenValues(2) - EigenValues(1))./EigenValues(3); 
                %FA(x,y,z,3)=EigenValues(1)./EigenValues(3);
                %FA(x,y,z,1)=(EigenValues(3) - EigenValues(2))./(EigenValues(3) + EigenValues(2));
                %FA(x,y,z,2)=(EigenValues(2) - EigenValues(1))./(EigenValues(1) + EigenValues(2));
                if(FAv>parameters.WhiteMatterExtractionThreshold)
                    %FA(x,y,z)=FAv;
                     VectorF(x,y,z,:)=EigenVectors(:,1)*EigenValues_old(1);
                    l1(x,y,z)=EigenValues_old(1);
                    l2(x,y,z)=EigenValues_old(2);
                    l3(x,y,z)=EigenValues_old(3);
                    coh(x,y,z)=(EigenValues_old(3)-EigenValues_old(1))./(EigenValues_old(3)+EigenValues_old(1));
                    %[EigenValues_old(1),EigenValues_old(2),EigenValues_old(3)]
                    %VectorF(x,y,z,:)=EigenVectors(:,end)*EigenValues_old(end);
                    %VectorF(x,y,z,1)= -VectorF(x,y,z,1); 
                    %VectorF(y,x,z,:)=EigenVectors(:,1)*EigenValues_old(1);
                end
            end
        end
    end
end

