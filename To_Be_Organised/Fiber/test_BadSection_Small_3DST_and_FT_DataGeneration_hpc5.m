%% Badsection_Small_FAST version - With Saving only the final VV_FiberTracks
tic;
clear;close all;clc;
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Badsection_small/a101_zplanes_corrected;
len=70;  %change this for new image
for i=1:len
infile=sprintf('z_%04d.tif',i-1);%outfile=sprintf('cut_%05d.tif',i-500);                       %change this for new image
im=imread(infile);
V1(:,:,i)=squeeze(im(:,:,1));
end
[Ny, Nx, Nz] = size(V1)
sigma =1;

InvV1 =  single(zeros([Ny Nx Nz]));
for i = 1:1:Nz
    InvV1(:,:,i) = 255-(squeeze(V1(:,:,i))); % Changed from rgb3grey to only red-plane
end

cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI;
usigma=imgaussian(InvV1,sigma,3);

% Calculate the gradients
ux=derivatives(usigma,'x');
uy=derivatives(usigma,'y');
uz=derivatives(usigma,'z');
% [ux,uy,uz] = imgradientxyz(usigma,'sobel');
% Compute the 3D structure tensors J of the image
rho =2;
[Jxx, Jxy, Jxz, Jyy, Jyz, Jzz] = StructureTensor3D(ux,uy,uz,rho);
%[Jxx, Jxy, Jxz, Jyy, Jyz, Jzz] = AveStructureTensor3D(V,ux,uy,uz,rho);

disp('Done Constructing tensor');

parametersDTI=[];
parametersDTI.BackgroundTreshold=10;
parametersDTI.WhiteMatterExtractionThreshold=0.10;
parametersDTI.textdisplay=true;

% Perform ST Analysis
[coh,EV,VectorF]=testStructureFiber3D1(InvV1,Jxx, Jxy, Jxz, Jyy, Jyz, Jzz,parametersDTI);





%#######################################################
% Midsmallcut  : Computing Angles and creating HSV model
%########################################################

y=squeeze(VectorF(:,:,:,1));x=squeeze(VectorF(:,:,:,2));z=squeeze(VectorF(:,:,:,3));
projy=(atan2(-z,x)).*(180/pi);
projx=(atan2(-z,y)).*(180/pi);
projz=(atan2(-y,x)).*(180/pi);
% converting angle range from -180to180 to -90to90
%##################################################

[p,q,r]=size(projy);
for i= 1:p
    for j=1:q
       for k=1:r 
        theta_element=projy(i,j,k);
        
        if theta_element >90
            temp =theta_element-180;
        elseif theta_element <-90
            temp = theta_element+180;
        else 
            temp=theta_element;
        end
        
        thetay(i,j,k)=temp;
       end
    end
end
clear temp;

[p,q,r]=size(projz);
for i= 1:p
    for j=1:q
       for k=1:r 
        theta_element=projz(i,j,k);
        
        if theta_element >90
            temp =theta_element-180;
        elseif theta_element <-90
            temp = theta_element+180;
        else 
            temp=theta_element;
        end
        
        thetaz(i,j,k)=temp;
       end
    end
end
clear temp;

[p,q,r]=size(projx);
for i= 1:p
    for j=1:q
       for k=1:r 
        theta_element=projx(i,j,k);
        
        if theta_element >90
            temp =theta_element-180;
        elseif theta_element <-90
            temp = theta_element+180;
        else 
            temp=theta_element;
        end
        
        thetax(i,j,k)=temp;
       end
    end
end
clear temp;
% Making original redplane inverted for B plane of HSB
% ####################################################
% cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Badsection_small
% len=70;  %change this for new image
% for i=1:len
% infile=sprintf('cut_%04d.tif',i-1);     %change this for new image
% im=imread(infile);
% V(:,:,i)=squeeze(im(:,:,1));
% end
% [Ny, Nx, Nz] = size(V)
% orig =  single(zeros([Ny Nx Nz]));
% for i = 1:1:Nz
%     orig(:,:,i) = 255-(squeeze(V(end:-1:1,:,i))); % Changed from rgb3grey to only red-plane
% end
% Making the hsv model of orientation 
% ###################################
% h=orientation angle, 
% s=1(can be replaced by coherence for better results), 
% v=inverted red channel of original(can be the inverted greyscale version also)
% The hsv model is converted to rgb for visualisation purpose(In order to
% match with the visualisation from 2D orientation)
% clear h;clear s;clear b;clear hsb;clear rgb;
h_x=(thetax+90)./180; % converting from (-90,90) to (0,1)
h_y=(thetay+90)./180; % converting from (-90,90) to (0,1)
h_z=(thetaz+90)./180; % converting from (-90,90) to (0,1)
s=coh;%ones(560,1000,200);
b=(InvV1./255);
hsb_x=single(zeros(Ny,Nx,Nz));
hsb_y=single(zeros(Ny,Nx,Nz));
hsb_z=single(zeros(Ny,Nx,Nz));
for i=1:r
    hsb_x(:,:,i,1)=h_x(:,:,i);
    hsb_y(:,:,i,1)=h_y(:,:,i);
    hsb_z(:,:,i,1)=h_z(:,:,i);
    hsb_x(:,:,i,2)=s(:,:,i);
    hsb_y(:,:,i,2)=s(:,:,i);
    hsb_z(:,:,i,2)=s(:,:,i);
    hsb_x(:,:,i,3)=b(:,:,i);
    hsb_y(:,:,i,3)=b(:,:,i);
    hsb_z(:,:,i,3)=b(:,:,i);
    rgb_x(:,:,i,:)=hsv2rgb(squeeze(hsb_x(:,:,i,:)));
    rgb_y(:,:,i,:)=hsv2rgb(squeeze(hsb_y(:,:,i,:)));
    rgb_z(:,:,i,:)=hsv2rgb(squeeze(hsb_z(:,:,i,:)));
end
% save('rgb_neg_ithetaz_orient_Large_Sig1kernel3_rho10_ker50_07Dec17.mat','rgb');




% #########################
% writing the output images
% #########################
% clear;clc;
% cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Large_block_25um_filtered/ST3D;
% rgb=importdata('rgb_thetay_orient_Large_Sig3kernel9_rho1_04Dec17.mat');
[~,~,r,~]=size(rgb_y);
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Badsection_small/a301_ST3D_results/s1m3rho2k7/;
for i=1:r
%     plane_x=squeeze(rgb_x(:,:,i,:));
%     plane_y=squeeze(rgb_y(:,:,i,:));
%     plane_z=squeeze(rgb_z(:,:,i,:));
    plane_revx=squeeze(rgb_x(:,:,r-i+1,:));
    plane_revy=squeeze(rgb_y(:,:,r-i+1,:));
    plane_revz=squeeze(rgb_z(:,:,r-i+1,:));
    namex=sprintf('x_%04d.png',i-1);
    namey=sprintf('y_%04d.png',i-1);
    namez=sprintf('z_%04d.png',i-1);
%     cd x;imwrite(plane_x,namex);cd ..;
%     cd y;imwrite(plane_y,namey);cd ..;
%     cd z;imwrite(plane_z,namez);cd ..;
    cd revx/;imwrite(plane_revx,namex);cd ..;
    cd revy;imwrite(plane_revy,namey);cd ..;
    cd revz;imwrite(plane_revz,namez);cd ..;
end
toc;