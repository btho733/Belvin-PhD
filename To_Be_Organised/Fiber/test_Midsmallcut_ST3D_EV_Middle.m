%% MidSmallcut_FAST version - With Saving only the final VV_FiberTracks
tic;
clear;close all;clc;
% cd /hpc/btho733/ABI/pacedSheep01/medsci_poster/stain_normalisation_toolbox/pacedsheepimages/f2v1/synth_op2/;  %change this for new image
% cd V:\ABI\pacedSheep01\medsci_poster\stain_normalisation_toolbox\pacedsheepimages\v8\output_RH\Nothreshold; 
% cd V:\ABI\JZ\Fiber_2016\Work_2017\CentralSection50um3\cut_for_elevation\bgdCorrect;
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Large_block_25um_filtered/Zplanes;
% cd /hpc_atog/btho733/ABI/JZ/Fiber_2016/Work_2017/CentralSection50um3/raw_25um3/bgdCorrect;
len=100;  %change this for new image
for i=1:len
infile=sprintf('z_%04d.tif',i-1);%outfile=sprintf('cut_%05d.tif',i-500);                       %change this for new image
im=imread(infile);
D1(:,:,i)=squeeze(im(:,:,1));
end
V1=D1(101:250,101:400,1:100);
[Ny, Nx, Nz] = size(V1)

% V=single(D1);
% cd /hpc/btho733/ABI/pacedSheep01/Anisotropic/;
% JR = CoherenceFilter(V,struct('T',20,'dt',2,'Scheme','R'));


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
rho =5;
[Jxx, Jxy, Jxz, Jyy, Jyz, Jzz] = StructureTensor3D(ux,uy,uz,rho);
%[Jxx, Jxy, Jxz, Jyy, Jyz, Jzz] = AveStructureTensor3D(V,ux,uy,uz,rho);

disp('Done Constructing tensor');

parametersDTI=[];
parametersDTI.BackgroundTreshold=10;
parametersDTI.WhiteMatterExtractionThreshold=0.10;
parametersDTI.textdisplay=true;

% Perform ST Analysis
[coh,EV,VectorF]=testStructureFiber3D2(InvV1,Jxx, Jxy, Jxz, Jyy, Jyz, Jzz,parametersDTI);





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
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/MidSmallCut/;
len=100;  %change this for new image
for i=1:len
infile=sprintf('cut_%04d.tif',i-1);     %change this for new image
im=imread(infile);
V(:,:,i)=squeeze(im(:,:,1));
end
[Ny, Nx, Nz] = size(V)
orig =  single(zeros([Ny Nx Nz]));
for i = 1:1:Nz
    orig(:,:,i) = 255-(squeeze(V(end:-1:1,:,i))); % Changed from rgb3grey to only red-plane
end
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
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/MidSmallCut/a302_ST3D_results_EVmiddle/s1m3rho5k25/;
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