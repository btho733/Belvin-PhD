%%   <<<<Adapted from test_seq_opns_fib_proc_Whole_ClickN_Go>>>>   Everything in Single Click and GO
tic;
%Loading 3D data
%###############
clear;close all;clc;
% cd /hpc/btho733/ABI/pacedSheep01/medsci_poster/stain_normalisation_toolbox/pacedsheepimages/f2v1/synth_op2/;  %change this for new image
% cd V:\ABI\JZ\Fiber_2016\Work_2017\CentralSection50um3\necrotic_scale4;
cd /hpc_atog/btho733/ABI/JZ/Fiber_2016/Work_2017/CentralSection50um3/raw_25um3/bgdBlack;
% cd V:\ABI\JZ\Fiber_2016\Work_2017\CentralSection50um3\cut_for_elevation\bgdCorrect;
% cd /hpc_atog/btho733/ABI/JZ/Fiber_2016/Work_2017/CentralSection50um3/raw_25um3/bgdCorrect;
% cd /hpc_atog/btho733/ABI/JZ/Fiber_2016/Work_2017/CentralSection50um3/raw_50um3/bgdBlack;
len=200;  %change this for new image
for i=1:len
infile=sprintf('cut_%04d.tif',i-1);%outfile=sprintf('cut_%05d.tif',i-500);                       %change this for new image
im=imread(infile);
% im=im(y1:y2,x1:x2,:);
p1=im(:,:,1);
p2=im(:,:,2);
p3=im(:,:,3);

D1(:,:,i,1)=p1;
D1(:,:,i,2)=p2;
D1(:,:,i,3)=p3;
disp(i);
end

[Ny, Nx, Nz,k] = size(D1);
clear p1;clear p2;clear p3;

% CIRCULAR Filter : New(using fpecial circular filter) Filtering for All 3D Y-planes and X-planes : <<<<   RGB Correct version  >>>>>>>
%######################################################################################################################################
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI;
circular_block=squeeze(D1(:,:,:,1));
corrected_zplane=255*ones(Ny,Nx);upto=Nz-1;
for zplane=1:upto
    r=10;
    if(zplane==1)       
        im1=squeeze(D1(:,:,zplane,1));
        im2=squeeze(D1(:,:,zplane+1,1));
    else
        im1=corrected_zplane;
        im2=squeeze(D1(:,:,zplane+1,1));
    end
% fspecial with disk argument takes the average of all elements in a window around each location. 
% Let's say you were looking at an element (pixel) at (100,200) and had a disc of radius 10. 
% So it would take all pixels in a circle from 90 to 110 and 190 to 210, 
% multiply them by the values in the fspecial array, 
% which are 1's in the disc, and 0's in the corners which are not included in the disc, 
% then sum these average values, and set it as output value at (100, 200). This will give the effect of blurring the array (image).
    H = fspecial('disk',r);
    avg1 = imfilter(im1,H,'replicate'); 
    avg2 = imfilter(im2,H,'replicate'); 
    diff=avg1-avg2;
    corrected_zplane=im2+diff;
    circular_block(:,:,zplane+1)=corrected_zplane;
end

% cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Large_block_25um_filtered/Zplanes/grey;
% len=200;  %change this for new image
% for i=1:len
% infile=sprintf('z_%04d.tif',i-1);                      
% im=imread(infile);
% Df(:,:,len-i+1)=im;
% 
% end
% cd V:\ABI\pacedSheep01\Anisotropic\functions;
% showcs3(single(Df));
V1=circular_block;
[Ny, Nx, Nz] = size(V1)

% V=single(D1);
% cd /hpc/btho733/ABI/pacedSheep01/Anisotropic/;
% JR = CoherenceFilter(V,struct('T',20,'dt',2,'Scheme','R'));


sigma =1;

InvV1 =  single(zeros([Ny Nx Nz]));
for i = 1:1:Nz
    InvV1(:,:,i) = 255-(squeeze(V1(:,:,i))); % Changed from rgb3grey to only red-plane
end

% cd V:\ABI\JZ\Fiber_DTI;
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/;
usigma=imgaussian(InvV1,sigma,3);

% Calculate the gradients
ux=derivatives(usigma,'x');
uy=derivatives(usigma,'y');
uz=derivatives(usigma,'z');
% [ux,uy,uz] = imgradientxyz(usigma,'sobel');
% Compute the 3D structure tensors J of the image
rho =8;
[Jxx, Jxy, Jxz, Jyy, Jyz, Jzz] = StructureTensor3D(ux,uy,uz,rho);
%[Jxx, Jxy, Jxz, Jyy, Jyz, Jzz] = AveStructureTensor3D(V,ux,uy,uz,rho);

disp('Done Constructing tensor');

parametersDTI=[];
parametersDTI.BackgroundTreshold=10;
parametersDTI.WhiteMatterExtractionThreshold=0.10;
parametersDTI.textdisplay=true;
disp('ST computation in progress');
% Perform ST Analysis
[coh,EV,VectorF]=testStructureFiber3D1(InvV1,Jxx, Jxy, Jxz, Jyy, Jyz, Jzz,parametersDTI);
toc;

tic;
disp('Computed EV and VectorF...Saving now');
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/Largeblock/results;
save('EV_s1m3rho8k40.mat','EV','-v7.3');
disp('saved EV...moving to VectorF');
save('VectorF_s1m3rho8k40.mat','VectorF','-v7.3');
toc;

tic;
disp('Computing Structural Anisotropy parameters and Fiber orientation estimates');
l1=squeeze(EV(:,:,:,1));l2=squeeze(EV(:,:,:,2));l3=squeeze(EV(:,:,:,3));
FAv=(1/sqrt(2)).*( sqrt((l3-l2).^2+(l2-l1).^2+(l3-l1).^2)./sqrt(l1.^2+l2.^2+l3.^2) );
cl=(l3-l2)./(l1+l2+l3);cp=2.*(l2-l1)./(l1+l2+l3);cs=3*l1./(l1+l2+l3);
MA_minus=-1.*((0.5*(-l3-l2+2*l1).*(2*l3-l2-l1).*(-l3+2.*l2-l1))./((l3.^2+l2.^2+l1.^2-l1.*l2-l2.*l3-l3.*l1).^1.5));


y=squeeze(VectorF(:,:,:,1));x=squeeze(VectorF(:,:,:,2));z=squeeze(VectorF(:,:,:,3));
projy=(atan2(-z,x)).*(180/pi);
neg_iprojz=(atan2(-y,x)).*(180/pi);
projx=(atan2(-z,y)).*(180/pi);
elevz=(atan2(z,sqrt(x.^2+y.^2))).*(180/pi);
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

[p,q,r]=size(neg_iprojz);
for i= 1:p
    for j=1:q
       for k=1:r 
        theta_element=neg_iprojz(i,j,k);
        
        if theta_element >90
            temp =theta_element-180;
        elseif theta_element <-90
            temp = theta_element+180;
        else 
            temp=theta_element;
        end
        
        neg_ithetaz(i,j,k)=temp;
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
h_z=(neg_ithetaz+90)./180; % converting from (-90,90) to (0,1)
h_elevz=(elevz+90)./180;
s=coh;%ones(560,1000,200);
b=(InvV1./255);
hsb_x=single(zeros(Ny,Nx,Nz));
hsb_y=single(zeros(Ny,Nx,Nz));
hsb_z=single(zeros(Ny,Nx,Nz));
hsb_elevz=single(zeros(Ny,Nx,Nz));
for i=1:r
    hsb_x(:,:,i,1)=h_x(:,:,i);hsb_x(:,:,i,2)=s(:,:,i);hsb_x(:,:,i,3)=b(:,:,i);rgb_x(:,:,i,:)=hsv2rgb(squeeze(hsb_x(:,:,i,:)));
    hsb_y(:,:,i,1)=h_y(:,:,i);hsb_y(:,:,i,2)=s(:,:,i);hsb_y(:,:,i,3)=b(:,:,i);rgb_y(:,:,i,:)=hsv2rgb(squeeze(hsb_y(:,:,i,:)));
    hsb_z(:,:,i,1)=h_z(:,:,i);hsb_z(:,:,i,2)=s(:,:,i);hsb_z(:,:,i,3)=b(:,:,i);rgb_z(:,:,i,:)=hsv2rgb(squeeze(hsb_z(:,:,i,:)));   
    hsb_elevz(:,:,i,1)=h_elevz(:,:,i);hsb_elevz(:,:,i,2)=s(:,:,i);hsb_elevz(:,:,i,3)=b(:,:,i);rgb_elevz(:,:,i,:)=hsv2rgb(squeeze(hsb_elevz(:,:,i,:)));   
end

toc;

tic;
disp('Saving the SA and Fiber orientation estimates');
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/Largeblock/results;
save('FA.mat','FAv','-v7.3');
save('MA.mat','MA_minus','-v7.3');
save('cl.mat','cl','-v7.3');
save('cp.mat','cp','-v7.3');
save('cs.mat','cs','-v7.3');
save('elevz.mat','elevz','-v7.3');
save('rgbx_s1m3rho8k40_Largeblock.mat','rgb_x','-v7.3');
save('rgby_s1m3rho8k40_Largeblock.mat','rgb_y','-v7.3');
save('rgbz_s1m3rho8k40_Largeblock.mat','rgb_z','-v7.3');
save('rgb_elevz_s1m3rho8k40_Largeblock.mat','rgb_elevz','-v7.3');
toc;

tic;
% writing the output images
%##########################
% clear;clc;
% cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Large_block_25um_filtered/ST3D;
% rgb=importdata('rgb_thetay_orient_Large_Sig3kernel9_rho1_04Dec17.mat');
[~,~,r,~]=size(rgb_y);
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/Largeblock/results/;
for i=1:r
    plane_x=squeeze(rgb_x(:,:,i,:));namex=sprintf('x_%04d.png',i-1);cd rgbx;imwrite(plane_x,namex);cd ..;
    plane_y=squeeze(rgb_y(:,:,i,:));namey=sprintf('y_%04d.png',i-1);cd rgby;imwrite(plane_y,namey);cd ..;
    plane_z=squeeze(rgb_z(:,:,i,:));namez=sprintf('z_%04d.png',i-1);cd rgbz;imwrite(plane_z,namez);cd ..;
    plane_elevz=squeeze(rgb_elevz(:,:,i,:));name_elevz=sprintf('elevz_%04d.png',i-1);cd rgb_elevz;imwrite(plane_elevz,name_elevz);cd ..;  
end
toc;
disp('Completed successfully....exiting');
exit;