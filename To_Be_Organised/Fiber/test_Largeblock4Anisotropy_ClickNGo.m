clear;close all;clc;
% cd V:\ABI\JZ\Fiber_DTI\Large_block_25um_filtered\Zplanes\grey\;
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Large_block_25um_filtered/Zplanes/grey;
len=200;  %change this for new image
for i=1:len
infile=sprintf('z_%04d.tif',i-1);                      
im=imread(infile);
Df(:,:,len-i+1)=im;

end
% cd V:\ABI\pacedSheep01\Anisotropic\functions;
% showcs3(single(Df));
V1=Df;
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
rho =1;
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

tic;
disp('Computed EV and VectorF...Saving now');
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/Largeblock4Anisotropy/s1m3rho1k5/;
save('EV_s1m3rho1k5.mat','EV','-v7.3');
disp('saved EV...moving to VectorF');
save('VectorF_s1m3rho1k5.mat','VectorF','-v7.3');
toc;

tic;
disp('Computing Structural Anisotropy parameters and Fiber orientation estimates');
o1=squeeze(EV(:,:,:,1));o2=squeeze(EV(:,:,:,2));o3=squeeze(EV(:,:,:,3));
l3=o1;l1=o3;l2=o1+o3-o2;   % ST Eigen value transformation
FAv=(1/sqrt(2)).*( sqrt((l1-l2).^2+(l2-l3).^2+(l3-l1).^2)./sqrt(l1.^2+l2.^2+l3.^2) );
cl=(l1-l2)./(l1+l2+l3);cp=2.*(l2-l3)./(l1+l2+l3);cs=3*l3./(l1+l2+l3);
clw=(l1-l2)./l1;cpw=(l2-l3)./l1;csw=l3./l1;
MA=((0.5*(-l1-l2+2*l3).*(2*l1-l2-l3).*(-l1+2.*l2-l3))./((l1.^2+l2.^2+l3.^2-l1.*l2-l2.*l3-l3.*l1).^1.5));


y=squeeze(VectorF(:,:,:,1));x=squeeze(VectorF(:,:,:,2));z=squeeze(VectorF(:,:,:,3));
projy=(atan2(-z,x)).*(180/pi);
% neg_iprojz=(atan2(-y,x)).*(180/pi);
% projx=(atan2(-z,y)).*(180/pi);
% elevz=(atan2(-z,sqrt(x.^2+y.^2))).*(180/pi);
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

% [p,q,r]=size(neg_iprojz);
% for i= 1:p
%     for j=1:q
%        for k=1:r 
%         theta_element=neg_iprojz(i,j,k);
%         
%         if theta_element >90
%             temp =theta_element-180;
%         elseif theta_element <-90
%             temp = theta_element+180;
%         else 
%             temp=theta_element;
%         end
%         
%         neg_ithetaz(i,j,k)=temp;
%        end
%     end
% end
% clear temp;

% [p,q,r]=size(projx);
% for i= 1:p
%     for j=1:q
%        for k=1:r 
%         theta_element=projx(i,j,k);
%         
%         if theta_element >90
%             temp =theta_element-180;
%         elseif theta_element <-90
%             temp = theta_element+180;
%         else 
%             temp=theta_element;
%         end
%         
%         thetax(i,j,k)=temp;
%        end
%     end
% end
% clear temp;

% Making the hsv model of orientation 
% ###################################
% h=orientation angle, 
% s=1(can be replaced by coherence for better results), 
% v=inverted red channel of original(can be the inverted greyscale version also)
% The hsv model is converted to rgb for visualisation purpose(In order to
% match with the visualisation from 2D orientation)
% clear h;clear s;clear b;clear hsb;clear rgb;
% h_x=(thetax+90)./180; % converting from (-90,90) to (0,1)
h_y=(thetay+90)./180; % converting from (-90,90) to (0,1)
% h_z=(neg_ithetaz+90)./180; % converting from (-90,90) to (0,1)
% h_elevz=(elevz+90)./180;
s=coh;%ones(560,1000,200);
MA_norm=(MA+1)./2;
s2=FAv.*MA_norm;
onematrix=ones(Ny,Nx,Nz);
s3=onematrix-s2;
b=(InvV1./255);
% hsb_x=single(zeros(Ny,Nx,Nz));
hsb_y=single(zeros(Ny,Nx,Nz));
t1=single(zeros(Ny,Nx,Nz));
t2=single(zeros(Ny,Nx,Nz));
t3=single(zeros(Ny,Nx,Nz));
t4=single(zeros(Ny,Nx,Nz));
t7=single(zeros(Ny,Nx,Nz));
t8=single(zeros(Ny,Nx,Nz));
% hsb_z=single(zeros(Ny,Nx,Nz));
% hsb_elevz=single(zeros(Ny,Nx,Nz));
for i=1:r
%     hsb_x(:,:,i,1)=h_x(:,:,i);hsb_x(:,:,i,2)=s(:,:,i);hsb_x(:,:,i,3)=b(:,:,i);rgb_x(:,:,i,:)=hsv2rgb(squeeze(hsb_x(:,:,i,:)));
    hsb_y(:,:,i,1)=h_y(:,:,i);hsb_y(:,:,i,2)=s(:,:,i);hsb_y(:,:,i,3)=b(:,:,i);rgb_y(:,:,i,:)=hsv2rgb(squeeze(hsb_y(:,:,i,:)));
    t1(:,:,i,1)=MA_norm(:,:,i);t1(:,:,i,2)=onematrix(:,:,i);t1(:,:,i,3)=FAv(:,:,i);rgb_t1(:,:,i,:)=hsv2rgb(squeeze(t1(:,:,i,:)));
    t2(:,:,i,1)=h_y(:,:,i);t2(:,:,i,2)=s2(:,:,i);t2(:,:,i,3)=b(:,:,i);rgb_t2(:,:,i,:)=hsv2rgb(squeeze(t2(:,:,i,:)));
    t3(:,:,i,1)=h_y(:,:,i);t3(:,:,i,2)=s3(:,:,i);t3(:,:,i,3)=b(:,:,i);rgb_t3(:,:,i,:)=hsv2rgb(squeeze(t3(:,:,i,:)));
    t4(:,:,i,1)=h_y(:,:,i);t4(:,:,i,2)=FAv(:,:,i);t4(:,:,i,3)=b(:,:,i);rgb_t4(:,:,i,:)=hsv2rgb(squeeze(t4(:,:,i,:)));
    t7(:,:,i,1)=x(:,:,i).*FAv(:,:,i);t7(:,:,i,2)=y(:,:,i).*FAv(:,:,i);t7(:,:,i,3)=z(:,:,i).*FAv(:,:,i);
    t8(:,:,i,1)=x(:,:,i).*FAv(:,:,i).*MA_norm(:,:,i);t8(:,:,i,2)=y(:,:,i).*FAv(:,:,i).*MA_norm(:,:,i);t8(:,:,i,3)=z(:,:,i).*FAv(:,:,i).*MA_norm(:,:,i);
%     hsb_z(:,:,i,1)=h_z(:,:,i);hsb_z(:,:,i,2)=s(:,:,i);hsb_z(:,:,i,3)=b(:,:,i);rgb_z(:,:,i,:)=hsv2rgb(squeeze(hsb_z(:,:,i,:)));   
%     hsb_elevz(:,:,i,1)=h_elevz(:,:,i);hsb_elevz(:,:,i,2)=s(:,:,i);hsb_elevz(:,:,i,3)=b(:,:,i);rgb_elevz(:,:,i,:)=hsv2rgb(squeeze(hsb_elevz(:,:,i,:)));   
end
toc;

tic;
disp('Saving the SA and Fiber orientation estimates');
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/Largeblock4Anisotropy/s1m3rho1k5/;
save('FA.mat','FAv','-v7.3');
save('MA.mat','MA','-v7.3');
save('cl.mat','cl','-v7.3');
save('cp.mat','cp','-v7.3');
save('cs.mat','cs','-v7.3');
save('clw.mat','clw','-v7.3');
save('cpw.mat','cpw','-v7.3');
save('t7.mat','cp','-v7.3');
save('t8.mat','cs','-v7.3');
% save('elevz.mat','elevz','-v7.3');
% save('rgbx_s1m3rho5k25.mat','rgb_x','-v7.3');
% save('rgby_s1m3rho5k25.mat','rgb_y','-v7.3');
% save('rgbz_s1m3rho5k25.mat','rgb_z','-v7.3');
% save('rgb_elevz_s1m3rho5k25.mat','rgb_elevz','-v7.3');
toc;

tic;
% writing the output images
%##########################
% clear;clc;
% cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Large_block_25um_filtered/ST3D;
% rgb=importdata('rgb_thetay_orient_Large_Sig3kernel9_rho1_04Dec17.mat');
[~,~,r,~]=size(rgb_y);
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/Largeblock4Anisotropy/s1m3rho1k5/;
for i=1:r
%     plane_x=squeeze(rgb_x(:,:,i,:));namex=sprintf('x_%04d.png',i-1);cd rgbx;imwrite(plane_x,namex);cd ..;
    plane_y=squeeze(rgb_y(:,:,i,:));namey=sprintf('y_%04d.png',i-1);cd rgby;imwrite(plane_y,namey);cd ..;
%     plane_z=squeeze(rgb_z(:,:,i,:));namez=sprintf('z_%04d.png',i-1);cd rgbz;imwrite(plane_z,namez);cd ..;
%     plane_elevz=squeeze(rgb_elevz(:,:,i,:));name_elevz=sprintf('elevz_%04d.png',i-1);cd rgb_elevz;imwrite(plane_elevz,name_elevz);cd ..;  
plane_t1=squeeze(rgb_t1(:,:,i,:));namet1=sprintf('t1_%04d.png',i-1);cd t1;imwrite(plane_t1,namet1);cd ..;
plane_t2=squeeze(rgb_t2(:,:,i,:));namet2=sprintf('t2_%04d.png',i-1);cd t2;imwrite(plane_t2,namet2);cd ..;
plane_t3=squeeze(rgb_t3(:,:,i,:));namet3=sprintf('t3_%04d.png',i-1);cd t3;imwrite(plane_t3,namet3);cd ..;
plane_t4=squeeze(rgb_t4(:,:,i,:));namet4=sprintf('t4_%04d.png',i-1);cd t4;imwrite(plane_t4,namet4);cd ..;
end
toc;
disp('Completed successfully....exiting rho1');


clear;
% cd V:\ABI\JZ\Fiber_DTI\Large_block_25um_filtered\Zplanes\grey\;
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Large_block_25um_filtered/Zplanes/grey;
len=200;  %change this for new image
for i=1:len
infile=sprintf('z_%04d.tif',i-1);                      
im=imread(infile);
Df(:,:,len-i+1)=im;

end
% cd V:\ABI\pacedSheep01\Anisotropic\functions;
% showcs3(single(Df));
V1=Df;
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
rho =3;
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

tic;
disp('Computed EV and VectorF...Saving now');
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/Largeblock4Anisotropy/s1m3rho3k15/;
save('EV_s1m3rho3k15.mat','EV','-v7.3');
disp('saved EV...moving to VectorF');
save('VectorF_s1m3rho3k15.mat','VectorF','-v7.3');
toc;

tic;
disp('Computing Structural Anisotropy parameters and Fiber orientation estimates');
o1=squeeze(EV(:,:,:,1));o2=squeeze(EV(:,:,:,2));o3=squeeze(EV(:,:,:,3));
l3=o1;l1=o3;l2=o1+o3-o2;   % ST Eigen value transformation
FAv=(1/sqrt(2)).*( sqrt((l1-l2).^2+(l2-l3).^2+(l3-l1).^2)./sqrt(l1.^2+l2.^2+l3.^2) );
cl=(l1-l2)./(l1+l2+l3);cp=2.*(l2-l3)./(l1+l2+l3);cs=3*l3./(l1+l2+l3);
clw=(l1-l2)./l1;cpw=(l2-l3)./l1;csw=l3./l1;
MA=((0.5*(-l1-l2+2*l3).*(2*l1-l2-l3).*(-l1+2.*l2-l3))./((l1.^2+l2.^2+l3.^2-l1.*l2-l2.*l3-l3.*l1).^1.5));


y=squeeze(VectorF(:,:,:,1));x=squeeze(VectorF(:,:,:,2));z=squeeze(VectorF(:,:,:,3));
projy=(atan2(-z,x)).*(180/pi);
% neg_iprojz=(atan2(-y,x)).*(180/pi);
% projx=(atan2(-z,y)).*(180/pi);
% elevz=(atan2(-z,sqrt(x.^2+y.^2))).*(180/pi);
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

% [p,q,r]=size(neg_iprojz);
% for i= 1:p
%     for j=1:q
%        for k=1:r 
%         theta_element=neg_iprojz(i,j,k);
%         
%         if theta_element >90
%             temp =theta_element-180;
%         elseif theta_element <-90
%             temp = theta_element+180;
%         else 
%             temp=theta_element;
%         end
%         
%         neg_ithetaz(i,j,k)=temp;
%        end
%     end
% end
% clear temp;

% [p,q,r]=size(projx);
% for i= 1:p
%     for j=1:q
%        for k=1:r 
%         theta_element=projx(i,j,k);
%         
%         if theta_element >90
%             temp =theta_element-180;
%         elseif theta_element <-90
%             temp = theta_element+180;
%         else 
%             temp=theta_element;
%         end
%         
%         thetax(i,j,k)=temp;
%        end
%     end
% end
% clear temp;

% Making the hsv model of orientation 
% ###################################
% h=orientation angle, 
% s=1(can be replaced by coherence for better results), 
% v=inverted red channel of original(can be the inverted greyscale version also)
% The hsv model is converted to rgb for visualisation purpose(In order to
% match with the visualisation from 2D orientation)
% clear h;clear s;clear b;clear hsb;clear rgb;
% h_x=(thetax+90)./180; % converting from (-90,90) to (0,1)
h_y=(thetay+90)./180; % converting from (-90,90) to (0,1)
% h_z=(neg_ithetaz+90)./180; % converting from (-90,90) to (0,1)
% h_elevz=(elevz+90)./180;
s=coh;%ones(560,1000,200);
MA_norm=(MA+1)./2;
s2=FAv.*MA_norm;
onematrix=ones(Ny,Nx,Nz);
s3=onematrix-s2;
b=(InvV1./255);
% hsb_x=single(zeros(Ny,Nx,Nz));
hsb_y=single(zeros(Ny,Nx,Nz));
t1=single(zeros(Ny,Nx,Nz));
t2=single(zeros(Ny,Nx,Nz));
t3=single(zeros(Ny,Nx,Nz));
t4=single(zeros(Ny,Nx,Nz));
t7=single(zeros(Ny,Nx,Nz));
t8=single(zeros(Ny,Nx,Nz));
% hsb_z=single(zeros(Ny,Nx,Nz));
% hsb_elevz=single(zeros(Ny,Nx,Nz));
for i=1:r
%     hsb_x(:,:,i,1)=h_x(:,:,i);hsb_x(:,:,i,2)=s(:,:,i);hsb_x(:,:,i,3)=b(:,:,i);rgb_x(:,:,i,:)=hsv2rgb(squeeze(hsb_x(:,:,i,:)));
    hsb_y(:,:,i,1)=h_y(:,:,i);hsb_y(:,:,i,2)=s(:,:,i);hsb_y(:,:,i,3)=b(:,:,i);rgb_y(:,:,i,:)=hsv2rgb(squeeze(hsb_y(:,:,i,:)));
    t1(:,:,i,1)=MA_norm(:,:,i);t1(:,:,i,2)=onematrix(:,:,i);t1(:,:,i,3)=FAv(:,:,i);rgb_t1(:,:,i,:)=hsv2rgb(squeeze(t1(:,:,i,:)));
    t2(:,:,i,1)=h_y(:,:,i);t2(:,:,i,2)=s2(:,:,i);t2(:,:,i,3)=b(:,:,i);rgb_t2(:,:,i,:)=hsv2rgb(squeeze(t2(:,:,i,:)));
    t3(:,:,i,1)=h_y(:,:,i);t3(:,:,i,2)=s3(:,:,i);t3(:,:,i,3)=b(:,:,i);rgb_t3(:,:,i,:)=hsv2rgb(squeeze(t3(:,:,i,:)));
    t4(:,:,i,1)=h_y(:,:,i);t4(:,:,i,2)=FAv(:,:,i);t4(:,:,i,3)=b(:,:,i);rgb_t4(:,:,i,:)=hsv2rgb(squeeze(t4(:,:,i,:)));
    t7(:,:,i,1)=x(:,:,i).*FAv(:,:,i);t7(:,:,i,2)=y(:,:,i).*FAv(:,:,i);t7(:,:,i,3)=z(:,:,i).*FAv(:,:,i);
    t8(:,:,i,1)=x(:,:,i).*FAv(:,:,i).*MA_norm(:,:,i);t8(:,:,i,2)=y(:,:,i).*FAv(:,:,i).*MA_norm(:,:,i);t8(:,:,i,3)=z(:,:,i).*FAv(:,:,i).*MA_norm(:,:,i);
%     hsb_z(:,:,i,1)=h_z(:,:,i);hsb_z(:,:,i,2)=s(:,:,i);hsb_z(:,:,i,3)=b(:,:,i);rgb_z(:,:,i,:)=hsv2rgb(squeeze(hsb_z(:,:,i,:)));   
%     hsb_elevz(:,:,i,1)=h_elevz(:,:,i);hsb_elevz(:,:,i,2)=s(:,:,i);hsb_elevz(:,:,i,3)=b(:,:,i);rgb_elevz(:,:,i,:)=hsv2rgb(squeeze(hsb_elevz(:,:,i,:)));   
end
toc;

tic;
disp('Saving the SA and Fiber orientation estimates');
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/Largeblock4Anisotropy/s1m3rho3k15/;
save('FA.mat','FAv','-v7.3');
save('MA.mat','MA','-v7.3');
save('cl.mat','cl','-v7.3');
save('cp.mat','cp','-v7.3');
save('clw.mat','clw','-v7.3');
save('cpw.mat','cpw','-v7.3');
save('cs.mat','cs','-v7.3');
save('t7.mat','cp','-v7.3');
save('t8.mat','cs','-v7.3');
% save('elevz.mat','elevz','-v7.3');
% save('rgbx_s1m3rho5k25.mat','rgb_x','-v7.3');
% save('rgby_s1m3rho5k25.mat','rgb_y','-v7.3');
% save('rgbz_s1m3rho5k25.mat','rgb_z','-v7.3');
% save('rgb_elevz_s1m3rho5k25.mat','rgb_elevz','-v7.3');
toc;

tic;
% writing the output images
%##########################
% clear;clc;
% cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Large_block_25um_filtered/ST3D;
% rgb=importdata('rgb_thetay_orient_Large_Sig3kernel9_rho1_04Dec17.mat');
[~,~,r,~]=size(rgb_y);
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/Largeblock4Anisotropy/s1m3rho3k15/;
for i=1:r
%     plane_x=squeeze(rgb_x(:,:,i,:));namex=sprintf('x_%04d.png',i-1);cd rgbx;imwrite(plane_x,namex);cd ..;
    plane_y=squeeze(rgb_y(:,:,i,:));namey=sprintf('y_%04d.png',i-1);cd rgby;imwrite(plane_y,namey);cd ..;
%     plane_z=squeeze(rgb_z(:,:,i,:));namez=sprintf('z_%04d.png',i-1);cd rgbz;imwrite(plane_z,namez);cd ..;
%     plane_elevz=squeeze(rgb_elevz(:,:,i,:));name_elevz=sprintf('elevz_%04d.png',i-1);cd rgb_elevz;imwrite(plane_elevz,name_elevz);cd ..;  
plane_t1=squeeze(rgb_t1(:,:,i,:));namet1=sprintf('t1_%04d.png',i-1);cd t1;imwrite(plane_t1,namet1);cd ..;
plane_t2=squeeze(rgb_t2(:,:,i,:));namet2=sprintf('t2_%04d.png',i-1);cd t2;imwrite(plane_t2,namet2);cd ..;
plane_t3=squeeze(rgb_t3(:,:,i,:));namet3=sprintf('t3_%04d.png',i-1);cd t3;imwrite(plane_t3,namet3);cd ..;
plane_t4=squeeze(rgb_t4(:,:,i,:));namet4=sprintf('t4_%04d.png',i-1);cd t4;imwrite(plane_t4,namet4);cd ..;
end
toc;
disp('Completed successfully....exiting rho3');
disp('*********************************************************************');
disp('*********************************************************************');
disp('*********************************************************************');
disp('*********************************************************************');

clear;
% cd V:\ABI\JZ\Fiber_DTI\Large_block_25um_filtered\Zplanes\grey\;
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Large_block_25um_filtered/Zplanes/grey;
len=200;  %change this for new image
for i=1:len
infile=sprintf('z_%04d.tif',i-1);                      
im=imread(infile);
Df(:,:,len-i+1)=im;

end
% cd V:\ABI\pacedSheep01\Anisotropic\functions;
% showcs3(single(Df));
V1=Df;
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
rho =5;
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

tic;
disp('Computed EV and VectorF...Saving now');
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/Largeblock4Anisotropy/s1m3rho5k25/;
save('EV_s1m3rho5k25.mat','EV','-v7.3');
disp('saved EV...moving to VectorF');
save('VectorF_s1m3rho5k25.mat','VectorF','-v7.3');
toc;

tic;
disp('Computing Structural Anisotropy parameters and Fiber orientation estimates');
o1=squeeze(EV(:,:,:,1));o2=squeeze(EV(:,:,:,2));o3=squeeze(EV(:,:,:,3));
l3=o1;l1=o3;l2=o1+o3-o2;   % ST Eigen value transformation
FAv=(1/sqrt(2)).*( sqrt((l1-l2).^2+(l2-l3).^2+(l3-l1).^2)./sqrt(l1.^2+l2.^2+l3.^2) );
cl=(l1-l2)./(l1+l2+l3);cp=2.*(l2-l3)./(l1+l2+l3);cs=3*l3./(l1+l2+l3);
clw=(l1-l2)./l1;cpw=(l2-l3)./l1;csw=l3./l1;
MA=((0.5*(-l1-l2+2*l3).*(2*l1-l2-l3).*(-l1+2.*l2-l3))./((l1.^2+l2.^2+l3.^2-l1.*l2-l2.*l3-l3.*l1).^1.5));


y=squeeze(VectorF(:,:,:,1));x=squeeze(VectorF(:,:,:,2));z=squeeze(VectorF(:,:,:,3));
projy=(atan2(-z,x)).*(180/pi);
% neg_iprojz=(atan2(-y,x)).*(180/pi);
% projx=(atan2(-z,y)).*(180/pi);
% elevz=(atan2(-z,sqrt(x.^2+y.^2))).*(180/pi);
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

% [p,q,r]=size(neg_iprojz);
% for i= 1:p
%     for j=1:q
%        for k=1:r 
%         theta_element=neg_iprojz(i,j,k);
%         
%         if theta_element >90
%             temp =theta_element-180;
%         elseif theta_element <-90
%             temp = theta_element+180;
%         else 
%             temp=theta_element;
%         end
%         
%         neg_ithetaz(i,j,k)=temp;
%        end
%     end
% end
% clear temp;

% [p,q,r]=size(projx);
% for i= 1:p
%     for j=1:q
%        for k=1:r 
%         theta_element=projx(i,j,k);
%         
%         if theta_element >90
%             temp =theta_element-180;
%         elseif theta_element <-90
%             temp = theta_element+180;
%         else 
%             temp=theta_element;
%         end
%         
%         thetax(i,j,k)=temp;
%        end
%     end
% end
% clear temp;

% Making the hsv model of orientation 
% ###################################
% h=orientation angle, 
% s=1(can be replaced by coherence for better results), 
% v=inverted red channel of original(can be the inverted greyscale version also)
% The hsv model is converted to rgb for visualisation purpose(In order to
% match with the visualisation from 2D orientation)
% clear h;clear s;clear b;clear hsb;clear rgb;
% h_x=(thetax+90)./180; % converting from (-90,90) to (0,1)
h_y=(thetay+90)./180; % converting from (-90,90) to (0,1)
% h_z=(neg_ithetaz+90)./180; % converting from (-90,90) to (0,1)
% h_elevz=(elevz+90)./180;
s=coh;%ones(560,1000,200);
MA_norm=(MA+1)./2;
s2=FAv.*MA_norm;
onematrix=ones(Ny,Nx,Nz);
s3=onematrix-s2;
b=(InvV1./255);
% hsb_x=single(zeros(Ny,Nx,Nz));
hsb_y=single(zeros(Ny,Nx,Nz));
t1=single(zeros(Ny,Nx,Nz));
t2=single(zeros(Ny,Nx,Nz));
t3=single(zeros(Ny,Nx,Nz));
t4=single(zeros(Ny,Nx,Nz));
t7=single(zeros(Ny,Nx,Nz));
t8=single(zeros(Ny,Nx,Nz));
% hsb_z=single(zeros(Ny,Nx,Nz));
% hsb_elevz=single(zeros(Ny,Nx,Nz));
for i=1:r
%     hsb_x(:,:,i,1)=h_x(:,:,i);hsb_x(:,:,i,2)=s(:,:,i);hsb_x(:,:,i,3)=b(:,:,i);rgb_x(:,:,i,:)=hsv2rgb(squeeze(hsb_x(:,:,i,:)));
    hsb_y(:,:,i,1)=h_y(:,:,i);hsb_y(:,:,i,2)=s(:,:,i);hsb_y(:,:,i,3)=b(:,:,i);rgb_y(:,:,i,:)=hsv2rgb(squeeze(hsb_y(:,:,i,:)));
    t1(:,:,i,1)=MA_norm(:,:,i);t1(:,:,i,2)=onematrix(:,:,i);t1(:,:,i,3)=FAv(:,:,i);rgb_t1(:,:,i,:)=hsv2rgb(squeeze(t1(:,:,i,:)));
    t2(:,:,i,1)=h_y(:,:,i);t2(:,:,i,2)=s2(:,:,i);t2(:,:,i,3)=b(:,:,i);rgb_t2(:,:,i,:)=hsv2rgb(squeeze(t2(:,:,i,:)));
    t3(:,:,i,1)=h_y(:,:,i);t3(:,:,i,2)=s3(:,:,i);t3(:,:,i,3)=b(:,:,i);rgb_t3(:,:,i,:)=hsv2rgb(squeeze(t3(:,:,i,:)));
    t4(:,:,i,1)=h_y(:,:,i);t4(:,:,i,2)=FAv(:,:,i);t4(:,:,i,3)=b(:,:,i);rgb_t4(:,:,i,:)=hsv2rgb(squeeze(t4(:,:,i,:)));
    t7(:,:,i,1)=x(:,:,i).*FAv(:,:,i);t7(:,:,i,2)=y(:,:,i).*FAv(:,:,i);t7(:,:,i,3)=z(:,:,i).*FAv(:,:,i);
    t8(:,:,i,1)=x(:,:,i).*FAv(:,:,i).*MA_norm(:,:,i);t8(:,:,i,2)=y(:,:,i).*FAv(:,:,i).*MA_norm(:,:,i);t8(:,:,i,3)=z(:,:,i).*FAv(:,:,i).*MA_norm(:,:,i);
%     hsb_z(:,:,i,1)=h_z(:,:,i);hsb_z(:,:,i,2)=s(:,:,i);hsb_z(:,:,i,3)=b(:,:,i);rgb_z(:,:,i,:)=hsv2rgb(squeeze(hsb_z(:,:,i,:)));   
%     hsb_elevz(:,:,i,1)=h_elevz(:,:,i);hsb_elevz(:,:,i,2)=s(:,:,i);hsb_elevz(:,:,i,3)=b(:,:,i);rgb_elevz(:,:,i,:)=hsv2rgb(squeeze(hsb_elevz(:,:,i,:)));   
end
toc;

tic;
disp('Saving the SA and Fiber orientation estimates');
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/Largeblock4Anisotropy/s1m3rho5k25/;
save('FA.mat','FAv','-v7.3');
save('MA.mat','MA','-v7.3');
save('cl.mat','cl','-v7.3');
save('cp.mat','cp','-v7.3');
save('clw.mat','clw','-v7.3');
save('cpw.mat','cpw','-v7.3');
save('cs.mat','cs','-v7.3');
save('t7.mat','cp','-v7.3');
save('t8.mat','cs','-v7.3');
% save('elevz.mat','elevz','-v7.3');
% save('rgbx_s1m3rho5k25.mat','rgb_x','-v7.3');
% save('rgby_s1m3rho5k25.mat','rgb_y','-v7.3');
% save('rgbz_s1m3rho5k25.mat','rgb_z','-v7.3');
% save('rgb_elevz_s1m3rho5k25.mat','rgb_elevz','-v7.3');
toc;

tic;
% writing the output images
%##########################
% clear;clc;
% cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Large_block_25um_filtered/ST3D;
% rgb=importdata('rgb_thetay_orient_Large_Sig3kernel9_rho1_04Dec17.mat');
[~,~,r,~]=size(rgb_y);
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/Largeblock4Anisotropy/s1m3rho5k25/;
for i=1:r
%     plane_x=squeeze(rgb_x(:,:,i,:));namex=sprintf('x_%04d.png',i-1);cd rgbx;imwrite(plane_x,namex);cd ..;
    plane_y=squeeze(rgb_y(:,:,i,:));namey=sprintf('y_%04d.png',i-1);cd rgby;imwrite(plane_y,namey);cd ..;
%     plane_z=squeeze(rgb_z(:,:,i,:));namez=sprintf('z_%04d.png',i-1);cd rgbz;imwrite(plane_z,namez);cd ..;
%     plane_elevz=squeeze(rgb_elevz(:,:,i,:));name_elevz=sprintf('elevz_%04d.png',i-1);cd rgb_elevz;imwrite(plane_elevz,name_elevz);cd ..;  
plane_t1=squeeze(rgb_t1(:,:,i,:));namet1=sprintf('t1_%04d.png',i-1);cd t1;imwrite(plane_t1,namet1);cd ..;
plane_t2=squeeze(rgb_t2(:,:,i,:));namet2=sprintf('t2_%04d.png',i-1);cd t2;imwrite(plane_t2,namet2);cd ..;
plane_t3=squeeze(rgb_t3(:,:,i,:));namet3=sprintf('t3_%04d.png',i-1);cd t3;imwrite(plane_t3,namet3);cd ..;
plane_t4=squeeze(rgb_t4(:,:,i,:));namet4=sprintf('t4_%04d.png',i-1);cd t4;imwrite(plane_t4,namet4);cd ..;
end
toc;
disp('Completed successfully....exiting rho5');
disp('*********************************************************************');
disp('*********************************************************************');
disp('*********************************************************************');
disp('*********************************************************************');

clear;
% cd V:\ABI\JZ\Fiber_DTI\Large_block_25um_filtered\Zplanes\grey\;
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Large_block_25um_filtered/Zplanes/grey;
len=200;  %change this for new image
for i=1:len
infile=sprintf('z_%04d.tif',i-1);                      
im=imread(infile);
Df(:,:,len-i+1)=im;

end
% cd V:\ABI\pacedSheep01\Anisotropic\functions;
% showcs3(single(Df));
V1=Df;
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
rho =10;
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

tic;
disp('Computed EV and VectorF...Saving now');
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/Largeblock4Anisotropy/s1m3rho10k50/;
save('EV_s1m3rho10k50.mat','EV','-v7.3');
disp('saved EV...moving to VectorF');
save('VectorF_s1m3rho10k50.mat','VectorF','-v7.3');
toc;

tic;
disp('Computing Structural Anisotropy parameters and Fiber orientation estimates');
o1=squeeze(EV(:,:,:,1));o2=squeeze(EV(:,:,:,2));o3=squeeze(EV(:,:,:,3));
l3=o1;l1=o3;l2=o1+o3-o2;   % ST Eigen value transformation
FAv=(1/sqrt(2)).*( sqrt((l1-l2).^2+(l2-l3).^2+(l3-l1).^2)./sqrt(l1.^2+l2.^2+l3.^2) );
cl=(l1-l2)./(l1+l2+l3);cp=2.*(l2-l3)./(l1+l2+l3);cs=3*l3./(l1+l2+l3);
clw=(l1-l2)./l1;cpw=(l2-l3)./l1;csw=l3./l1;
MA=((0.5*(-l1-l2+2*l3).*(2*l1-l2-l3).*(-l1+2.*l2-l3))./((l1.^2+l2.^2+l3.^2-l1.*l2-l2.*l3-l3.*l1).^1.5));


y=squeeze(VectorF(:,:,:,1));x=squeeze(VectorF(:,:,:,2));z=squeeze(VectorF(:,:,:,3));
projy=(atan2(-z,x)).*(180/pi);
% neg_iprojz=(atan2(-y,x)).*(180/pi);
% projx=(atan2(-z,y)).*(180/pi);
% elevz=(atan2(-z,sqrt(x.^2+y.^2))).*(180/pi);
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

% [p,q,r]=size(neg_iprojz);
% for i= 1:p
%     for j=1:q
%        for k=1:r 
%         theta_element=neg_iprojz(i,j,k);
%         
%         if theta_element >90
%             temp =theta_element-180;
%         elseif theta_element <-90
%             temp = theta_element+180;
%         else 
%             temp=theta_element;
%         end
%         
%         neg_ithetaz(i,j,k)=temp;
%        end
%     end
% end
% clear temp;

% [p,q,r]=size(projx);
% for i= 1:p
%     for j=1:q
%        for k=1:r 
%         theta_element=projx(i,j,k);
%         
%         if theta_element >90
%             temp =theta_element-180;
%         elseif theta_element <-90
%             temp = theta_element+180;
%         else 
%             temp=theta_element;
%         end
%         
%         thetax(i,j,k)=temp;
%        end
%     end
% end
% clear temp;

% Making the hsv model of orientation 
% ###################################
% h=orientation angle, 
% s=1(can be replaced by coherence for better results), 
% v=inverted red channel of original(can be the inverted greyscale version also)
% The hsv model is converted to rgb for visualisation purpose(In order to
% match with the visualisation from 2D orientation)
% clear h;clear s;clear b;clear hsb;clear rgb;
% h_x=(thetax+90)./180; % converting from (-90,90) to (0,1)
h_y=(thetay+90)./180; % converting from (-90,90) to (0,1)
% h_z=(neg_ithetaz+90)./180; % converting from (-90,90) to (0,1)
% h_elevz=(elevz+90)./180;
s=coh;%ones(560,1000,200);
MA_norm=(MA+1)./2;
s2=FAv.*MA_norm;
onematrix=ones(Ny,Nx,Nz);
s3=onematrix-s2;
b=(InvV1./255);
% hsb_x=single(zeros(Ny,Nx,Nz));
hsb_y=single(zeros(Ny,Nx,Nz));
t1=single(zeros(Ny,Nx,Nz));
t2=single(zeros(Ny,Nx,Nz));
t3=single(zeros(Ny,Nx,Nz));
t4=single(zeros(Ny,Nx,Nz));
t7=single(zeros(Ny,Nx,Nz));
t8=single(zeros(Ny,Nx,Nz));
% hsb_z=single(zeros(Ny,Nx,Nz));
% hsb_elevz=single(zeros(Ny,Nx,Nz));
for i=1:r
%     hsb_x(:,:,i,1)=h_x(:,:,i);hsb_x(:,:,i,2)=s(:,:,i);hsb_x(:,:,i,3)=b(:,:,i);rgb_x(:,:,i,:)=hsv2rgb(squeeze(hsb_x(:,:,i,:)));
    hsb_y(:,:,i,1)=h_y(:,:,i);hsb_y(:,:,i,2)=s(:,:,i);hsb_y(:,:,i,3)=b(:,:,i);rgb_y(:,:,i,:)=hsv2rgb(squeeze(hsb_y(:,:,i,:)));
    t1(:,:,i,1)=MA_norm(:,:,i);t1(:,:,i,2)=onematrix(:,:,i);t1(:,:,i,3)=FAv(:,:,i);rgb_t1(:,:,i,:)=hsv2rgb(squeeze(t1(:,:,i,:)));
    t2(:,:,i,1)=h_y(:,:,i);t2(:,:,i,2)=s2(:,:,i);t2(:,:,i,3)=b(:,:,i);rgb_t2(:,:,i,:)=hsv2rgb(squeeze(t2(:,:,i,:)));
    t3(:,:,i,1)=h_y(:,:,i);t3(:,:,i,2)=s3(:,:,i);t3(:,:,i,3)=b(:,:,i);rgb_t3(:,:,i,:)=hsv2rgb(squeeze(t3(:,:,i,:)));
    t4(:,:,i,1)=h_y(:,:,i);t4(:,:,i,2)=FAv(:,:,i);t4(:,:,i,3)=b(:,:,i);rgb_t4(:,:,i,:)=hsv2rgb(squeeze(t4(:,:,i,:)));
    t7(:,:,i,1)=x(:,:,i).*FAv(:,:,i);t7(:,:,i,2)=y(:,:,i).*FAv(:,:,i);t7(:,:,i,3)=z(:,:,i).*FAv(:,:,i);
    t8(:,:,i,1)=x(:,:,i).*FAv(:,:,i).*MA_norm(:,:,i);t8(:,:,i,2)=y(:,:,i).*FAv(:,:,i).*MA_norm(:,:,i);t8(:,:,i,3)=z(:,:,i).*FAv(:,:,i).*MA_norm(:,:,i);
%     hsb_z(:,:,i,1)=h_z(:,:,i);hsb_z(:,:,i,2)=s(:,:,i);hsb_z(:,:,i,3)=b(:,:,i);rgb_z(:,:,i,:)=hsv2rgb(squeeze(hsb_z(:,:,i,:)));   
%     hsb_elevz(:,:,i,1)=h_elevz(:,:,i);hsb_elevz(:,:,i,2)=s(:,:,i);hsb_elevz(:,:,i,3)=b(:,:,i);rgb_elevz(:,:,i,:)=hsv2rgb(squeeze(hsb_elevz(:,:,i,:)));   
end
toc;

tic;
disp('Saving the SA and Fiber orientation estimates');
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/Largeblock4Anisotropy/s1m3rho10k50/;
save('FA.mat','FAv','-v7.3');
save('MA.mat','MA','-v7.3');
save('cl.mat','cl','-v7.3');
save('cp.mat','cp','-v7.3');
save('clw.mat','clw','-v7.3');
save('cpw.mat','cpw','-v7.3');
save('cs.mat','cs','-v7.3');
save('t7.mat','cp','-v7.3');
save('t8.mat','cs','-v7.3');
% save('elevz.mat','elevz','-v7.3');
% save('rgbx_s1m3rho5k25.mat','rgb_x','-v7.3');
% save('rgby_s1m3rho5k25.mat','rgb_y','-v7.3');
% save('rgbz_s1m3rho5k25.mat','rgb_z','-v7.3');
% save('rgb_elevz_s1m3rho5k25.mat','rgb_elevz','-v7.3');
toc;

tic;
% writing the output images
%##########################
% clear;clc;
% cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Large_block_25um_filtered/ST3D;
% rgb=importdata('rgb_thetay_orient_Large_Sig3kernel9_rho1_04Dec17.mat');
[~,~,r,~]=size(rgb_y);
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/Largeblock4Anisotropy/s1m3rho10k50/;
for i=1:r
%     plane_x=squeeze(rgb_x(:,:,i,:));namex=sprintf('x_%04d.png',i-1);cd rgbx;imwrite(plane_x,namex);cd ..;
    plane_y=squeeze(rgb_y(:,:,i,:));namey=sprintf('y_%04d.png',i-1);cd rgby;imwrite(plane_y,namey);cd ..;
%     plane_z=squeeze(rgb_z(:,:,i,:));namez=sprintf('z_%04d.png',i-1);cd rgbz;imwrite(plane_z,namez);cd ..;
%     plane_elevz=squeeze(rgb_elevz(:,:,i,:));name_elevz=sprintf('elevz_%04d.png',i-1);cd rgb_elevz;imwrite(plane_elevz,name_elevz);cd ..;  
plane_t1=squeeze(rgb_t1(:,:,i,:));namet1=sprintf('t1_%04d.png',i-1);cd t1;imwrite(plane_t1,namet1);cd ..;
plane_t2=squeeze(rgb_t2(:,:,i,:));namet2=sprintf('t2_%04d.png',i-1);cd t2;imwrite(plane_t2,namet2);cd ..;
plane_t3=squeeze(rgb_t3(:,:,i,:));namet3=sprintf('t3_%04d.png',i-1);cd t3;imwrite(plane_t3,namet3);cd ..;
plane_t4=squeeze(rgb_t4(:,:,i,:));namet4=sprintf('t4_%04d.png',i-1);cd t4;imwrite(plane_t4,namet4);cd ..;
end
toc;
disp('Completed successfully....exiting rho10');
disp('*********************************************************************');
disp('*********************************************************************');
disp('*********************************************************************');
disp('*********************************************************************');


exit;
