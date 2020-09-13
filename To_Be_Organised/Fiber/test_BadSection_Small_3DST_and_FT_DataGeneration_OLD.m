%% %% Badsection_Small_FAST version - With Saving only the final VV_FiberTracks

clear;close all;clc;
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Badsection_small/a101_zplanes_corrected;
len=70;  %change this for new image
for i=1:len
infile=sprintf('z_%04d.tif',i-1);%outfile=sprintf('cut_%05d.tif',i-500);                       %change this for new image
im=imread(infile);
D1(:,:,i)=squeeze(im(:,:,1));
end
[Ny, Nx, Nz] = size(D1)

% V=single(D1);
% cd /hpc/btho733/ABI/pacedSheep01/Anisotropic/;
% JR = CoherenceFilter(V,struct('T',20,'dt',2,'Scheme','R'));

sigma =1;

BBori =  single(zeros([Ny Nx Nz]));
for i = 1:1:Nz
    BBori(:,:,i) = 255-(squeeze(D1(end:-1:1,:,i))); % Changed from rgb3grey to only red-plane
end
% Interpolate BBori from 50um to 8 um resolution for commputer modeling

% k = 6.25/25
% d = size(BBori);
% [xi,yi,zi] = meshgrid(1:d(2),1:d(1),1:k:d(3));
% BBori = interp3(BBori,xi,yi,zi,'cubic');


% BBori(BBori>180)=180;
% interest region
% Roi0 = logical(zeros([Ny Nx Nz]));
% 
% Roi0(BBori>10) = 1;
% 
% V = zeros([Ny Nx Nz],'single');
% Roi = zeros([Ny Nx Nz]);
% for i = 1:1:Nz,
%     V(:,:,i) = (BBori(:,:,i));
%     Roi(:,:,i) = (Roi0(:,:,i));
% end
% 
% Roi = logical(Roi);
% clear Roi0;
% 
% Roi(1:1:1,:,:) = 0;
% Roi((Ny):1:Ny,:,:) = 0;
% Roi(:,1:1:1,:) = 0;
% Roi(:,(Nx):1:Nx,:) = 0;
% Roi(:,:,1:1:1) = 0;
% Roi(:,:,(Nz):1:Nz) = 0;

cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI;
usigma=imgaussian(BBori,sigma,3);

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

% Perform DTI calculation
[coh,EV,VectorF]=testStructureFiber3D1(BBori,Jxx, Jxy, Jxz, Jyy, Jyz, Jzz,parametersDTI);

% clear D1;
% 
% cd /hpc_atog/btho733/ABI/JZ/Fiber_2016/Work_2017/CentralSection50um3/raw_25um3/bgdCorrect;
% len=100;  %change this for new image
% for p=1:len
% infile=sprintf('cut_%04d.tif',p-1);%outfile=sprintf('cut_%05d.tif',i-500);                       %change this for new image
% im=imread(infile);
% D1(:,:,p)=squeeze(im(101:250,101:400,1)); % taking only the redplanes
% end


% InvD1 =  single(zeros([Ny Nx Nz]));
% for p = 1:Nz
%     InvD1(:,:,p) = 255-(squeeze(D1(:,:,p))); 
% end

% interest region
Roi0 = logical(zeros([Ny Nx Nz]));
Roi0(BBori>60) = 1;
Roi=logical(Roi0);
Roi(1:2,:,:) = 0;
Roi((Ny-1):1:Ny,:,:) = 0;
Roi(:,1:2,:) = 0;
Roi(:,(Nx-1):1:Nx,:) = 0;
Roi(:,:,1:2) = 0;
Roi(:,:,(Nz-1):1:Nz) = 0;

% Fiber Tracking Constants
%#########################
% cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Large_block_25um_filtered/ST3D;
% VectorF=importdata('VectorF_MidsmallCutLarge_Sig1kernel3_rho3_ker15_07Jan18.mat');
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI;
parametersFT=[];
parametersFT.FiberLengthMax=1000000000;
parametersFT.FiberLengthMin=20;
parametersFT.DeviationAngleMax=1;
parametersFT.Step=0.4;
parametersFT.FiberTrackingComputationTreshold=0.05;
parametersFT.Sampling=1;
parametersFT.textdisplay=true;

parametersFT.FAmin = 100;% was 100
parametersFT.FAmax = 255;
parameters=parametersFT;


% Perform fiber tracking
%####################### 
fibers=FT2(VectorF,Roi,parametersFT);

N = length(fibers);

MeasureAngles = zeros(N,1);
for i=1:1:N
    fiber=fibers{i};
    for j=1:1:(length(fiber)-1)
    x1 = fiber(round(j),1);
    y1 = fiber(round(j),2);
    z1 = fiber(round(j),3);
    x2 = fiber(round(j+1),1);
    y2 = fiber(round(j+1),2);
    z2 = fiber(round(j+1),3);
    x = x2 - x1;
    y = y2 - y1;
    z = z2 - z1;

%     theta=atan2(y,x);
%     if theta < 0
%         temp = (theta + pi );
%         if temp > pi/2
%             temp = pi-temp;
%         end
%       
%         MeasureAngles(i) =  temp /(pi/2)+  MeasureAngles(i);
%     else
%         temp = theta;
%         if temp > pi/2
%             temp=pi-temp;
%         end
%      
%         MeasureAngles(i) =  temp/(pi/2) +  MeasureAngles(i);
%     end
%     end
    
    
    
    projy(i) = atan2(-z,y)*180/pi;
    if projy(i) >90
        temp =projy(i)-180;
    elseif projy(i) <-90
        temp = projy(i)+180;
    else
        temp=projy(i);
    end
        
    MeasureAng(i)=temp;
%     if (abs(phi) >= pi/4)
%         MeasureAngles(i) = (abs(phi) - pi/4)/(pi/4)*0.5+0.5 + MeasureAngles(i);
%     else
%         MeasureAngles(i) = abs(phi)/(pi/4)*0.5 + MeasureAngles(i);
%     end
   end


MeasureAngles(i)=uint8((255*(MeasureAng(i)+90))/180);
%     MeasureAngles(i) = uint8(255*MeasureAngles(i)/(length(fiber)-1));
    fibres{i}=uint16(fiber);
end

% clear fibers; 
clear x ; clear y,clear N;
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Large_block_25um_filtered/;
disp('saving in progress....');
save('VV_FT_BadSectionSmall_sig1_rho5_k15_fiber20_12Jan18_negzy_BSS_yreversed.mat','fibres','MeasureAngles','-v7.3');
%% BadSection_small  : Computing Angles and creating HSV model

y=squeeze(VectorF(:,:,:,1));x=squeeze(VectorF(:,:,:,2));z=squeeze(VectorF(:,:,:,3));
projy=(atan2(-z,x)).*(180/pi);
neg_iprojz=(atan2(y,x)).*(180/pi);

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


% Making original redplane inverted for B plane of HSB
% ####################################################
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Badsection_small;
len=70;  %change this for new image
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
h_y=(thetay+90)./180; % converting from (-90,90) to (0,1)
h_z=(neg_ithetaz+90)./180; % converting from (-90,90) to (0,1)
s=coh;%ones(560,1000,200);
b=(orig./255);
hsb=single(zeros(Ny,Nx,Nz));
for i=1:r
    hsb_y(:,:,i,1)=h_y(:,:,i);
    hsb_z(:,:,i,1)=h_z(:,:,i);
    hsb_y(:,:,i,2)=s(:,:,i);
    hsb_z(:,:,i,2)=s(:,:,i);
    hsb_y(:,:,i,3)=b(:,:,i);
    hsb_z(:,:,i,3)=b(:,:,i);
    rgb_y(:,:,i,:)=hsv2rgb(squeeze(hsb_y(:,:,i,:)));
    rgb_z(:,:,i,:)=hsv2rgb(squeeze(hsb_z(:,:,i,:)));
end
% save('rgb_neg_ithetaz_orient_Large_Sig1kernel3_rho10_ker50_07Dec17.mat','rgb');
%% writing the output images
% clear;clc;
% cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Large_block_25um_filtered/ST3D;
% rgb=importdata('rgb_thetay_orient_Large_Sig3kernel9_rho1_04Dec17.mat');
[~,~,r,~]=size(rgb_y);
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Badsection_small/a301_ST3D_results;
for i=1:r
    plane_y=squeeze(rgb_y(:,:,i,:));
    plane_z=squeeze(rgb_z(:,:,i,:));
    namey=sprintf('y_%04d.tif',i-1);
    namez=sprintf('z_%04d.tif',i-1);
    cd y;imwrite(plane_y,namey,'tif');cd ..;
    cd z;imwrite(plane_z,namez,'tif');cd ..;
end