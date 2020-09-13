%% use the default yxz axis instead of rotating around 
clear;close all;clc;
% cd /hpc/btho733/ABI/pacedSheep01/medsci_poster/stain_normalisation_toolbox/pacedsheepimages/f2v1/synth_op2/;  %change this for new image
% cd V:\ABI\pacedSheep01\medsci_poster\stain_normalisation_toolbox\pacedsheepimages\v8\output_RH\Nothreshold; 
% cd V:\ABI\JZ\Fiber_2016\Work_2017\CentralSection50um3\cut_for_elevation\bgdCorrect;
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Large_block_25um_filtered/New/Yplanes/;
% cd /hpc_atog/btho733/ABI/JZ/Fiber_2016/Work_2017/CentralSection50um3/raw_25um3/bgdCorrect;
len=560;  %change this for new image
for i=1:len
infile=sprintf('y_%04d.tif',i-1);%outfile=sprintf('cut_%05d.tif',i-500);                       %change this for new image
im=imread(infile);
% im=im(y1:y2,x1:x2,:);
D1(:,:,i)=im;

end

[Ny, Nx, Nz,k] = size(D1)
%load PVblock/LowResSegedPVori.mat;
%load PVblock/SmoothedLowResPVnewori.mat
%load PVblock/XYaxisSmoothedLowResSegedPVori2.mat;
%load PVblock/XYaxis6Smoothed2LowResSegedPVori.mat;
%%
V3=D1;
sigma = 0.5;
[Ny, Nx, Nz, k] = size(V3)
BBori =  single(zeros([Ny Nx Nz]));
for i = 1:1:Nz
    BBori(:,:,i) = 255-(squeeze(V3(:,:,i,1))); % Changed from rgb3grey to only red-plane
end
% Interpolate BBori from 50um to 8 um resolution for commputer modeling

% k = 6.25/25
% d = size(BBori);
% [xi,yi,zi] = meshgrid(1:d(2),1:d(1),1:k:d(3));
% BBori = interp3(BBori,xi,yi,zi,'cubic');

[Ny, Nx, Nz] = size(BBori)

% interest region
Roi0 = logical(zeros([Ny Nx Nz]));

Roi0(BBori>10) = 1;

V = zeros([Ny Nx Nz],'single');
Roi = zeros([Ny Nx Nz]);
for i = 1:1:Nz,
    V(:,:,i) = (BBori(:,:,i));
    Roi(:,:,i) = (Roi0(:,:,i));
end

Roi = logical(Roi);
clear Roi0;

Roi(1:1:1,:,:) = 0;
Roi((Ny):1:Ny,:,:) = 0;
Roi(:,1:1:1,:) = 0;
Roi(:,(Nx):1:Nx,:) = 0;
Roi(:,:,1:1:1) = 0;
Roi(:,:,(Nz):1:Nz) = 0;

cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI;
usigma=imgaussian(V,sigma,5*sigma);

% Calculate the gradients
ux=derivatives(usigma,'x');
uy=derivatives(usigma,'y');
uz=derivatives(usigma,'z');
% [ux,uy,uz] = imgradientxyz(usigma,'sobel');
% Compute the 3D structure tensors J of the image
rho = 2;
[Jxx, Jxy, Jxz, Jyy, Jyz, Jzz] = StructureTensor3D(ux,uy,uz,rho);
%[Jxx, Jxy, Jxz, Jyy, Jyz, Jzz] = AveStructureTensor3D(V,ux,uy,uz,rho);

disp('Done Constructing tensor');

parametersDTI=[];
parametersDTI.BackgroundTreshold=10;
parametersDTI.WhiteMatterExtractionThreshold=0.10;
parametersDTI.textdisplay=true;

% Perform DTI calculation
VectorF=testStructureFiber3D1(BBori,Jxx, Jxy, Jxz, Jyy, Jyz, Jzz,parametersDTI);
% save('v8_vectorF.mat','VectorF');
disp('Done FA');

needaveangles = 0;
steps = 5;
values = 5;
if needaveangles == 1
   V2 = AveFiberAngles3D(V,VectorF,steps,values);
end

% Save the resulting data for the FT_test.m script.
%VectorF = V2;
% save('PVblock/HighResPVsmallblocksmoothedVF2','VectorF');
% Read the Roi, through which all fibers must go (corpus callosum)
%info = gipl_read_header('corpus_callosum.gipl');
%Roi = gipl_read_volume(info)>0;

% Fiber Tracking Constants
%load('Atria_BB','FA','VectorF');
parametersFT=[];
parametersFT.FiberLengthMax=1000;
parametersFT.FiberLengthMin=20;
parametersFT.DeviationAngleMax=1;
parametersFT.Step=0.4;
parametersFT.FiberTrackingComputationTreshold=0.05;
parametersFT.Sampling=1;
parametersFT.textdisplay=true;

parametersFT.FAmin = 10;
parametersFT.FAmax = 255;

% Perform fiber tracking
fibers=FT2(VectorF,Roi,parametersFT);
% 
% % Show FA
% save -v7.3 Large_block_fibers.mat;


% cd /hpc/btho733/ABI/JZ/Fiber_2016/v8_f53_comparison_Matlab_Vs_VV/Matlab;
% cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI;
% clear;clc;close all;
% load cut4elevation_fibers.mat;
% % jz=importdata('v8_f53_fiber30.mat');
% disp('done loading.....');
N = length(fibers);

MeasureAngles = zeros(N,1);
MeasureAngles_fib=zeros(N,1);
phi_all=zeros(N,1);
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
    
    
    
    phi = atan2(z,sqrt(x.^2 + y.^2));
    
%     if (abs(phi) >= pi/4)
%         MeasureAngles_fib(i) = (abs(phi) - pi/4)/(pi/4)*0.5+0.5 + MeasureAngles_fib(i);
%     else
%         MeasureAngles_fib(i) = abs(phi)/(pi/4)*0.5 + MeasureAngles_fib(i);
%     end
    end
    phi_all(i)=phi;
    MeasureAngles(i)=uint8((255*(phi+pi/2))/pi);
%     MeasureAngles(i) = uint8(255*MeasureAngles_fib(i)/(length(fiber)-1));
%     MeasureAngles(i)=255-MeasureAngles(i);
%     MeasureAngles(i) = uint8((128+127*interm(i))/(length(fiber)-1));
    fibres{i}=uint16(fiber);
end
% clear fibers;
clear x ; clear y,clear N;
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI;
disp('saving in progress....');
save('Largeblock_16bit_phiz_test_whole_rho2_sigma0.5.mat','fibres','MeasureAngles','-v7.3');

%%  All the above stuffs to do fiber tracking (Not useful as of now)
%% Trying 3D Coherence enhancing anis.Diffusion filter (Direct calculation of orientations from 3D Structure Tensor Output)

clear;close all;clc;
% cd /hpc/btho733/ABI/pacedSheep01/medsci_poster/stain_normalisation_toolbox/pacedsheepimages/f2v1/synth_op2/;  %change this for new image
% cd V:\ABI\pacedSheep01\medsci_poster\stain_normalisation_toolbox\pacedsheepimages\v8\output_RH\Nothreshold; 
% cd V:\ABI\JZ\Fiber_2016\Work_2017\CentralSection50um3\cut_for_elevation\bgdCorrect;
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Large_block_25um_filtered/Zplanes;
% cd /hpc_atog/btho733/ABI/JZ/Fiber_2016/Work_2017/CentralSection50um3/raw_25um3/bgdCorrect;
len=200;  %change this for new image
for i=1:len
infile=sprintf('z_%04d.tif',i-1);%outfile=sprintf('cut_%05d.tif',i-500);                       %change this for new image
im=imread(infile);
D1(:,:,i)=squeeze(im(:,:,1));
end
V3=D1(101:250,101:400,1:100);
[Ny, Nx, Nz] = size(V3)

% V=single(D1);
% cd /hpc/btho733/ABI/pacedSheep01/Anisotropic/;
% JR = CoherenceFilter(V,struct('T',20,'dt',2,'Scheme','R'));



sigma =1;

BBori =  single(zeros([Ny Nx Nz]));
for i = 1:1:Nz
    BBori(:,:,i) = 255-(squeeze(V3(:,:,i))); % Changed from rgb3grey to only red-plane
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
rho =3;
[Jxx, Jxy, Jxz, Jyy, Jyz, Jzz] = StructureTensor3D(ux,uy,uz,rho);
%[Jxx, Jxy, Jxz, Jyy, Jyz, Jzz] = AveStructureTensor3D(V,ux,uy,uz,rho);

disp('Done Constructing tensor');

parametersDTI=[];
parametersDTI.BackgroundTreshold=10;
parametersDTI.WhiteMatterExtractionThreshold=0.10;
parametersDTI.textdisplay=true;

% Perform DTI calculation
[coh,EV,VectorF]=testStructureFiber3D1(BBori,Jxx, Jxy, Jxz, Jyy, Jyz, Jzz,parametersDTI);

cd /hpc/btho733/ABI/JZ/Fiber_DTI/Large_block_25um_filtered/ST3D/;
save('VectorF_MidsmallCutLarge_Sig1kernel3_rho3_ker15_07Jan18.mat','VectorF');
%VectorF=importdata('VectorF_Large25um_zplanes_sig2_rho1_30Nov17.mat');
%% Computing Angles and creating HSV model

y=squeeze(VectorF(:,:,:,1));x=squeeze(VectorF(:,:,:,2));z=squeeze(VectorF(:,:,:,3));

% phi = (atan2(z,sqrt(x.^2 + y.^2))).*(180/pi);
% projz=(atan2(x,y)).*(180/pi);
projy=(atan2(z,x)).*(180/pi);
% projx=(atan2(z,y)).*(180/pi); % Naturally, I thought the order is (y,z). But from the observations, I had to reverse it.
% iprojz=(atan2(y,x)).*(180/pi);
neg_iprojz=(atan2(-y,x)).*(180/pi);
% neg_projz=(atan2(-x,y)).*(180/pi);
% converting angle range from -180to180 to -90to90
%##################################################

% [p,q,r]=size(projz);
% for i= 1:p
%     for j=1:q
%        for k=1:r 
%         theta_element=projz(i,j,k);
%         
%         if theta_element >90
%             temp =theta_element-180;
%         elseif theta_element <-90
%             temp = theta_element+180;
%         else 
%             temp=theta_element;
%         end
%         
%         thetaz(i,j,k)=temp;
%        end
%     end
% end
% clear temp;

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

% [p,q,r]=size(iprojz);
% for i= 1:p
%     for j=1:q
%        for k=1:r 
%         theta_element=iprojz(i,j,k);
%         
%         if theta_element >90
%             temp =theta_element-180;
%         elseif theta_element <-90
%             temp = theta_element+180;
%         else 
%             temp=theta_element;
%         end
%         
%         ithetaz(i,j,k)=temp;
%        end
%     end
% end
% clear temp;


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

% [p,q,r]=size(neg_projz);
% for i= 1:p
%     for j=1:q
%        for k=1:r 
%         theta_element=neg_projz(i,j,k);
%         
%         if theta_element >90
%             temp =theta_element-180;
%         elseif theta_element <-90
%             temp = theta_element+180;
%         else 
%             temp=theta_element;
%         end
%         
%         neg_thetaz(i,j,k)=temp;
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
h=(thetay+90)./180; % converting from (-90,90) to (0,1)
s=coh;%ones(560,1000,200);
b=(BBori./255);
hsb=single(zeros(Ny,Nx,Nz));
for i=1:r
    hsb(:,:,i,1)=h(:,:,i);
    hsb(:,:,i,2)=s(:,:,i);
    hsb(:,:,i,3)=b(:,:,i);
    rgb(:,:,i,:)=hsv2rgb(squeeze(hsb(:,:,i,:)));
end
% save('rgb_neg_ithetaz_orient_Large_Sig1kernel3_rho10_ker50_07Dec17.mat','rgb');
%% writing the output images
% clear;clc;
% cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Large_block_25um_filtered/ST3D;
% rgb=importdata('rgb_thetay_orient_Large_Sig3kernel9_rho1_04Dec17.mat');
[~,~,r,~]=size(rgb);
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Large_block_25um_filtered/ST3D/coh_sig1_ker3_rho10_ker50_no_anisfilter/thetay;
for i=1:r
    plane=squeeze(rgb(:,:,i,:));
    name=sprintf('thetay_%04d.png',i-1);
    imwrite(plane,name);
end

%% Fiber Tracking
clc;clear;
tic

% Making the fiber tracking ROI
%##############################
cd /hpc_atog/btho733/ABI/JZ/Fiber_2016/Work_2017/CentralSection50um3/raw_25um3/bgdCorrect;
len=200;  %change this for new image
for p=1:len
infile=sprintf('cut_%04d.tif',p-1);%outfile=sprintf('cut_%05d.tif',i-500);                       %change this for new image
im=imread(infile);
D1(:,:,p)=squeeze(im(:,:,1)); % taking only the redplanes
end

[Ny, Nx, Nz,k] = size(D1)

InvD1 =  single(zeros([Ny Nx Nz]));
for p = 1:Nz
    InvD1(:,:,p) = 255-(squeeze(D1(:,:,p))); 
end

% interest region
Roi0 = logical(zeros([Ny Nx Nz]));
Roi0(InvD1>60) = 1;
Roi=logical(Roi0);
Roi(1:2,:,:) = 0;
Roi((Ny-1):1:Ny,:,:) = 0;
Roi(:,1:2,:) = 0;
Roi(:,(Nx-1):1:Nx,:) = 0;
Roi(:,:,1:2) = 0;
Roi(:,:,(Nz-1):1:Nz) = 0;

% Fiber Tracking Constants
%#########################
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Large_block_25um_filtered/ST3D;
VectorF=importdata('VectorF_Large_Sig1kernel3_rho10_ker50_07Dec17.mat');
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI;
parametersFT=[];
parametersFT.FiberLengthMax=1000000000;
parametersFT.FiberLengthMin=30;
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

cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Large_block_25um_filtered/;
save('FT_Large_Sig1kernel3_rho10_ker50_07Decdata_28Dec17.mat','fibers','-v7.3');
toc


%% Convert from matlab to VV compatible mat file

cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Large_block_25um_filtered/;
clear;clc;close all;
load FT_Smallcut_Large_Sig1kernel3_rho10_ker50_07Decdata_04Jan18.mat;
disp('done loading.....');
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
    
    
    
    projy(i) = atan2(z,-x)*180/pi;
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

%% Saving for VV visualisation
% clear fibers; 
clear x ; clear y,clear N;
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Large_block_25um_filtered/;
disp('saving in progress....');
save('VV_FT_Large25um_sig1_rho10_fiber10_05Jan18_option1_2_znegx_Midsmallcut.mat','fibres','MeasureAngles','-v7.3');

%% Fiber Tracking for Smallcut
clc;clear;
tic

% Making the fiber tracking ROI
%##############################
cd /hpc_atog/btho733/ABI/JZ/Fiber_2016/Work_2017/CentralSection50um3/raw_25um3/bgdCorrect;
len=100;  %change this for new image
for p=1:len
infile=sprintf('cut_%04d.tif',p-1);%outfile=sprintf('cut_%05d.tif',i-500);                       %change this for new image
im=imread(infile);
D1(:,:,p)=squeeze(im(101:250,101:400,1)); % taking only the redplanes
end

[Ny, Nx, Nz,k] = size(D1)

InvD1 =  single(zeros([Ny Nx Nz]));
for p = 1:Nz
    InvD1(:,:,p) = 255-(squeeze(D1(:,:,p))); 
end

% interest region
Roi0 = logical(zeros([Ny Nx Nz]));
Roi0(InvD1>60) = 1;
Roi=logical(Roi0);
Roi(1:2,:,:) = 0;
Roi((Ny-1):1:Ny,:,:) = 0;
Roi(:,1:2,:) = 0;
Roi(:,(Nx-1):1:Nx,:) = 0;
Roi(:,:,1:2) = 0;
Roi(:,:,(Nz-1):1:Nz) = 0;

% Fiber Tracking Constants
%#########################
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Large_block_25um_filtered/ST3D;
VectorF=importdata('VectorF_MidsmallCutLarge_Sig1kernel3_rho10_ker50_04Jan18.mat');
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI;
parametersFT=[];
parametersFT.FiberLengthMax=1000000000;
parametersFT.FiberLengthMin=10;
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

cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Large_block_25um_filtered/;
save('FT_MidsmallCut_Large_Sig1kernel3_rho10_ker50_04Jan18.mat','fibers','-v7.3');
toc

%% For HPC  to cut out MidSmallCut:  Right way to cut a 3d block
clear;clc;close all;
% cd /hpc/btho733/ABI/pacedSheep01/Remontage_Align/;
cd /hpc_atog/btho733/ABI/JZ/Fiber_2016/Work_2017/CentralSection50um3/raw_25um3/bgdCorrect;
im=uint8(imread('cut_0000.tif'));
% im=imresize(im,0.5);
im=im(101:250,101:400,1:3);
[Ny, Nx,k] = size(im);
start_no=1;
end_no=100;
Nz=end_no-start_no+1;
out1= uint8(zeros([Ny,Nx,Nz]));
final_out=uint8(zeros([Ny,Nx,Nz,k]));
for i=1:k

    for j = start_no:1:end_no
    
    infile = sprintf('cut_%04d.tif',j-1)
    file = uint8(imread(infile));
    out1(:,:,j-start_no+1) = file(101:250,101:400,i);
    end
    final_out(:,:,:,i)=out1;
end

D = squeeze(final_out); 


%% For HPC:  Mat to tiff
clc;
% cd /hpc/btho733/ABI/JZ/Fiber_DTI;
d=D; %importdata('CentralSection_after_anisotropicDiff.mat');
[~,~,n,~]=size(d);
cd /hpc/btho733/ABI/JZ/Fiber_DTI/MidSmallCut/;
for i=1:n
dcut(:,:,:)=squeeze(d(:,:,i,:));
outfile=sprintf('cut_%04d.tif',i-1);
imwrite(dcut,outfile,'tif');
end
%% MidSmallcut_FAST version - With Saving only the final VV_FiberTracks

clear;close all;clc;
% cd /hpc/btho733/ABI/pacedSheep01/medsci_poster/stain_normalisation_toolbox/pacedsheepimages/f2v1/synth_op2/;  %change this for new image
% cd V:\ABI\pacedSheep01\medsci_poster\stain_normalisation_toolbox\pacedsheepimages\v8\output_RH\Nothreshold; 
% cd V:\ABI\JZ\Fiber_2016\Work_2017\CentralSection50um3\cut_for_elevation\bgdCorrect;
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Large_block_25um_filtered/Zplanes;
% cd /hpc_atog/btho733/ABI/JZ/Fiber_2016/Work_2017/CentralSection50um3/raw_25um3/bgdCorrect;
len=200;  %change this for new image
for i=1:len
infile=sprintf('z_%04d.tif',i-1);%outfile=sprintf('cut_%05d.tif',i-500);                       %change this for new image
im=imread(infile);
D1(:,:,i)=squeeze(im(:,:,1));
end
V3=D1(101:250,101:400,1:100);
[Ny, Nx, Nz] = size(V3)

% V=single(D1);
% cd /hpc/btho733/ABI/pacedSheep01/Anisotropic/;
% JR = CoherenceFilter(V,struct('T',20,'dt',2,'Scheme','R'));



sigma =1;

BBori =  single(zeros([Ny Nx Nz]));
for i = 1:1:Nz
    BBori(:,:,i) = 255-(squeeze(V3(:,:,i))); % Changed from rgb3grey to only red-plane
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

clear D1;

cd /hpc_atog/btho733/ABI/JZ/Fiber_2016/Work_2017/CentralSection50um3/raw_25um3/bgdCorrect;
len=100;  %change this for new image
for p=1:len
infile=sprintf('cut_%04d.tif',p-1);%outfile=sprintf('cut_%05d.tif',i-500);                       %change this for new image
im=imread(infile);
D1(:,:,p)=squeeze(im(101:250,101:400,1)); % taking only the redplanes
end

[Ny, Nx, Nz,k] = size(D1)

InvD1 =  single(zeros([Ny Nx Nz]));
for p = 1:Nz
    InvD1(:,:,p) = 255-(squeeze(D1(:,:,p))); 
end

% interest region
Roi0 = logical(zeros([Ny Nx Nz]));
Roi0(InvD1>60) = 1;
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
parametersFT.FiberLengthMin=10;
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
    
    
    
    projy(i) = atan2(z,x)*180/pi;
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
save('VV_FT_Large25um_sig1_rho5_k15_fiber10_12Jan18_zx_Midsmallcut.mat','fibres','MeasureAngles','-v7.3');
%% %% Badsection_Small_FAST version - With Saving only the final VV_FiberTracks

clear;close all;clc;
% cd /hpc/btho733/ABI/pacedSheep01/medsci_poster/stain_normalisation_toolbox/pacedsheepimages/f2v1/synth_op2/;  %change this for new image
% cd V:\ABI\pacedSheep01\medsci_poster\stain_normalisation_toolbox\pacedsheepimages\v8\output_RH\Nothreshold; 
% cd V:\ABI\JZ\Fiber_2016\Work_2017\CentralSection50um3\cut_for_elevation\bgdCorrect;
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Badsection_small/a101_zplanes_corrected;
% cd /hpc_atog/btho733/ABI/JZ/Fiber_2016/Work_2017/CentralSection50um3/raw_25um3/bgdCorrect;
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
    namey=sprintf('y_%04d.png',i-1);
    namez=sprintf('z_%04d.png',i-1);
    cd y;imwrite(plane_y,namey);cd ..;
    cd z;imwrite(plane_z,namez);cd ..;
end

%% Midsmallcut  : Computing Angles and creating HSV model

y=squeeze(VectorF(:,:,:,1));x=squeeze(VectorF(:,:,:,2));z=squeeze(VectorF(:,:,:,3));
projy=(atan2(-z,x)).*(180/pi);
neg_iprojz=(atan2(y,x)).*(180/pi);
projx=(atan2(z,y)).*(180/pi);
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
h_z=(neg_ithetaz+90)./180; % converting from (-90,90) to (0,1)
s=coh;%ones(560,1000,200);
b=(orig./255);
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
%% writing the output images
% clear;clc;
% cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Large_block_25um_filtered/ST3D;
% rgb=importdata('rgb_thetay_orient_Large_Sig3kernel9_rho1_04Dec17.mat');
[~,~,r,~]=size(rgb_y);
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/MidSmallCut/a301_ST3D_results/s1m3rho5k15/;
for i=1:r
    plane_x=squeeze(rgb_x(:,:,i,:));
    plane_y=squeeze(rgb_y(:,:,i,:));
    plane_z=squeeze(rgb_z(:,:,i,:));
    namex=sprintf('x_%04d.png',i-1);
    namey=sprintf('y_%04d.png',i-1);
    namez=sprintf('z_%04d.png',i-1);
    cd x/zx;imwrite(plane_x,namex);cd ..;cd ..;
    cd y;imwrite(plane_y,namey);cd ..;
    cd z;imwrite(plane_z,namez);cd ..;
end
%% Processing midSmallCut to get Contrast enhanced version

%##############        Method 1     #########

% cutting out Midsmallcut
clear;close all;clc;
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Large_block_25um_filtered/Zplanes;
len=200;  %change this for new image
for i=1:len
infile=sprintf('z_%04d.tif',i-1);%outfile=sprintf('cut_%05d.tif',i-500);                       %change this for new image
im=imread(infile);
D1(:,:,i)=squeeze(im(:,:,1));
end
V1=D1(101:250,101:400,1:100);
[Ny, Nx, Nz] = size(V1);

% For Saving zplanes (making a101_filtered_zplanes)
% cd /hpc/btho733/ABI/JZ/Fiber_DTI/MidSmallCut/a101_filtered_zplanes
% for u=1:Nz
%     p1=squeeze(V1(:,:,u));
%     name=sprintf('z_%04d.tif',u-1);
%     imwrite(p1,name);
% end

% For converting indexed 2 RGB before contrast enhancement in ImageJ(making a102_filtered_rgb)
cd /hpc/btho733/ABI/JZ/Fiber_DTI/MidSmallCut;
for u=1:Nz
    p1=squeeze(V1(:,:,u));
    p1_RGB=uint8(ind2rgb(p1,255*summer(256)));
    name=sprintf('z_%04d.tif',u-1);
    cd a102_filtered_rgb; imwrite(p1_RGB,name);cd ..;
end



%##############        Method 2      #########
% For converting indexed 2 RGB after contrast enhancement in ImageJ(making a105_rgb_froma103)
clear;close all;clc;

len=100;  %change this for new image
for i=1:len
infile=sprintf('z_%04d.tif',i-1);%outfile=sprintf('cut_%05d.tif',i-500);                       %change this for new image
cd /hpc/btho733/ABI/JZ/Fiber_DTI/MidSmallCut/a103_Contrast_enhanced_froma101;im=imread(infile);
p1_RGB=uint8(ind2rgb(im,255*summer(256)));
cd /hpc/btho733/ABI/JZ/Fiber_DTI/MidSmallCut/a105_rgb_froma103; imwrite(p1_RGB,infile);
D1(:,:,i,:)=p1_RGB;
end


%############## LOading Contrast enhanced a104_Reversed ############
clear;close all;clc;
len=100;  %change this for new image
for i=1:len
infile=sprintf('z_%04d.tif',i-1);%outfile=sprintf('cut_%05d.tif',i-500);                       %change this for new image
cd /hpc/btho733/ABI/JZ/Fiber_DTI/MidSmallCut/a104_contrast_enhanced_from_a102/Reversed/;
im=imread(infile);
D1(:,:,i,:)=im;
end
cd /hpc/btho733/ABI/pacedSheep01/Anisotropic/functions/;
showcs3(single(D1))
