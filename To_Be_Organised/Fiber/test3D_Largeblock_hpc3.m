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
len=70;  %change this for new image
for i=1:len
infile=sprintf('z_%04d.tif',i-1);%outfile=sprintf('cut_%05d.tif',i-500);                       %change this for new image
im=imread(infile);
D1(:,:,i)=squeeze(im(131:350,401:700,1));
end

[Ny, Nx, Nz,k] = size(D1)

V=single(D1);
% cd /hpc/btho733/ABI/pacedSheep01/Anisotropic/;
% JR = CoherenceFilter(V,struct('T',20,'dt',2,'Scheme','R'));

V3=V;

sigma =12;
[Ny, Nx, Nz, k] = size(V3)
BBori =  single(zeros([Ny Nx Nz]));
for i = 1:1:Nz
    BBori(:,:,i) = 255-(squeeze(V3(:,:,i))); % Changed from rgb3grey to only red-plane
end
% Interpolate BBori from 50um to 8 um resolution for commputer modeling

% k = 6.25/25
% d = size(BBori);
% [xi,yi,zi] = meshgrid(1:d(2),1:d(1),1:k:d(3));
% BBori = interp3(BBori,xi,yi,zi,'cubic');

[Ny, Nx, Nz] = size(BBori)

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
usigma=imgaussian(BBori,sigma,10);

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

% Perform DTI calculation
VectorF=testStructureFiber3D1(BBori,Jxx, Jxy, Jxz, Jyy, Jyz, Jzz,parametersDTI);


%VectorF=importdata('VectorF_Large25um_zplanes_sig2_rho1_30Nov17.mat');
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

cd /hpc/btho733/ABI/JZ/Fiber_DTI/Large_block_25um_filtered/ST3D/;
save('VectorF_PartLarge25um_zplanes_sig12kernel10_rho10_05Dec17.mat','VectorF');
% Making the hsv model of orientation 
% ###################################
% h=orientation angle, 
% s=1(can be replaced by coherence for better results), 
% v=inverted red channel of original(can be the inverted greyscale version also)
% The hsv model is converted to rgb for visualisation purpose(In order to
% match with the visualisation from 2D orientation)
% clear h;clear s;clear b;clear hsb;clear rgb;
h=(neg_ithetaz+90)./180;
s=ones(Ny,Nx,Nz);
b=(BBori./255);
hsb=single(zeros(Ny,Nx,Nz));
for i=1:r
    hsb(:,:,i,1)=h(:,:,i);
    hsb(:,:,i,2)=s(:,:,i);
    hsb(:,:,i,3)=b(:,:,i);
    rgb(:,:,i,:)=hsv2rgb(squeeze(hsb(:,:,i,:)));
end
save('rgb_neg_ithetaz_orient_PartLarge_Sig12kernel10_rho10_05Dec17.mat','rgb');
%% writing the output images

cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Large_block_25um_filtered/partial_block2/sig12_ker10_rho10_ker10/neg_ithetaz;
for i=1:r
    plane=squeeze(rgb(:,:,i,:));
    name=sprintf('ithetaz_%04d.png',i-1);
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
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Large_block_25um_filtered/ST3D_results;
VectorF=importdata('VectorF_Large25um_zplanes_sig2_rho1_30Nov17.mat');
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
save('FT_Large25um_zplanes_sig2_rho1_fiber30_01Dec17.mat','fibers','-v7.3');
toc