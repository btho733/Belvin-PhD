%% use the default yxz axis instead of rotating around 
clear;close all;clc;
% cd /hpc/btho733/ABI/pacedSheep01/medsci_poster/stain_normalisation_toolbox/pacedsheepimages/f2v1/synth_op2/;  %change this for new image
% cd V:\ABI\pacedSheep01\medsci_poster\stain_normalisation_toolbox\pacedsheepimages\v8\output_RH\Nothreshold; 
cd V:\ABI\JZ\Fiber_2016\Work_2017\CentralSection50um3\cut_for_elevation\bgdCorrect;
% cd V:\ABI\JZ\Fiber_DTI\filtered_images\contrast_enhanced;
% cd V:\ABI\JZ\Fiber_2016\Work_2017\CentralSection50um3\raw_25um3\bgdCorrect;
len=100;  %change this for new image
for i=1:len
infile=sprintf('cut_%04d.tif',i-1);%outfile=sprintf('cut_%05d.tif',i-500);                       %change this for new image
im=imread(infile);
% im=im(y1:y2,x1:x2,:);
im1=im(:,:,1);
im2=im(:,:,2);
im3=im(:,:,3);

D1(:,:,i,1)=im1;
D1(:,:,i,2)=im2;
D1(:,:,i,3)=im3;
end

[Ny, Nx, Nz,k] = size(D1)
%load PVblock/LowResSegedPVori.mat;
%load PVblock/SmoothedLowResPVnewori.mat
%load PVblock/XYaxisSmoothedLowResSegedPVori2.mat;
%load PVblock/XYaxis6Smoothed2LowResSegedPVori.mat;
%%
V3=D1;
sigma = 1;
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
BBori(BBori>180) = 180;  %%% added by jichao on 5 Dec 2017

% interest region
Roi0 = logical(zeros([Ny Nx Nz]));

%Roi0(BBori>10) = 1;
Roi0(BBori>90) = 1;  %%% added by jichao on Dec 2017

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

cd V:\ABI\JZ\Fiber_DTI;
usigma=imgaussian(V,sigma,5*sigma);

% Calculate the gradients
ux=derivatives(usigma,'x');
uy=derivatives(usigma,'y');
uz=derivatives(usigma,'z');
% [ux,uy,uz] = imgradientxyz(usigma,'sobel');
% Compute the 3D structure tensors J of the image
rho = 1;
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

parametersFT.FAmin = 90;
parametersFT.FAmax = 255;

% Perform fiber tracking
fibers=FT2(VectorF,Roi,parametersFT);
% 
% % Show FA
save -v7.3 cut4elevation_fibers.mat;

%%

% cd /hpc/btho733/ABI/JZ/Fiber_2016/v8_f53_comparison_Matlab_Vs_VV/Matlab;
cd V:\ABI\JZ\Fiber_DTI;
clear;clc;close all;
load cut4elevation_fibers.mat;
% jz=importdata('v8_f53_fiber30.mat');
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
    
    
    
    phi = atan2(z,sqrt(x.^2 + y.^2));
    if (abs(phi) >= pi/4)
        MeasureAngles(i) = (abs(phi) - pi/4)/(pi/4)*0.5+0.5 + MeasureAngles(i);
    else
        MeasureAngles(i) = abs(phi)/(pi/4)*0.5 + MeasureAngles(i);
    end
    end

    MeasureAngles(i) = uint8(255*MeasureAngles(i)/(length(fiber)-1));
    fibres{i}=uint16(fiber);
end
clear fibers; clear x ; clear y,clear N;
cd V:\ABI\JZ\Fiber_DTI;
disp('saving in progress....');
save('cut4elevation_16bit_phiz.mat','fibres','MeasureAngles','-v7.3');
