clear;close all;clc;
% cd V:\ABI\JZ\Fiber_DTI\Large_block_25um_filtered\Zplanes\grey\;
cd /hpc/btho733/ABI/JZ/Fiber_DTI/Badsection_small/a501_contrast_improvement/a3_CLAHE_15_255_5/
len=70;  %change this for new image
for i=1:len
infile=sprintf('clahe_15_255_5_%04d.tif',i-1);                      
im=imread(infile);
Df(:,:,len-i+1)=im;

end
% cd V:\ABI\pacedSheep01\Anisotropic\functions;
% showcs3(single(Df));
V1=Df;
[Ny, Nx, Nz] = size(V1)

% V=single(D1);
% cd /hpc/btho733/ABI/pacedSheep01/Anisotropic/;
% JR = CoherenceFilter(V,struct('T',15,'dt',2,'Scheme','R'));


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
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/Badsectionsmall4Anisotropy/s1m3rho1k5/;
save('EV_s1m3rho1k5.mat','EV','-v7.3');
disp('saved EV...moving to VectorF');
save('VectorF_s1m3rho1k5.mat','VectorF','-v7.3');
toc;

% ###############################################################################

clear;
% cd V:\ABI\JZ\Fiber_DTI\Large_block_25um_filtered\Zplanes\grey\;
cd /hpc/btho733/ABI/JZ/Fiber_DTI/Badsection_small/a501_contrast_improvement/a3_CLAHE_15_255_5/
len=70;  %change this for new image
for i=1:len
infile=sprintf('clahe_15_255_5_%04d.tif',i-1);                      
im=imread(infile);
Df(:,:,len-i+1)=im;

end
% cd V:\ABI\pacedSheep01\Anisotropic\functions;
% showcs3(single(Df));
V1=Df;
[Ny, Nx, Nz] = size(V1)

% V=single(D1);
% cd /hpc/btho733/ABI/pacedSheep01/Anisotropic/;
% JR = CoherenceFilter(V,struct('T',15,'dt',2,'Scheme','R'));


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
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/Badsectionsmall4Anisotropy/s1m3rho3k15/;
save('EV_s1m3rho3k15.mat','EV','-v7.3');
disp('saved EV...moving to VectorF');
save('VectorF_s1m3rho3k15.mat','VectorF','-v7.3');
toc;


% ############################################################################

clear;
% cd V:\ABI\JZ\Fiber_DTI\Large_block_25um_filtered\Zplanes\grey\;
cd /hpc/btho733/ABI/JZ/Fiber_DTI/Badsection_small/a501_contrast_improvement/a3_CLAHE_15_255_5/;
len=70;  %change this for new image
for i=1:len
infile=sprintf('clahe_15_255_5_%04d.tif',i-1);                      
im=imread(infile);
Df(:,:,len-i+1)=im;

end
% cd V:\ABI\pacedSheep01\Anisotropic\functions;
% showcs3(single(Df));
V1=Df;
[Ny, Nx, Nz] = size(V1)

% V=single(D1);
% cd /hpc/btho733/ABI/pacedSheep01/Anisotropic/;
% JR = CoherenceFilter(V,struct('T',15,'dt',2,'Scheme','R'));


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
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/Badsectionsmall4Anisotropy/s1m3rho5k25/;
save('EV_s1m3rho5k25.mat','EV','-v7.3');
disp('saved EV...moving to VectorF');
save('VectorF_s1m3rho5k25.mat','VectorF','-v7.3');
toc;

% ##############################################################################################

clear;close all;clc;
% cd V:\ABI\JZ\Fiber_DTI\Large_block_25um_filtered\Zplanes\grey\;
cd /hpc/btho733/ABI/JZ/Fiber_DTI/Badsection_small/a501_contrast_improvement/a3_CLAHE_15_255_5/;
len=70;  %change this for new image
for i=1:len
infile=sprintf('clahe_15_255_5_%04d.tif',i-1);                      
im=imread(infile);
Df(:,:,len-i+1)=im;

end
% cd V:\ABI\pacedSheep01\Anisotropic\functions;
% showcs3(single(Df));
V1=Df;
[Ny, Nx, Nz] = size(V1)

% V=single(D1);
% cd /hpc/btho733/ABI/pacedSheep01/Anisotropic/;
% JR = CoherenceFilter(V,struct('T',15,'dt',2,'Scheme','R'));


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
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/Badsectionsmall4Anisotropy/s1m3rho10k50/;
save('EV_s1m3rho10k50.mat','EV','-v7.3');
disp('saved EV...moving to VectorF');
save('VectorF_s1m3rho10k50.mat','VectorF','-v7.3');
toc;

exit;

%%  LOading Eigenvalues and computing ratios
clear;close all;clc;
% cd V:\ABI\JZ\Fiber_DTI\Large_block_25um_filtered\Zplanes\grey\;
cd /hpc/btho733/ABI/JZ/Fiber_DTI/Badsection_small/a103_zplanes_froma101_contrast_enhanced_imagej;
len=70;  %change this for new image
for i=1:len
infile=sprintf('z_%04d.tif',i-1);

im=imread(infile);
Df(:,:,len-i+1)=im;

end
[Ny, Nx, Nz] = size(Df);

InvV1 =  single(zeros([Ny Nx Nz]));
for i = 1:1:Nz
    InvV1(:,:,i) = 255-(squeeze(Df(:,:,i))); % Changed from rgb3grey to only red-plane
end
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/Badsectionsmall4Anisotropy/After_contrast_improvement/a3_clahe/s1m3rho1k5/
EV1=importdata('EV_s1m3rho1k5.mat');
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/Badsectionsmall4Anisotropy/After_contrast_improvement/a3_clahe/s1m3rho3k15/;
EV3=importdata('EV_s1m3rho3k15.mat');
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/Badsectionsmall4Anisotropy/After_contrast_improvement/a3_clahe/s1m3rho5k25;
EV5=importdata('EV_s1m3rho5k25.mat');
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/Badsectionsmall4Anisotropy/After_contrast_improvement/a3_clahe/s1m3rho10k50;
EV10=importdata('EV_s1m3rho10k50.mat');
% 
EV1_1=EV1(:,:,:,1);EV1_2=EV1(:,:,:,2);EV1_3=EV1(:,:,:,3);
EV3_1=EV3(:,:,:,1);EV3_2=EV3(:,:,:,2);EV3_3=EV3(:,:,:,3);
EV5_1=EV5(:,:,:,1);EV5_2=EV5(:,:,:,2);EV5_3=EV5(:,:,:,3);
EV10_1=EV10(:,:,:,1);EV10_2=EV10(:,:,:,2);EV10_3=EV10(:,:,:,3);
r23_s1=EV1_2./EV1_3;r23_s3=EV3_2./EV3_3;r23_s5=EV5_2./EV5_3;r23_s10=EV10_2./EV10_3;
% 
% h1=(EV1_3-EV1_2)./EV1_3;
% s1=(EV1_2-EV1_1)./EV1_3;
% b1=InvV1./255;
% b1df=Df./255;
% hsb1(:,:,:,1)=h1;hsb1(:,:,:,2)=s1;hsb1(:,:,:,3)=b1df;
%% Figures
close all;
figure,imagesc(Df(:,:,51));colormap(gray);title('original slice 51')
figure,imagesc(Df(:,:,52));colormap(gray);title('original slice 52')
% figure,imagesc(squeeze(EV1(:,:,51,:)));title('scale 1(slice 51)');
% figure,imagesc(squeeze(EV1(:,:,52,:)));title('scale 1(slice 52)');
% figure,imagesc(squeeze(EV3(:,:,51,:)));title('scale 3(slice 51)');
% figure,imagesc(squeeze(EV3(:,:,52,:)));title('scale 3(slice 52)');
% figure,imagesc(squeeze(EV5(:,:,51,:)));title('scale 5(slice 51)');
% figure,imagesc(squeeze(EV5(:,:,52,:)));title('scale 5(slice 52)');
% figure,imagesc(squeeze(EV10(:,:,51,:)));title('scale 10(slice 51)');
% figure,imagesc(squeeze(EV10(:,:,52,:)));title('scale 10(slice 52)');
cd /hpc/btho733/ABI/JZ/Fiber_DTI/Badsection_small/a301_ST3D_results/s1m3rho5k15/revy;
im51=imread('y_0051.png');
im52=imread('y_0052.png');
figure,imagesc(im52);title('cross-plane orientation 52 (scale 5)');
figure,imagesc(im51);title('cross-plane orientation 51 (scale 5)');
figure,imagesc(squeeze(r23_s1(:,:,51,:)));title('Ratio(2/3) scale 1(slice 51)');colormap(jet);
figure,imagesc(squeeze(r23_s1(:,:,52,:)));title('Ratio(2/3) scale 1(slice 52)');colormap(jet);
figure,imagesc(squeeze(r23_s3(:,:,51,:)));title('Ratio(2/3) scale 3(slice 51)');colormap(jet);
figure,imagesc(squeeze(r23_s3(:,:,52,:)));title('Ratio(2/3) scale 3(slice 52)');colormap(jet);
figure,imagesc(squeeze(r23_s5(:,:,51,:)));title('Ratio(2/3) scale 5(slice 51)');colormap(jet);
figure,imagesc(squeeze(r23_s5(:,:,52,:)));title('Ratio(2/3) scale 5(slice 52)');colormap(jet);
figure,imagesc(squeeze(r23_s10(:,:,51,:)));title('Ratio(2/3) scale 10(slice 51)');colormap(jet);
figure,imagesc(squeeze(r23_s10(:,:,52,:)));title('Ratio(2/3) scale 10(slice 52)');colormap(jet);