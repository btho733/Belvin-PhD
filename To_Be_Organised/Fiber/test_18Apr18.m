%%
% For details, see the notes

clear;close all;
clc;
% cd V:\ABI\JZ\Fiber_DTI\Large_block_25um_filtered\Zplanes\grey\;
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Large_block_25um_filtered/Zplanes/grey;
len=200;  %change this for new image
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
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/Largeblock4Anisotropy/s1m3rho1k5;
EV1=importdata('EV_s1m3rho1k5.mat');
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/Largeblock4Anisotropy/s1m3rho3k15/;
EV3=importdata('EV_s1m3rho3k15.mat');
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/Largeblock4Anisotropy/s1m3rho5k25;
EV5=importdata('EV_s1m3rho5k25.mat');
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/Largeblock4Anisotropy/s1m3rho10k50;
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
figure,imagesc(Df(:,:,190));colormap(gray);title('original slice 190')
figure,imagesc(Df(:,:,191));colormap(gray);title('original slice 191')
figure,imagesc(squeeze(EV1(:,:,190,:)));title('scale 1(slice 190)');
figure,imagesc(squeeze(EV1(:,:,191,:)));title('scale 1(slice 191)');
figure,imagesc(squeeze(EV3(:,:,190,:)));title('scale 3(slice 190)');
figure,imagesc(squeeze(EV3(:,:,191,:)));title('scale 3(slice 191)');
figure,imagesc(squeeze(EV5(:,:,190,:)));title('scale 5(slice 190)');
figure,imagesc(squeeze(EV5(:,:,191,:)));title('scale 5(slice 191)');
figure,imagesc(squeeze(EV10(:,:,190,:)));title('scale 10(slice 190)');
figure,imagesc(squeeze(EV10(:,:,191,:)));title('scale 10(slice 191)');
cd /hpc/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/Largeblock4Anisotropy/s1m3rho5k25/rgby;
im190=imread('y_0190.png');
im191=imread('y_0191.png');
figure,imagesc(im191);title('cross-plane orientation 191 (scale 5)');
figure,imagesc(im190);title('cross-plane orientation 190 (scale 5)');
figure,imagesc(squeeze(r23_s1(:,:,190,:)));title('Ratio(2/3) scale 1(slice 190)');colormap(jet);
figure,imagesc(squeeze(r23_s1(:,:,191,:)));title('Ratio(2/3) scale 1(slice 191)');colormap(jet);
figure,imagesc(squeeze(r23_s3(:,:,190,:)));title('Ratio(2/3) scale 3(slice 190)');colormap(jet);
figure,imagesc(squeeze(r23_s3(:,:,191,:)));title('Ratio(2/3) scale 3(slice 191)');colormap(jet);
figure,imagesc(squeeze(r23_s5(:,:,190,:)));title('Ratio(2/3) scale 5(slice 190)');colormap(jet);
figure,imagesc(squeeze(r23_s5(:,:,191,:)));title('Ratio(2/3) scale 5(slice 191)');colormap(jet);
figure,imagesc(squeeze(r23_s10(:,:,190,:)));title('Ratio(2/3) scale 10(slice 190)');colormap(jet);
figure,imagesc(squeeze(r23_s10(:,:,191,:)));title('Ratio(2/3) scale 10(slice 191)');colormap(jet);

%% thresholding out

x=r23_s1(:,:,190,:);thr=0.075
x(x<thr)=0;
x(x>thr)=1;
figure,imagesc(x);title('0.075 thresh Ratio(2/3) scale 1(slice 190)');colormap(jet);

%% Trying 3D Coherence enhancing anis.Diffusion filter
clear;close all;clc;
% cd /hpc/btho733/ABI/pacedSheep01/medsci_poster/stain_normalisation_toolbox/pacedsheepimages/f2v1/synth_op2/;  %change this for new image
% cd V:\ABI\pacedSheep01\medsci_poster\stain_normalisation_toolbox\pacedsheepimages\v8\output_RH\Nothreshold; 
% cd V:\ABI\JZ\Fiber_2016\Work_2017\CentralSection50um3\cut_for_elevation\bgdCorrect;
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Large_block_25um_filtered/Zplanes/grey/;
% cd /hpc_atog/btho733/ABI/JZ/Fiber_2016/Work_2017/CentralSection50um3/raw_25um3/bgdCorrect;
len=200;  %change this for new image
for i=1:len
infile=sprintf('z_%04d.tif',i-1);%outfile=sprintf('cut_%05d.tif',i-500);                       %change this for new image
im=imread(infile);

D1(:,:,i)=im(101:250,1:200);

end

[Ny, Nx, Nz,k] = size(D1)

V=single(D1);
cd /hpc/btho733/ABI/pacedSheep01/Anisotropic/;
JR = CoherenceFilter(V,struct('T',20,'dt',2,'Scheme','R'));