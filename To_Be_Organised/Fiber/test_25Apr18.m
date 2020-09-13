tic;
%Loading 3D data
%###############
clear;close all;clc;
% cd /hpc/btho733/ABI/pacedSheep01/medsci_poster/stain_normalisation_toolbox/pacedsheepimages/f2v1/synth_op2/;  %change this for new image
% cd V:\ABI\JZ\Fiber_2016\Work_2017\CentralSection50um3\necrotic_scale4;
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/small_section_from_rgboutpng_351to600;
% cd V:\ABI\JZ\Fiber_2016\Work_2017\CentralSection50um3\cut_for_elevation\bgdCorrect;
% cd /hpc_atog/btho733/ABI/JZ/Fiber_2016/Work_2017/CentralSection50um3/raw_25um3/bgdCorrect;
% cd /hpc_atog/btho733/ABI/JZ/Fiber_2016/Work_2017/CentralSection50um3/raw_50um3/bgdBlack;
len=250;  %change this for new image
for i=1:len
infile=sprintf('Im_351to600_%04d.tif',i-1);%outfile=sprintf('cut_%05d.tif',i-500);                       %change this for new image
im=imread(infile);
% im=im(y1:y2,x1:x2,:);
p1=im(:,:,1);l
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
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/small_section_from_rgboutpng_351to600;
save('InvV1_351to600_afterfiltering.mat','InvV1','-v7.3');

cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/small_section_from_rgboutpng_351to600/results_s1m3rho5k25/
EV5=importdata('EV_s1m3rho5k25.mat');

cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/small_section_from_rgboutpng_351to600/results_s1m3rho50k250/
EV50=importdata('EV_s1m3rho50k250.mat');
% 
% EV1_1=EV1(:,:,:,1);EV1_2=EV1(:,:,:,2);EV1_3=EV1(:,:,:,3);
% EV3_1=EV3(:,:,:,1);EV3_2=EV3(:,:,:,2);EV3_3=EV3(:,:,:,3);
EV5_1=EV5(:,:,:,1);EV5_2=EV5(:,:,:,2);EV5_3=EV5(:,:,:,3);
% EV10_1=EV10(:,:,:,1);EV10_2=EV10(:,:,:,2);EV10_3=EV10(:,:,:,3);
EV50_1=EV50(:,:,:,1);EV50_2=EV50(:,:,:,2);EV50_3=EV50(:,:,:,3);
% r23_s1=EV1_2./EV1_3;r23_s3=EV3_2./EV3_3;
r23_s5=EV5_2./EV5_3;
% r23_s10=EV10_2./EV10_3;
r23_s50=EV50_2./EV50_3;
toc;
% 
% h1=(EV1_3-EV1_2)./EV1_3;
% s1=(EV1_2-EV1_1)./EV1_3;
% b1=InvV1./255;
% b1df=Df./255;
% hsb1(:,:,:,1)=h1;hsb1(:,:,:,2)=s1;hsb1(:,:,:,3)=b1df;
%% Figures

% figure,imagesc(squeeze(EV1(:,:,190,:)));title('scale 1(slice 190)');
% figure,imagesc(squeeze(EV1(:,:,191,:)));title('scale 1(slice 191)');
% figure,imagesc(squeeze(EV3(:,:,190,:)));title('scale 3(slice 190)');
% figure,imagesc(squeeze(EV3(:,:,191,:)));title('scale 3(slice 191)');
% figure,imagesc(squeeze(EV5(:,:,190,:)));title('scale 5(slice 190)');
% figure,imagesc(squeeze(EV5(:,:,191,:)));title('scale 5(slice 191)');
% figure,imagesc(squeeze(EV10(:,:,190,:)));title('scale 10(slice 190)');
% figure,imagesc(squeeze(EV10(:,:,191,:)));title('scale 10(slice 191)');
% im190=imread('y_0190.png');
% im191=imread('y_0191.png');
% figure,imagesc(im191);title('cross-plane orientation 191 (scale 5)');
% figure,imagesc(im190);title('cross-plane orientation 190 (scale 5)');
figure,imagesc(squeeze(r23_s1(:,:,190,:)));title('Ratio(2/3) scale 1(slice 190)');colormap(jet);
figure,imagesc(squeeze(r23_s1(:,:,191,:)));title('Ratio(2/3) scale 1(slice 191)');colormap(jet);
figure,imagesc(squeeze(r23_s3(:,:,190,:)));title('Ratio(2/3) scale 3(slice 190)');colormap(jet);
figure,imagesc(squeeze(r23_s3(:,:,191,:)));title('Ratio(2/3) scale 3(slice 191)');colormap(jet);
figure,imagesc(squeeze(r23_s5(:,:,120,:)));title('Ratio(2/3) scale 5(slice 190)');colormap(jet);
figure,imagesc(squeeze(r23_s5(:,:,191,:)));title('Ratio(2/3) scale 5(slice 191)');colormap(jet);
figure,imagesc(squeeze(r23_s10(:,:,190,:)));title('Ratio(2/3) scale 10(slice 190)');colormap(jet);
figure,imagesc(squeeze(r23_s10(:,:,191,:)));title('Ratio(2/3) scale 10(slice 191)');colormap(jet);