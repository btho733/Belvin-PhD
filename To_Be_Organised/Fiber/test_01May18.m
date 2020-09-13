clear;close all;
clc;
% cd V:\ABI\JZ\Fiber_DTI\Large_block_25um_filtered\Zplanes\grey\;
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Large_block_25um_filtered/Zplanes/grey;
len=100;  %change this for new image
for i=1:len
infile=sprintf('z_%04d.tif',i-1);                      
im=imread(infile);
Ns(:,:,len-i+1)=im(151:300,61:160);
Df(:,:,len-i+1)=imsharpen(im(151:300,61:160),'Radius',2,'Amount',1,'Threshold',0.001);

end
cd /hpc_atog/btho733/ABI/pacedSheep01/Anisotropic/functions/;
showcs3(single(Df));
%%
cd /hpc_atog/btho733/ABI/pacedSheep01/Anisotropic;

JR=CoherenceFilter(Df,struct('T',20,'dt',2,'Scheme','R'));
AD=JR;
JR(JR>135)=255;
JR(JR<100)=0;
cd /hpc_atog/btho733/ABI/pacedSheep01/Anisotropic/functions/;
showcs3(JR);
%%
cd /hpc_atog/btho733/ABI/pacedSheep01/Anisotropic;JR1=CoherenceFilter(JR,struct('T',20,'dt',2,'Scheme','R'));
cd /hpc_atog/btho733/ABI/pacedSheep01/Anisotropic/functions/;showcs3(JR1);

%%
cd /hpc_atog/btho733/ABI/pacedSheep01/Anisotropic;JS1 = CoherenceFilter(JR,struct('T',5,'dt',0.15,'Scheme','S','eigenmode',4));
cd /hpc_atog/btho733/ABI/pacedSheep01/Anisotropic/functions/;showcs3(JS1);

%%
cd /hpc_atog/btho733/ABI/pacedSheep01/Anisotropic;JR2 = CoherenceFilter(JR,struct('T',1,'dt',0.1,'rho',4,'Scheme','R','eigenmode',4));
cd /hpc_atog/btho733/ABI/pacedSheep01/Anisotropic/functions/;showcs3(JR2);
%%
cd /hpc_atog/btho733/ABI/pacedSheep01/Anisotropic;JR3 = CoherenceFilter(JR,struct('T',1,'dt',0.1,'rho',10,'Scheme','R','eigenmode',4));
cd /hpc_atog/btho733/ABI/pacedSheep01/Anisotropic/functions/;showcs3(JR3);
%%
cd /hpc_atog/btho733/ABI/pacedSheep01/Anisotropic;JR4=CoherenceFilter(JR,struct('T',20,'dt',2,'Scheme','R','eigenmode',3));
cd /hpc_atog/btho733/ABI/pacedSheep01/Anisotropic/functions/;showcs3(JR4);

%%

t4=JR4;
t4(t4>150)=255;
t4(t4<130)=0;
showcs3(t4);

%%
cd /hpc_atog/btho733/ABI/pacedSheep01/Anisotropic;JR5=CoherenceFilter(t4,struct('T',20,'dt',2,'Scheme','R'));
cd /hpc_atog/btho733/ABI/pacedSheep01/Anisotropic/functions/;showcs3(JR5);
%%
len=100;  
for i=1:len                     
im=squeeze(JR4(:,:,i));
sh(:,:,i)=imsharpen(im,'Radius',1,'Amount',1,'Threshold',0.8);

end
showcs3(sh)

%%  Clustering the color images using kmeans
%############################################

%% %% Approach #1 : First doing kmeans and then removing jitters 
clear;close all;clc;

len=100;  %change this for new image
for i=1:len
infile=sprintf('cut_%04d.tif',i-1);%outfile=sprintf('cut_%05d.tif',i-500);                       %change this for new image
cd /hpc_atog/btho733/ABI/JZ/Fiber_2016/Work_2017/CentralSection50um3/raw_25um3/bgdCorrect/;
im=imread(infile);
cd /hpc_atog/btho733/ABI/matlab_central/clustering;
[label_im,vec_mean] = kmeans_fast_Color_original(im(151:300,61:160,:),2,1);
D1_label(:,:,len-i+1)=label_im;
p1=im(151:300,61:160,1);
p2=im(151:300,61:160,2);
p3=im(151:300,61:160,3);

D1(:,:,len-i+1,1)=p1;
D1(:,:,len-i+1,2)=p2;
D1(:,:,len-i+1,3)=p3;

disp(i);
end

%% CIRCULAR Filter : New(using fpecial circular filter) Filtering for All 3D Y-planes and X-planes : <<<<   RGB Correct version  >>>>>>>
%######################################################################################################################################

[Ny, Nx, Nz,k] = size(D1_label);
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI;
circular_block=squeeze(D1_label(:,:,:,1));
corrected_zplane=255*ones(Ny,Nx);upto=Nz-1;
for zplane=1:upto
    r=10;
    if(zplane==1)       
        im1=squeeze(D1_label(:,:,zplane,1));
        im2=squeeze(D1_label(:,:,zplane+1,1));
    else
        im1=corrected_zplane;
        im2=squeeze(D1_label(:,:,zplane+1,1));
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
%%
cd /hpc_atog/btho733/ABI/pacedSheep01/Anisotropic;JR1=CoherenceFilter(circular_block,struct('T',20,'dt',2,'Scheme','R'));
cd /hpc_atog/btho733/ABI/pacedSheep01/Anisotropic/functions/;showcs3(JR1);

%% Approach #2 : removing jitters and then doing kmeans 


clear;close all;clc;

len=100;  %change this for new image
for i=1:len
infile=sprintf('cut_%04d.tif',i-1);%outfile=sprintf('cut_%05d.tif',i-500);                       %change this for new image
cd /hpc_atog/btho733/ABI/JZ/Fiber_2016/Work_2017/CentralSection50um3/raw_25um3/bgdCorrect/;
im=imread(infile);
p1=im(101:240,61:160,1);
p2=im(101:240,61:160,2);
p3=im(101:240,61:160,3);
% p1=im(151:300,61:160,1);
% p2=im(151:300,61:160,2);
% p3=im(151:300,61:160,3);

D1(:,:,len-i+1,1)=p1;
D1(:,:,len-i+1,2)=p2;
D1(:,:,len-i+1,3)=p3;

disp(i);
end


% For Red plane

[Ny, Nx, Nz,k] = size(D1);
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
red=circular_block;

% For green plane

[Ny, Nx, Nz,k] = size(D1);
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI;
circular_block=squeeze(D1(:,:,:,2));
corrected_zplane=255*ones(Ny,Nx);upto=Nz-1;
for zplane=1:upto
    r=10;
    if(zplane==1)       
        im1=squeeze(D1(:,:,zplane,2));
        im2=squeeze(D1(:,:,zplane+1,2));
    else
        im1=corrected_zplane;
        im2=squeeze(D1(:,:,zplane+1,2));
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
green=circular_block;

% For blue plane

[Ny, Nx, Nz,k] = size(D1);
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI;
circular_block=squeeze(D1(:,:,:,3));
corrected_zplane=255*ones(Ny,Nx);upto=Nz-1;
for zplane=1:upto
    r=10;
    if(zplane==1)       
        im1=squeeze(D1(:,:,zplane,3));
        im2=squeeze(D1(:,:,zplane+1,3));
    else
        im1=corrected_zplane;
        im2=squeeze(D1(:,:,zplane+1,3));
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
blue=circular_block;

test_color(:,:,:,1)=red;
test_color(:,:,:,2)=green;
test_color(:,:,:,3)=blue;
%%
len=100;  %change this for new image

for i=1:len
% infile=sprintf('cut_%04d.tif',i-1);%outfile=sprintf('cut_%05d.tif',i-500);                       %change this for new image
cd /hpc_atog/btho733/ABI/JZ/Fiber_2016/Work_2017/CentralSection50um3/raw_25um3/bgdCorrect/;
im=squeeze(test_color(:,:,i,:));%imread(infile);
cd /hpc_atog/btho733/ABI/matlab_central/clustering;
[label_im,vec_mean] = kmeans_fast_Color_original(im,2,1);%kmeans_fast_Color(im(151:300,61:160,:),2,1);
D1_label(:,:,i)=label_im;


disp(i);
end
%%
cd /hpc_atog/btho733/ABI/pacedSheep01/Anisotropic;JR1=CoherenceFilter(D1_label,struct('T',20,'dt',2,'Scheme','R'));
cd /hpc_atog/btho733/ABI/pacedSheep01/Anisotropic/functions/;showcs3(JR1);
%%
cd /hpc_atog/btho733/ABI/pacedSheep01/Anisotropic;JR2=CoherenceFilter(red,struct('T',20,'dt',2,'Scheme','R'));
cd /hpc_atog/btho733/ABI/pacedSheep01/Anisotropic/functions/;showcs3(JR2);
%% normalise b/w 0 and 255, invert and levelset
for i=1:len
I=JR1(:,:,i);
mini=min(min(I));maxi=max(max(I));
Inorm=255.*((I-mini)./(maxi-mini));
% Ineg=255-Inorm;
mask = zeros(size(I));mask(2:end-2,2:end-2) = 1;
bw(:,:,i) = activecontour(Inorm,mask,300);
bw4show(:,:,i)=1-bw(:,:,i);
disp(i);
end
% save('geo1.mat','bw');
% bw = activecontour(Ineg, mask, 300, 'chan-vese','SmoothFactor',0);
% figure,imagesc(I);
% figure,imagesc(bw);
%%  Mat to tiff for kmeans_output (Binary case...NOT for greyscale)
clc;clear;close all;
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/clustering;
m=importdata('kmeans_output_02may18.mat');
[~,~,sliceno]=size(m);
for i=1:sliceno
name=sprintf('cut_%04d.tif',i-1);    
plane=uint8(squeeze(m(:,:,i)));
plane(plane==1)=0;
plane(plane==2)=255;
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/clustering/kmeans_output_tiffs;
imwrite(plane,name);
end

%%  Mat to tiff for thresh_sharp_output(Greyscale scenario)
clc;clear;close all;
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/clustering;
m=importdata('thresh_sharp_input_02may18.mat');
[~,~,sliceno]=size(m);
for i=1:sliceno
name=sprintf('cut_%04d.tif',i-1);    
plane=uint8(squeeze(m(:,:,i)));
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/clustering/mix_90_155_tifs/
imwrite(plane,name);
end

%%
clc;clear;close all;
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/clustering;
m=importdata('kmeans_output_02may18.mat');
n=importdata('thresh_sharp_output_02may18.mat');
[~,~,sliceno]=size(m);
parfor i=1:sliceno
    kmeanslice=squeeze(m(:,:,i));thr_slice=squeeze(n(:,:,i));
    kmeanslice(kmeanslice==1)=0;
    kmeanslice(kmeanslice==2)=255;
    cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/;
    mix(:,:,i)=mixtheoutputs(kmeanslice,thr_slice);
    
end
%%
cd /hpc_atog/btho733/ABI/pacedSheep01/Anisotropic;mix_AD=CoherenceFilter(mix,struct('T',20,'dt',2,'Scheme','R'));
cd /hpc_atog/btho733/ABI/pacedSheep01/Anisotropic/functions/;showcs3(mix_AD);
