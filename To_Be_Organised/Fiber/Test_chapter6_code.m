%% Loading 3D data
clear;close all;clc;
% cd /hpc/btho733/ABI/pacedSheep01/medsci_poster/stain_normalisation_toolbox/pacedsheepimages/f2v1/synth_op2/;  %change this for new image
% cd V:\ABI\pacedSheep01\medsci_poster\stain_normalisation_toolbox\pacedsheepimages\v8\output_RH\Nothreshold; 
% cd V:\ABI\JZ\Fiber_2016\Work_2017\CentralSection50um3\cut_for_elevation\bgdCorrect;
cd /hpc_atog/btho733/ABI/JZ/Fiber_2016/Work_2017/CentralSection50um3/cut_for_elevation/bgdCorrect;
% cd /hpc_atog/btho733/ABI/JZ/Fiber_2016/Work_2017/CentralSection50um3/raw_25um3/bgdCorrect;
% cd /hpc_atog/btho733/ABI/JZ/Fiber_2016/Work_2017/CentralSection50um3/raw_50um3/bgdBlack;
len=100;  %change this for new image
for i=1:len
infile=sprintf('cut_%04d.tif',i-1);%outfile=sprintf('cut_%05d.tif',i-500);                       %change this for new image
im=imread(infile);
% im=im(y1:y2,x1:x2,:);
% p1=im(90:end,90:end,1);
% p2=im(90:end,90:end,2);
% p3=im(90:end,90:end,3);
p1=im(:,:,1);
p2=im(:,:,2);
p3=im(:,:,3);

D1(:,:,i,1)=p1;
D1(:,:,i,2)=p2;
D1(:,:,i,3)=p3;
end

[Ny, Nx, Nz,k] = size(D1);
clear p1;clear p2;clear p3;
%% Loading 3D data
clear;close all;clc;
% cd /hpc/btho733/ABI/pacedSheep01/medsci_poster/stain_normalisation_toolbox/pacedsheepimages/f2v1/synth_op2/;  %change this for new image
% cd V:\ABI\pacedSheep01\medsci_poster\stain_normalisation_toolbox\pacedsheepimages\v8\output_RH\Nothreshold; 
% cd V:\ABI\JZ\Fiber_2016\Work_2017\CentralSection50um3\cut_for_elevation\bgdCorrect;
cd /hpc_atog/btho733/ABI/JZ/Fiber_2016/Work_2017/CentralSection50um3/cut_for_elevation/output_RH;
% cd /hpc_atog/btho733/ABI/JZ/Fiber_2016/Work_2017/CentralSection50um3/raw_25um3/bgdCorrect;
% cd /hpc_atog/btho733/ABI/JZ/Fiber_2016/Work_2017/CentralSection50um3/raw_50um3/bgdBlack;
len=100;  %change this for new image
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
end

[Ny, Nx, Nz,k] = size(D1);
clear p1;clear p2;clear p3;
%%

% For Red plane

[Ny, Nx, Nz,k] = size(D1);
cd V:\ABI\JZ\Fiber_DTI;
% cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/;
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
cd V:\ABI\JZ\Fiber_DTI;
% cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/;
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
cd V:\ABI\JZ\Fiber_DTI;
% cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/;
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


%% Loading 3D data
% clear;close all;clc;
% cd /hpc/btho733/ABI/pacedSheep01/medsci_poster/stain_normalisation_toolbox/pacedsheepimages/f2v1/synth_op2/;  %change this for new image
% cd V:\ABI\pacedSheep01\medsci_poster\stain_normalisation_toolbox\pacedsheepimages\v8\output_RH\Nothreshold; 
% cd V:\ABI\JZ\Fiber_2016\Work_2017\CentralSection50um3\cut_for_elevation\bgdCorrect;
% cd V:\ABI\JZ\Fiber_DTI\filtered_images;
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/filtered_images;
% cd /hpc_atog/btho733/ABI/JZ/Fiber_2016/Work_2017/CentralSection50um3/raw_25um3/bgdCorrect;
% cd /hpc_atog/btho733/ABI/JZ/Fiber_2016/Work_2017/CentralSection50um3/raw_50um3/bgdBlack;
len=100;  %change this for new image
for i=1:len
infile=sprintf('cut_%04d.tif',i);%outfile=sprintf('cut_%05d.tif',i-500);                       %change this for new image
im=imread(infile);
% im=im(y1:y2,x1:x2,:);
p1=im(:,:,1);
p2=im(:,:,2);
p3=im(:,:,3);

D2(:,:,i,1)=p1;
D2(:,:,i,2)=p2;
D2(:,:,i,3)=p3;
end

[Ny, Nx, Nz,k] = size(D2);
clear p1;clear p2;clear p3;

%%
clear;close all;clc;
cd V:\ABI\JZ\Fiber_DTI\filtered_images\contrast_enhanced;
len=100;  %change this for new image
for i=1:len
infile=sprintf('contrast_enhanced_%04d.tif',i-1);%outfile=sprintf('cut_%05d.tif',i-500);                       %change this for new image
im=imread(infile);
% im=im(y1:y2,x1:x2,:);
p1=im(:,:,1);
p2=im(:,:,2);
p3=im(:,:,3);

D2(:,:,i,1)=p1;
D2(:,:,i,2)=p2;
D2(:,:,i,3)=p3;
end

[Ny, Nx, Nz,k] = size(D2);
clear p1;clear p2;clear p3;

%%
% cd V:\ABI\pacedSheep01\Anisotropic\;
D4=squeeze(D2(:,:,:,1));
cd /hpc_atog/btho733/ABI/pacedSheep01/Anisotropic;
JR = CoherenceFilter(D4,struct('T',20,'dt',2,'Scheme','R'));