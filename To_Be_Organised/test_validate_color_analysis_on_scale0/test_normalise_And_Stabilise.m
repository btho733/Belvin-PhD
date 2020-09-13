% %Loading 3D data
% %###############
clear;close all;clc;

cd /hpc_atog/btho733/ABI/JZ/Fiber_2016/Work_2017/CentralSection50um3/cut_for_elevation/%output_RH/;
% cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/filtered_images/;
for i=1:100
infile=sprintf('cut_%04d.tif',i);%outfile=sprintf('cut_%05d.tif',i-500);                       %change this for new image
im=imread(infile);
% im=im(y1:y2,x1:x2,:);
p1=im(:,:,1);
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
% then sum these average values, and set it as output value at (100, 200). 

    H = fspecial('disk',r);
    avg1 = imfilter(im1,H,'replicate'); 
    avg2 = imfilter(im2,H,'replicate'); 
    diff=avg2-avg1;
    corrected_zplane=im2-diff;
    circular_block(:,:,zplane+1)=corrected_zplane;
end

for i=1:Nz
%     outname=sprintf('Image_%04d.tif',i);
    p1(:,:,i)=squeeze(circular_block(:,:,i));
    p1_RGB(:,:,i,:)=uint8(ind2rgb(squeeze(p1(:,:,i)),255*summer(256)));
%     cd /hpc_atog/btho733/ABI/matlab_central/active_contour_without_edge/final/3d_seg/sections60forcheck/outputs/tifs/grey_filtered;
%     imwrite(p1,outname,'tif');
%     cd /hpc_atog/btho733/ABI/matlab_central/active_contour_without_edge/final/3d_seg/sections60forcheck/outputs/tifs/RGB_filtered/;
%     imwrite(p1_RGB,outname,'tif');
end
cd /hpc_atog/btho733/ABI/pacedSheep01/Anisotropic/functions;showcs3(single(p1_RGB));
showcs3(single(D1))
showcs3(single(p1))
%% For Tuning of r parameter (figure generation)
clear;close all;clc;
% cd /hpc/btho733/ABI/pacedSheep01/medsci_poster/stain_normalisation_toolbox/pacedsheepimages/f2v1/synth_op2/;  %change this for new image
% cd V:\ABI\pacedSheep01\medsci_poster\stain_normalisation_toolbox\pacedsheepimages\v8\output_RH\Nothreshold; 
% cd V:\ABI\JZ\Fiber_2016\Work_2017\CentralSection50um3\cut_for_elevation\bgdCorrect;
% cd /hpc_atog/btho733/ABI/JZ/Fiber_2016/Work_2017/CentralSection50um3/cut_for_elevation/bgdCorrect;
% cd /hpc_atog/btho733/ABI/JZ/Fiber_2016/Work_2017/CentralSection50um3/raw_25um3/bgdCorrect;
% cd /hpc_atog/btho733/ABI/JZ/Fiber_2016/Work_2017/CentralSection50um3/raw_50um3/bgdBlack;
% cd V:\ABI\JZ\Fiber_DTI\filtered_images;
cd V:\ABI\JZ\Fiber_2016\Work_2017\CentralSection50um3\cut_for_elevation\;
len=100;  %change this for new image
for i=1:len
infile=sprintf('cut_%04d.tif',i);%outfile=sprintf('cut_%05d.tif',i-500);                       %change this for new image
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

% For Y-plane (XZ)
im=squeeze(D1(80,:,:,:));
figure,imagesc(imgradient(im(:,2:end,1)));colormap(jet);axis off;


corrected_2=255*ones(180,100);
upto=99; 
figure;
% figure;title('Corrected planes at different window sizes');
for num=1:8
    r=num*2;
for column=1:upto
    if(column==1)
    im1(:,1)=squeeze(im(:,column,1));
    im1(:,2)=squeeze(im(:,column+1,1));
    else
    im1(:,1)=corrected_2(:,column);
    im1(:,2)=squeeze(im(:,column+1,1));  
    end
  
    m1=movmean(im1(:,1),r,1);
    m2=movmean(im1(:,2),r,1);
    diff=m1-m2;
    corrected_2(:,column+1)=double(im1(:,2))+diff;
end
corrected_r(:,:,num)=corrected_2;
rnum=sprintf('r = %d',r);
% figure,imagesc(squeeze(corrected_r(:,:,num)));colormap(jet);

subplot(4,2,num);
imagesc(squeeze(imgradient(corrected_r(:,2:end,num))));colormap(jet);title(rnum);axis off;
end


