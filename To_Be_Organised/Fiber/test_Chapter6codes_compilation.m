%% Cut and save section
clc;close all;clear;
len=200; % s1: 350
offset=749;% s1: 549

parfor i=1:len
infile=sprintf('d03_overlaid_iso0_%04d.tif',i+offset);
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/Final_Fiber_Set_rgby/d03_Geometrysegmented_iso1;
% cd /hpc_atog/btho733/ABI/pacedSheep01/Dataset_Scale4/After_Amira/rgbseg_outputpng;                       %change this for new image
im=imread(infile);
% im=im(y1:y2,x1:x2,:);
s2=im(1251:1450,3301:3500,:);  % s1=im(861:1080,2166:2355,:);
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Chapter6_finalSections/s2;
savefile=sprintf('cut_%05d.tif',i);
imwrite(s2,savefile);
disp(i);
end
%% Cut and save fibers
clc;close all;clear;
len=351;
offset=655;

parfor i=1:len
    infile=sprintf('s4_%05d.png',i+offset);
cd /hpc_atog/btho733/ABI/pacedSheep01/Dataset_Scale4/montage/;
% infile=sprintf('d03_overlaid_iso0_%04d.tif',i+offset);
% cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/Final_Fiber_Set_rgby/d03_Geometrysegmented_iso1/;                  %change this for new image
im=imread(infile);
% im=im(y1:y2,x1:x2,:);
p1=im(801:1180,1901:2500,:);
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Chapter6_finalSections/s1/geo_largesection_unsegmented;
savefile=sprintf('cut_%04d.tif',i);
imwrite(p1,savefile);
disp(i);
end
%% Cut and save s2_unsegmented
clc;close all;clear;
len=200; % s1: 350
offset=856;% s1: 549

parfor i=1:len
infile=sprintf('s4_%05d.png',i+offset);
cd /hpc_atog/btho733/ABI/pacedSheep01/Dataset_Scale4/montage/;
% cd /hpc_atog/btho733/ABI/pacedSheep01/Dataset_Scale4/After_Amira/rgbseg_outputpng;                       %change this for new image
im=imread(infile);
% im=im(y1:y2,x1:x2,:);
s2=im(1251:1450,3301:3500,:);  % s1=im(861:1080,2166:2355,:);
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Chapter6_finalSections/s2/s2_unsegmented;
savefile=sprintf('cut_%05d.tif',i);
imwrite(s2,savefile);
disp(i);
end

%%  Overlay s2 smoothmasks
% clear; close all; clc;
startno=1;endno=100;
for i=startno:endno
    mname=sprintf('cut_%05d.tif',i);  
    cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Chapter6_finalSections/s2/s2_seg2_segmented/masks/;
    m=imread(mname);
%     m=te(:,:,i);
    imname=sprintf('cut_%04d.tif',i);  
    cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Chapter6_finalSections/s2/s2_seg2_segmented/grey_un/;
    im=imread(imname);
    im(m==255)=0;
%     im1=im(:,:,1);im2=im(:,:,2);im3=im(:,:,3);
%    im1(m==255)=0;im2(m==255)=0;im3(m==255)=0;
%    out(:,:,1)=im1;out(:,:,2)=im2;out(:,:,3)=im3;
   cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Chapter6_finalSections/s2/s2_seg2_segmented/grey_corrected;
   imwrite(im,imname);cd ..;
end
%% smooth3
clear; close all; clc;
startno=1;endno=100;
V=zeros(200,200,100);
for i=startno:endno
    imname=sprintf('mask_%04d.tif',i);  
     cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Chapter6_finalSections/s2/s2_seg1masks/;
    im=imread(imname);
    V(:,:,i)=im;
end
S11_1=smooth3(V,'gaussian',[51,51,51],1.5);
S3_1=smooth3(V,'gaussian',[3,3,3],1);
S7_1=smooth3(V,'gaussian',[7,7,7],1);
S11_3=smooth3(V,'gaussian',[11,11,11],3);
S3_3=smooth3(V,'gaussian',[3,3,3],3);
S7_3=smooth3(V,'gaussian',[7,7,7],3);

%% Normalisation
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Chapter6_finalSections/s2/s2_seg2_segmented/;
TargetImage = imread('cut_00047_template.tif');

parfor i=1:100    
Sourcename=sprintf('cut_%05d.tif',i);outfile=sprintf('cut_%04d.tif',i);
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Chapter6_finalSections/s2/s2_seg2_segmented/;
SourceImage = imread(Sourcename);
cd /hpc_atog/btho733/ABI/pacedSheep01/medsci_poster/stain_normalisation_toolbox/StainNormalisation;
% [ NormHS ] = Norm( SourceImage, TargetImage, 'RGBHist');
[ NormRH ] = Norm( SourceImage, TargetImage, 'Reinhard');
% [ NormMM ] = Norm(SourceImage, TargetImage, 'Macenko', 255, 0.15, 1);
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Chapter6_finalSections/s2/s2_seg2_segmented/;
% cd output_HS;imwrite(NormHS,outfile);cd ..;
cd output_RH;imwrite(NormRH,outfile);cd ..;
% cd output_MM;imwrite(NormMM,outfile);cd ..;
end
%%

clear;clc;
% cd /hpc/btho733/ABI/pacedSheep01/medsci_poster/stain_normalisation_toolbox/pacedsheepimages/f2v1/synth_op2/;  %change this for new image
% cd V:\ABI\JZ\Fiber_2016\Work_2017\CentralSection50um3\necrotic_scale4;
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Chapter6_finalSections/s2/s2_seg2_segmented/;
% cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/small_section_from_rgboutpng_351to600;
% cd V:\ABI\JZ\Fiber_2016\Work_2017\CentralSection50um3\cut_for_elevation\bgdCorrect;
% cd /hpc_atog/btho733/ABI/JZ/Fiber_2016/Work_2017/CentralSection50um3/raw_25um3/bgdCorrect;
% cd /hpc_atog/btho733/ABI/JZ/Fiber_2016/Work_2017/CentralSection50um3/raw_50um3/bgdBlack;
len=100;  %change this for new image
offset=0;
for i=1:len
%  cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Chapter6_finalSections/s2/s2_seg2_segmented/;   
infile=sprintf('cut_%05d.tif',i+offset);%outfile=sprintf('cut_%05d.tif',i-500);                       %change this for new image
im=imread(infile);
% im=im(y1:y2,x1:x2,:);
p1=im(:,:,1);
p2=im(:,:,2);
p3=im(:,:,3);
% m=zeros(200,200);
% m(p1==0 & p2==0 & p3==0)=255;
% cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Chapter6_finalSections/s2/s2_seg2_segmented/masks/;
% imwrite(m,infile);
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

% showcs3(single(circular_block));

% SAVING Grey Corrected

% for i=1:100
%     
%     outfile=sprintf('cut_%04d.tif',i);
%     im=squeeze(circular_block(:,:,i));
%     mname=sprintf('cut_%05d.tif',i);  
%     cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Chapter6_finalSectiongrays/s2/s2_seg2_segmented/masks/;
%     m=imread(mname);
%     im(m==255)=0;
%     cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Chapter6_finalSections/s2/s2_seg2_segmented/grey_corrected/;
%     imwrite(im,outfile);
% end

% SAVING RGB Corrected (PSEUDO-COLOURING)

% for i=1:100
%     
%     outfile=sprintf('cut_%04d.tif',i);
%     im=squeeze(circular_block(:,:,i));
%     im_RGB=uint8(ind2rgb(im,255*summer(256)));
%     im1=im_RGB(:,:,1);im2=im_RGB(:,:,2);im3=im_RGB(:,:,3);
%     mname=sprintf('cut_%05d.tif',i);  
%     cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Chapter6_finalSections/s2/s2_seg2_segmented/masks/;
%     m=imread(mname);
%     im1(m==255)=0;im2(m==255)=0;im3(m==255)=0;
%     out(:,:,1)=im1;out(:,:,2)=im2;out(:,:,3)=im3;
%     cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Chapter6_finalSections/s2/s2_seg2_segmented/rgb_corrected/;
%     imwrite(out,outfile);
% end

%% Attempt for smooth masks
clear;close all;clc;    
m=zeros(2490,4140,1060);
parfor i=1:1060
    infile=sprintf('d03_overlaid_iso0_%04d.tif',i);
    cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/Final_Fiber_Set_rgby/d03_Geometrysegmented_iso1/;
    im=imread(infile);
    im1=im(:,:,1);im2=im(:,:,2);im3=im(:,:,3);
    mi=squeeze(m(:,:,i));
    mi(im1==0 & im2==0 & im3==0)=255;
    m(:,:,i)=mi; 
    disp(i);
end
disp('smoothing...');

S11_1=smooth3(m,'gaussian',[11,11,11],1);
%% Overlaying z fibers
clear;close all;clc;    
m=zeros(2490,4140,1060);
for i=1:1060
    infile=sprintf('d03_overlaid_iso0_%04d.tif',i);
    cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/Final_Fiber_Set_rgby/d03_Geometrysegmented_iso1/;
    im=imread(infile);
    im1=im(:,:,1);im2=im(:,:,2);im3=im(:,:,3);
    mi=squeeze(m(:,:,i));
    mi(im1==0 & im2==0 & im3==0)=1;
    outfile=sprintf('z_%04d.tif',i);
    cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/Final_Fiber_Set_rgbz/a01_unsegmented/;
    out=imread(outfile);
    out1=out(:,:,1);out2=out(:,:,2);out3=out(:,:,3);
    out1(mi==1)=0;  out2(mi==1)=0;  out3(mi==1)=0; 
    plane(:,:,1)=out1;plane(:,:,2)=out2;plane(:,:,3)=out3;
    cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/Final_Fiber_Set_rgbz/d04_highresfibers/
    imwrite(plane,outfile);
end
%% Cut and save s2_unsegmented fibers
clc;close all;clear;
len=100; % s1: 350
offset=749;% s1: 549

parfor i=1:len
infile=sprintf('z_%04d.tif',i+offset);
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/Final_Fiber_Set_rgbz/a01_unsegmented/;
% cd /hpc_atog/btho733/ABI/pacedSheep01/Dataset_Scale4/After_Amira/rgbseg_outputpng;                       %change this for new image
im=imread(infile);
% im=im(y1:y2,x1:x2,:);
s2=im(1251:1450,3301:3500,:);  % s1=im(861:1080,2166:2355,:);
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Chapter6_finalSections/s2/s2_seg2_segmented/zfiber;
savefile=sprintf('cut_%04d.tif',i);
imwrite(s2,savefile);
disp(i);
end
%%   Overlay s2 smoothmasks on fibers
clear; close all; clc;
startno=1;endno=100;
for i=startno:endno
    mname=sprintf('cut_%05d.tif',i);  
    cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Chapter6_finalSections/s2/s2_seg2_segmented/masks/;
    m=imread(mname);
%     m=te(:,:,i);
    imname=sprintf('cut_%05d.tif',i);  
    cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Chapter6_finalSections/s2/s2_seg2_segmented/fiber/;
    im=imread(imname);
    im1=im(:,:,1);im2=im(:,:,2);im3=im(:,:,3);
   im1(m==255)=0;im2(m==255)=0;im3(m==255)=0;
   out(:,:,1)=im1;out(:,:,2)=im2;out(:,:,3)=im3;
   cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Chapter6_finalSections/s2/s2_seg2_segmented/fiber_seg/;
   imwrite(out,imname);cd ..;
end


%% STARTING TISSUE-NONTISSUE SEGMENTATION  (Code is adapted From test_01May18.m)
%##########################################

%% Section 1   --    Approach #2 : removing jitters and then doing kmeans 


% clear;close all;clc;

len=350;  %change this for new image
for i=1:len
infile=sprintf('cut_%05d.tif',i);%outfile=sprintf('cut_%05d.tif',i-500);                       %change this for new image
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Chapter6_finalSections/s1/%output_RH/;
im=imread(infile);
p1=im(:,:,1);
p2=im(:,:,2);
p3=im(:,:,3);
% p1=im(151:300,61:160,1);
% p2=im(151:300,61:160,2);
% p3=im(151:300,61:160,3);

D1(:,:,i,1)=p1; % if needed change to len-i+1
D1(:,:,i,2)=p2;
D1(:,:,i,3)=p3;

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
len=350;  %change this for new image

for i=1:len
% infile=sprintf('cut_%04d.tif',i-1);%outfile=sprintf('cut_%05d.tif',i-500);                       %change this for new image
im=squeeze(test_color(:,:,i,:));%imread(infile);
cd /hpc_atog/btho733/ABI/matlab_central/clustering;
[label_im,vec_mean] = kmeans_fast_Color_original(im,2,1);%kmeans_fast_Color(im(151:300,61:160,:),2,1);
D1_label(:,:,i)=label_im;


disp(i);
end
%%
cd /hpc_atog/btho733/ABI/pacedSheep01/Anisotropic;JR1=CoherenceFilter(D1_label,struct('T',20,'dt',2,'Scheme','R'));
cd /hpc_atog/btho733/ABI/pacedSheep01/Anisotropic/functions/;showcs3(JR1);
%% Looping and saving seg outputs

for plane=1:350
I=JR1(:,:,plane);
m = zeros(size(I,1),size(I,2));
m(10:size(I,1)-9,10:size(I,2)-9) = 1;
seg = chenvese(I,m,500,-1.8,'chan'); % ability on gray image
useg=uint8(seg);
useg(seg==1)=255;useg(seg==0)=0;
% useg=uint8(seg);
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Chapter6_finalSections/s1/Levelset_outs_neg1.8;
outname=sprintf('cut_%04d.png',plane);
imwrite(useg,outname);
disp(plane);
close all;
end

%% making mat of logicals for putting into propagation code

clc;clear;close all;
offset=165;start_prop=92;
for plane=offset:350
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Chapter6_finalSections/s1/Levelset_outs_neg1/rem_outlier_bright1.5/;
imname=sprintf('cut_%04d.png',plane);
i=imread(imname);
l=logical(i);
map(:,:,plane-offset+1)=l;
end
 clearvars -except map start_prop plane offset;
% cd /hpc_atog/btho733/ABI/matlab_central/FM_Tests/test6/fm2_matlab-master/examples;

% Adding to path the algorithms folder. This is not required if it is
% permanently included in the Matlab path.
p1=genpath('/hpc_atog/btho733/ABI/matlab_central/FM_Tests/test6');
    addpath(p1);    

p2=genpath('/hpc_atog/btho733/ABI/matlab_central/FM_Tests/test2');
    addpath(p2); 

% Parameters:
sat =0; % Between 0 and 1.
% figure,imagesc(map(:,:,start_prop));hold on;
% [xp,yp]=getpts();
%     xp=round(xp);yp=round(yp);
% Load map and adapt it
% map = imread('../data/map.bmp');
% Changing order of start and goal because of the (x,y) - (row,col)
 % correspondence.
start =[95;110;93];%[yp;xp;start_prop];%[100;127;92];%[119;98;1];%
goal = [59;36;plane-offset+1];%[46;72;23];%
W = double(map);
% Parameters setting
options.nb_iter_max = Inf;
options.Tmax        = sum(size(W));
options.end_points  = goal;
% tic;

[T,~] = perform_fast_marching(W, start, options);
% Plotting times-of-arrival map.

% toc;
% figure,imagesc(T(:,:,1));
A1 = unique(T);
out = A1(end-1);

T(T>out)=out;
% colormap jet;
% axis xy;
% axis image;
% axis off;
cd /hpc_atog/btho733/ABI/pacedSheep01/Anisotropic/functions;
showcs3(single(T(:,:,end:-1:1)));
%% colorbar;
rem=mod(length(gg),2);
if(rem==0)
gg=unique(T);
rgg=reshape(gg,length(gg)/2,2);
figure, imagesc(gg);colormap(jet);colorbar;
else
gg=unique(T);gg2=gg(2:end);
rgg=reshape(gg2,length(gg2)/2,2);
figure, imagesc(gg);colormap(jet);colorbar;
end
%% Inpainting, ST computation and Anisotropy table
Tnan=T;Tnan(T==max(max(max(T))))=NaN;
[~,~,l]=size(Tnan);
for i=1:l
  plane=squeeze(Tnan(:,:,i));  
  cd /hpc_atog/btho733/ABI/matlab_central/FM_Tests/test6/fm2_matlab-master/examples;
  inpainted_plane=inpaint_nans(plane);
  inpainted(:,:,i)=inpainted_plane;
end


JR=inpainted./(max(max(max(inpainted))));
sigma=1;
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/;
usigma=imgaussian(JR,sigma,3);

% Calculate the gradients
ux=derivatives(usigma,'x');
uy=derivatives(usigma,'y');
uz=derivatives(usigma,'z');
% [ux,uy,uz] = imgradientxyz(usigma,'sobel');
% Compute the 3D structure tensors J of the image
rho =50;
[Jxx, Jxy, Jxz, Jyy, Jyz, Jzz] = StructureTensor3D(ux,uy,uz,rho);

disp('Done Constructing tensor');

parametersDTI=[];
parametersDTI.BackgroundTreshold=0;
parametersDTI.WhiteMatterExtractionThreshold=0.0;
parametersDTI.textdisplay=true;

% Perform ST Analysis
[~,l1,l2,l3,~]=Prop_Parallel_testStructureFiber3D1(JR,Jxx, Jxy, Jxz, Jyy, Jyz, Jzz,parametersDTI);

disp('ST computation over');
o3=l3;o1=l1;o2=l2;
l3=o1;l1=o3;
l2=o1+o3-o2;
ul1=unique(l1);l1(l1==0)=ul1(2);%replace zeros in eigen values with next minimum value
ul2=unique(l2);l2(l2==0)=ul2(2);%replace zeros in eigen values with next minimum value
ul3=unique(l3);l3(l3==0)=ul3(2);%replace zeros in eigen values with next minimum value
[p,q,r]=size(JR);
r1=l1./l3;r2=l2./l3;r3=l1./l2;

mask=map;


mr1=r1.*single(mask);
mr2=r2.*single(mask);
mr3=r3.*single(mask);
% showcs3(single(mr3(:,end:-1:1,:)));


Avgr1=sum(sum(sum(mr1)))./sum(sum(sum(mask))); % Avgr1
Avgr2=sum(sum(sum(mr2)))./sum(sum(sum(mask))); % Avgr2
Avgr3=sum(sum(sum(mr3)))./sum(sum(sum(mask))); % Avgr3



fullavg1=sum(sum(sum(r1)))./(p*q*r); % Avgr1
fullavg2=sum(sum(sum(r2)))./(p*q*r); % Avgr2
fullavg3=sum(sum(sum(r3)))./(p*q*r); % Avgr3

cutr1=r1(1:end/2,:,:);cutr2=r2(1:end/2,:,:);cutr3=r3(1:end/2,:,:);
cutmr1=mr1(1:end/2,:,:);cutmr2=mr2(1:end/2,:,:);cutmr3=mr3(1:end/2,:,:);
cutmask=mask(1:end/2,:,:);[p,q,r]=size(cutmask);

cutAvgr1=sum(sum(sum(cutmr1)))./sum(sum(sum(cutmask))); % Avgr1
cutAvgr2=sum(sum(sum(cutmr2)))./sum(sum(sum(cutmask))); % Avgr2
cutAvgr3=sum(sum(sum(cutmr3)))./sum(sum(sum(cutmask))); % Avgr3



cutfullavg1=sum(sum(sum(cutr1)))./(p*q*r); % Avgr1
cutfullavg2=sum(sum(sum(cutr2)))./(p*q*r); % Avgr2
cutfullavg3=sum(sum(sum(cutr3)))./(p*q*r); % Avgr3


[p,q,r]=size(JR);
FAv=(1/sqrt(2)).*( sqrt((l1-l2).^2+(l2-l3).^2+(l3-l1).^2)./sqrt(l1.^2+l2.^2+l3.^2) );
cl=(l1-l2)./(l1+l2+l3);cp=2.*(l2-l3)./(l1+l2+l3);cs=3*l3./(l1+l2+l3);
clw=(l1-l2)./l1;cpw=(l2-l3)./l1;csw=l3./l1;
MA=((0.5*(-l1-l2+2*l3).*(2*l1-l2-l3).*(-l1+2.*l2-l3))./((l1.^2+l2.^2+l3.^2-l1.*l2-l2.*l3-l3.*l1).^1.5));
dti=(2.*l1)./(l2+l3);
disp('done');
AvgFAv=sum(sum(sum(FAv)))./(p*q*r);%AvgFAv
AvgMode=sum(sum(sum(MA)))./(p*q*r);%AvgMode
Avgcl=sum(sum(sum(cl)))./(p*q*r);%Avgcl
Avgcp=sum(sum(sum(cp)))./(p*q*r);%Avgcp
Avgcs=sum(sum(sum(cs)))./(p*q*r);%Avgcs
Avgclw=sum(sum(sum(clw)))./(p*q*r);%Avgclw
Avgcpw=sum(sum(sum(cpw)))./(p*q*r);%Avgcpw
Avgcsw=sum(sum(sum(csw)))./(p*q*r);%Avgcsw
Avgdti=sum(sum(sum(dti)))./(p*q*r);%dti


%% Section 2   --    Approach #2 : removing jitters and then doing kmeans 


clear;close all;clc;

len=100;  %change this for new image
for i=1:len
infile=sprintf('cut_%04d.tif',i);%outfile=sprintf('cut_%05d.tif',i-500);                       %change this for new image
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Chapter6_finalSections/s2/s2_seg2_segmented/output_RH/;
im=imread(infile);
p1=im(:,:,1);
p2=im(:,:,2);
p3=im(:,:,3);
% p1=im(151:300,61:160,1);
% p2=im(151:300,61:160,2);
% p3=im(151:300,61:160,3);

D1(:,:,i,1)=p1; % if needed change to len-i+1
D1(:,:,i,2)=p2;
D1(:,:,i,3)=p3;

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
new_test_color=test_color(:,80:160,i,:);
for i=1:60
    mname=sprintf('cut_%05d.tif',i);
    cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Chapter6_finalSections/s2/s2_seg2_segmented/masks/;
    m=imread(mname);
    m=m(:,80:160);
    test_color_slice=squeeze(test_color(:,80:160,i,:));
    t1=test_color_slice(:,:,1);t2=test_color_slice(:,:,2);t3=test_color_slice(:,:,3);
    t1(m==255)=243;t2(m==255)=202;t3(m==255)=25;
    new_test_color(:,:,i,1)=t1;new_test_color(:,:,i,2)=t2;new_test_color(:,:,i,3)=t3;
end

%%
len=60;  %change this for new image

for i=1:len
% infile=sprintf('cut_%04d.tif',i-1);%outfile=sprintf('cut_%05d.tif',i-500);                       %change this for new image
im=squeeze(new_test_color(:,:,i,:));%imread(infile);
cd /hpc_atog/btho733/ABI/matlab_central/clustering;
[label_im,vec_mean] = kmeans_fast_Color_original(im,2,1);%kmeans_fast_Color(im(151:300,61:160,:),2,1);
D1_label(:,:,i)=label_im;
disp(i);
end
%%
cd /hpc_atog/btho733/ABI/pacedSheep01/Anisotropic;JR2=CoherenceFilter(red,struct('T',20,'dt',2,'Scheme','R'));
cd /hpc_atog/btho733/ABI/pacedSheep01/Anisotropic/functions/;showcs3(single(JR2(:,160:-1:80,1:60,:)));
% Thr=zeros(200,200,100);
% Thr(JR1>1.25)=255;
% showcs3(single(Thr));
%% Looping and saving seg outputs

for plane=1:60
I=JR1(:,:,plane);
m = zeros(size(I,1),size(I,2));
m(10:size(I,1)-9,10:size(I,2)-9) = 1;
seg = chenvese(I,m,500,-1,'chan'); % ability on gray image
useg=uint8(seg);
useg(seg==1)=255;useg(seg==0)=0;
% useg=uint8(seg);
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Chapter6_finalSections/s2/s2_seg2_segmented/Levelset_outs_neg1
outname=sprintf('cut_%04d.png',plane);
imwrite(useg,outname);
disp(plane);
close all;
end

%% making mat of logicals for putting into propagation code

clc;clear;close all;
for plane=1:60
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Chapter6_finalSections/s2/s2_seg2_segmented/Levelset_outs_0.001/;
imname=sprintf('cut_%04d.png',plane);
i=imread(imname);
l=logical(i);
map(:,:,plane)=l;
end
 clearvars -except map;
% cd /hpc_atog/btho733/ABI/matlab_central/FM_Tests/test6/fm2_matlab-master/examples;

% Adding to path the algorithms folder. This is not required if it is
% permanently included in the Matlab path.
p1=genpath('/hpc_atog/btho733/ABI/matlab_central/FM_Tests/test6');
    addpath(p1);    

p2=genpath('/hpc_atog/btho733/ABI/matlab_central/FM_Tests/test2');
    addpath(p2); 

% Parameters:
sat =0; % Between 0 and 1.
% figure,imagesc(map(:,:,start_prop));hold on;
% [xp,yp]=getpts();
%     xp=round(xp);yp=round(yp);
% Load map and adapt it
% map = imread('../data/map.bmp');
% Changing order of start and goal because of the (x,y) - (row,col)
 % correspondence.
start =[105;40;30];%[yp;xp;start_prop];%[20;76;60];%[181;3;23];
goal = [195;1;1];%[46;72;23];
W = double(map);
% Parameters setting
options.nb_iter_max = Inf;
options.Tmax        = sum(size(W));
options.end_points  = goal;

% tic;
[T,~] = perform_fast_marching(W, start, options);
% Plotting times-of-arrival map.
% toc;

% figure,imagesc(T(:,:,1));
A1 = unique(T);
out = A1(end-1);

T(T>out)=out;
% colormap jet;
% axis xy;
% axis image;
% axis off;
cd /hpc_atog/btho733/ABI/pacedSheep01/Anisotropic/functions;
% cd V:\ABI\pacedSheep01\Anisotropic\functions;
showcs3(single(T(:,end:-1:1,:)));
%% colorbar;
rem=mod(length(gg),2);
if(rem==0)
gg=unique(T);
rgg=reshape(gg,length(gg)/2,2);
figure, imagesc(gg);colormap(jet);colorbar;
else
gg=unique(T);gg2=gg(2:end);
rgg=reshape(gg2,length(gg2)/2,2);
figure, imagesc(gg);colormap(jet);colorbar;
end
%% Inpainting, ST computation and Anisotropy table
Tnan=T;Tnan(T==max(max(max(T))))=NaN;
[~,~,l]=size(Tnan);
for i=1:l
  plane=squeeze(Tnan(:,:,i));  
  cd /hpc_atog/btho733/ABI/matlab_central/FM_Tests/test6/fm2_matlab-master/examples;
  inpainted_plane=inpaint_nans(plane);
  inpainted(:,:,i)=inpainted_plane;
end


JR=inpainted./(max(max(max(inpainted))));
sigma=1;
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/;
usigma=imgaussian(JR,sigma,3);

% Calculate the gradients
ux=derivatives(usigma,'x');
uy=derivatives(usigma,'y');
uz=derivatives(usigma,'z');
% [ux,uy,uz] = imgradientxyz(usigma,'sobel');
% Compute the 3D structure tensors J of the image
rho =60;
[Jxx, Jxy, Jxz, Jyy, Jyz, Jzz] = StructureTensor3D(ux,uy,uz,rho);

disp('Done Constructing tensor');

parametersDTI=[];
parametersDTI.BackgroundTreshold=0;
parametersDTI.WhiteMatterExtractionThreshold=0.0;
parametersDTI.textdisplay=true;

% Perform ST Analysis
[~,l1,l2,l3,~]=Prop_Parallel_testStructureFiber3D1(JR,Jxx, Jxy, Jxz, Jyy, Jyz, Jzz,parametersDTI);

disp('ST computation over');
o3=l3;o1=l1;o2=l2;
l3=o1;l1=o3;
l2=o1+o3-o2;
ul1=unique(l1);l1(l1==0)=ul1(2);%replace zeros in eigen values with next minimum value
ul2=unique(l2);l2(l2==0)=ul2(2);%replace zeros in eigen values with next minimum value
ul3=unique(l3);l3(l3==0)=ul3(2);%replace zeros in eigen values with next minimum value
[p,q,r]=size(JR);
r1=l1./l3;r2=l2./l3;r3=l1./l2;

for i=1:60
      mname=sprintf('cut_%05d.tif',i);
      cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Chapter6_finalSections/s2/s2_seg2_segmented/masks/;
    m=imread(mname);
    mask(:,:,i)=(255-m(:,80:160))./255;
end
mr1=r1.*single(mask);
mr2=r2.*single(mask);
mr3=r3.*single(mask);
% showcs3(single(mr3(:,end:-1:1,:)));
Avgr1=sum(sum(sum(mr1)))./sum(sum(sum(mask))); % Avgr1
Avgr2=sum(sum(sum(mr2)))./sum(sum(sum(mask))); % Avgr2
Avgr3=sum(sum(sum(mr3)))./sum(sum(sum(mask))); % Avgr3



% AvgTable(iter,1)=sum(sum(sum(r1)))./(p*q*r); % Avgr1
% AvgTable(iter,2)=sum(sum(sum(r2)))./(p*q*r); % Avgr2
% AvgTable(iter,3)=sum(sum(sum(r3)))./(p*q*r); % Avgr3
% 
% FAv=(1/sqrt(2)).*( sqrt((l1-l2).^2+(l2-l3).^2+(l3-l1).^2)./sqrt(l1.^2+l2.^2+l3.^2) );
% cl=(l1-l2)./(l1+l2+l3);cp=2.*(l2-l3)./(l1+l2+l3);cs=3*l3./(l1+l2+l3);
% clw=(l1-l2)./l1;cpw=(l2-l3)./l1;csw=l3./l1;
% MA=((0.5*(-l1-l2+2*l3).*(2*l1-l2-l3).*(-l1+2.*l2-l3))./((l1.^2+l2.^2+l3.^2-l1.*l2-l2.*l3-l3.*l1).^1.5));
% dti=(2.*l1)./(l2+l3);
% disp('done');
% AvgTable(iter,4)=sum(sum(sum(FAv)))./(p*q*r);%AvgFAv
% AvgTable(iter,5)=sum(sum(sum(MA)))./(p*q*r);%AvgMode
% AvgTable(iter,6)=sum(sum(sum(cl)))./(p*q*r);%Avgcl
% AvgTable(iter,7)=sum(sum(sum(cp)))./(p*q*r);%Avgcp
% AvgTable(iter,8)=sum(sum(sum(cs)))./(p*q*r);%Avgcs
% AvgTable(iter,9)=sum(sum(sum(clw)))./(p*q*r);%Avgclw
% AvgTable(iter,10)=sum(sum(sum(cpw)))./(p*q*r);%Avgcpw
% AvgTable(iter,11)=sum(sum(sum(csw)))./(p*q*r);%Avgcsw
% AvgTable(iter,12)=sum(sum(sum(dti)))./(p*q*r);%dti


%%
clear;close all;clc;
for i=1:60
      mname=sprintf('cut_%05d.tif',i);
      cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Chapter6_finalSections/s2/s2_seg2_segmented/masks/;
    m=imread(mname);
    mask=(255-m(:,80:160))./255;
    cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Chapter6_finalSections/s2/s2_seg2_segmented/zfiber/;
       unsegname=sprintf('cut_%04d.tif',i);
       u=imread(unsegname);
    unseg1=squeeze(u(:,80:160,1)); unseg2=squeeze(u(:,80:160,2)); unseg3=squeeze(u(:,80:160,3));
    unseg1(mask==0)=0;  unseg2(mask==0)=0;  unseg3(mask==0)=0;
    out(:,:,1)=unseg1; out(:,:,2)=unseg2; out(:,:,3)=unseg3;
    cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Chapter6_finalSections/tr3_froms2/testcut1_fiber_rgbz/;
    imwrite(out,unsegname);
%     name=sprintf('cut_%04d.tif',i);
% %     nc=red(:,80:160,i);
% %     c=red(:,:,i);
%     cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Chapter6_finalSections/s2/s2_seg2_segmented/testcut1_output_RH
%     imwrite(m,name);
%     cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Chapter6_finalSections/s1/corrected;
%     imwrite(c,name);
end

%% Section #1

clear;close all;clc;

len=90;  %change this for new image
xoffset=0;%1600;
yoffset=0;%880;
zoffset=0;%394;
for i=1:len
infile=sprintf('cut_%04d.tif',i-1);
% infile=sprintf('d03_overlaid_iso0_%04d.tif',i+zoffset);                       %change this for new image
% cd V:\ABI\JZ\Fiber_2016\Work_2017\CentralSection50um3\raw_25um3\bgdCorrect\;
% cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/Final_Fiber_Set_rgby/d03_Geometrysegmented_iso1;
cd /hpc_atog/btho733/ABI/JZ/Fiber_2016/Work_2017/CentralSection50um3/raw_25um3/bgdCorrect/;
im=imread(infile);
p1=im(yoffset+163:yoffset+242,xoffset+372:xoffset+511,1);
p2=im(yoffset+163:yoffset+242,xoffset+372:xoffset+511,2);
p3=im(yoffset+163:yoffset+242,xoffset+372:xoffset+511,3);
% p1=im(151:300,61:160,1);
% p2=im(151:300,61:160,2);
% p3=im(151:300,61:160,3);

D1(:,:,i,1)=p1;
D1(:,:,i,2)=p2;
D1(:,:,i,3)=p3;

disp(i);
end


% For Red plane

[Ny, Nx, Nz,k] = size(D1);
% cd V:\ABI\JZ\Fiber_DTI;
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/;
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
% cd V:\ABI\JZ\Fiber_DTI;
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/;
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
% cd V:\ABI\JZ\Fiber_DTI;
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/;
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

% cd V:\ABI\pacedSheep01\Anisotropic\functions\
% showcs3(single(test_color))
% showcs3(single(D1))
%%
len=90;  %change this for new image

for i=1:len
% infile=sprintf('cut_%04d.tif',i-1);%outfile=sprintf('cut_%05d.tif',i-500);                       %change this for new image
im=squeeze(test_color(:,:,i,:));%imread(infile);
cd /hpc_atog/btho733/ABI/matlab_central/clustering;
[label_im,vec_mean] = kmeans_fast_Color_original(im,2,1);%kmeans_fast_Color(im(151:300,61:160,:),2,1);
D1_label(:,:,i)=label_im;


disp(i);
end
%%
cd /hpc_atog/btho733/ABI/pacedSheep01/Anisotropic;JR1=CoherenceFilter(D1_label,struct('T',20,'dt',2,'Scheme','R'));
cd /hpc_atog/btho733/ABI/pacedSheep01/Anisotropic/functions/;showcs3(single(JR1(:,:,end:-1:1,:)))
%%
figure,vol3d('cdata',JR1,'texture','3D');view(3);colormap(summer);
%%

clear;close all;clc;

len=90;  %change this for new image
xoffset=1600;
yoffset=880;
zoffset=393;
for i=1:len
% infile=sprintf('cut_%04d.tif',i-1);
infile=sprintf('d02_overlaid_iso0_%04d.tif',i+zoffset);                       %change this for new image
% cd V:\ABI\JZ\Fiber_2016\Work_2017\CentralSection50um3\raw_25um3\bgdCorrect\;
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/Final_Fiber_Set_rgby/d02_segmented_iso1/
% cd /hpc_atog/btho733/ABI/JZ/Fiber_2016/Work_2017/CentralSection50um3/raw_25um3/bgdCorrect/;
im=imread(infile);
f=im(yoffset+163:yoffset+242,xoffset+372:xoffset+511,:);
fname=sprintf('cut_%04d.tif',i);
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Chapter6_finalSections/tr2/fiber_rgby/;
imwrite(f,fname);
disp(i);
end
%%
for i=1:90
f=squeeze(red(:,:,i));
fname=sprintf('cut_%04d.tif',i);
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Chapter6_finalSections/tr2/red_grayscale_geo_4AMIRA/
imwrite(f,fname);
end

%%  Overlay s2 smoothmasks
% clear; close all; clc;
startno=1;endno=351;
for i=startno:endno
    mname=sprintf('cut_%04d.tif',i);  
    cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Chapter6_finalSections/s1/geo_largesection_smoothMasks/;
    m=imread(mname);
%     m=te(:,:,i);
    imname=sprintf('cut_%04d.tif',i);  
    cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Chapter6_finalSections/s1/geo_largesection_unsegmented/;
    im=imread(imname);
%     im(m==0)=0;
  im1=im(:,:,1);im2=im(:,:,2);im3=im(:,:,3);
    im1(m==0)=0;im2(m==0)=0;im3(m==0)=0;
   out(:,:,1)=im1;out(:,:,2)=im2;out(:,:,3)=im3;
   cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Chapter6_finalSections/s1/geo_largesection_seg;
   imwrite(out,imname);cd ..;
end

%% Making f02 folder

clear; close all; clc;
startno=1;endno=100;
for i=startno:endno
    mname=sprintf('Scale10_Levelset_%04d.tif',i);  
    cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/Final_Fiber_Set_rgby/f01_Scale10_LevelSet;
    m=imread(mname);
     cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/Final_Fiber_Set_rgby/f03_Scale10
   imwrite(out,mname);cd ..;
%     m=te(:,:,i);
   [y,x]=size(m);
    mask=zeros(y,x);    
    mask(m>0)=1; %tissue
   
    imname=sprintf('Image_%05d.tif',i+10);  
    cd /hpc_atog/btho733/ABI/matlab_central/active_contour_without_edge/final/scale10/iso250um/selection;
    im=imread(imname);
%     im(m==0)=0;
  im1=im(:,:,1);im2=im(:,:,2);im3=im(:,:,3);
    im1(m==0)=0;im2(m==0)=0;im3(m==0)=0;
   out(:,:,1)=im1;out(:,:,2)=im2;out(:,:,3)=im3;
   cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/Final_Fiber_Set_rgby/f02_Scale10_Levelset_Color;
   imwrite(out,mname);cd ..;
end

%% Septum block angle figure -3D block, fiber block and angles generation
clc;close all;clear;
cd /hpc/btho733/ABI/JZ/Fiber_DTI/Chapter6_finalSections/s1/01_corrected;
offset=165;
for i=offset:350
imname=sprintf('cut_%04d.tif',i);
V(:,:,i-offset+1)=imread(imname);
end
cd /hpc_atog/btho733/ABI/pacedSheep01;
D = squeeze(V(:,end:-1:1,end:-1:1));
h = vol3d('cdata',D,'texture','3D');
view(3);  
axis tight;colormap(gray);
%% above section for local
clc;close all;clear;
cd V:\ABI\JZ\Fiber_DTI\Chapter6_finalSections\s1\01_corrected;
offset=180;
for i=offset:350
imname=sprintf('cut_%04d.tif',i);
im=imread(imname);V(:,:,i-offset+1)=im(1:190,1:190);
end
cd V:\ABI\pacedSheep01;
D = squeeze(V(:,end:-1:1,end:-1:1));
h = vol3d('cdata',D,'texture','3D');
view(-194,30); colormap(gray);daspect([1,1,190/(350-offset)]);
axis tight;
ax = gca;
ax.BoxStyle = 'full';
box(ax,'on'); 
camproj('perspective')
%% fiber block
clc;close all;clear;
cd V:\ABI\JZ\Fiber_DTI\Chapter6_finalSections\s1\02_fiber;
offset=180;
for i=offset:350
imname=sprintf('fiber_%04d.tif',i);
im=imread(imname);V(:,:,i-offset+1,:)=im(1:190,1:190,:);
end
cd V:\ABI\pacedSheep01;
D = squeeze(V(:,end:-1:1,end:-1:1,:));
h = vol3d('cdata',D,'texture','3D');
view(-194,30); colormap(gray);daspect([1,1,190/(350-offset)]);
axis tight;
ax = gca;
ax.BoxStyle = 'full';
box(ax,'on'); 

camproj('perspective')
%% BB-  3D Block


clear;close all;clc;

len=90;  %change this for new image
for i=1:len
infile=sprintf('cut_%04d.tif',i-1);%outfile=sprintf('cut_%05d.tif',i-500);                       %change this for new image
cd V:\ABI\JZ\Fiber_2016\Work_2017\CentralSection50um3\raw_25um3\bgdCorrect\;
% cd /hpc_atog/btho733/ABI/JZ/Fiber_2016/Work_2017/CentralSection50um3/raw_25um3/bgdCorrect;
im=imread(infile);
p1=im(163:242,372:511,1);
p2=im(163:242,372:511,2);
p3=im(163:242,372:511,3);
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

V=red;
cd V:\ABI\pacedSheep01;
D = squeeze(V(:,end:-1:1,:));figure,
h = vol3d('cdata',D,'texture','3D');
view(-194,30); colormap(gray);daspect([80/140,1,90/140]);
axis tight;
ax = gca;
ax.BoxStyle = 'full';
box(ax,'on');
camproj('perspective')
%% BB-fiber block
clc;close all;clear;
cd V:\ABI\JZ\Fiber_DTI\Chapter6_finalSections\tr2\fiber_rgby\;
for i=1:90
imname=sprintf('cut_%04d.tif',i);
im=imread(imname);V(:,:,90-i+1,:)=im;
end
cd V:\ABI\pacedSheep01;
D = squeeze(V(:,end:-1:1,:,:));
h = vol3d('cdata',D,'texture','3D');
view(-194,30); colormap(gray);daspect([80/140,1,90/140]);
axis tight;
ax = gca;
ax.BoxStyle = 'full';
box(ax,'on'); 
camproj('perspective')


%% RA Smooth wall- 3D Block


clear;close all;clc;

len=100;  %change this for new image
for i=1:len
infile=sprintf('cut_%04d.tif',i-1);%outfile=sprintf('cut_%05d.tif',i-500);                       %change this for new image
cd V:\ABI\JZ\Fiber_2016\Work_2017\CentralSection50um3\raw_25um3\bgdCorrect\;
% cd /hpc_atog/btho733/ABI/JZ/Fiber_2016/Work_2017/CentralSection50um3/raw_25um3/bgdCorrect/;
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

V=red;
cd V:\ABI\pacedSheep01;
D = squeeze(V(:,end:-1:1,:));
figure,
h = vol3d('cdata',D,'texture','3D');
view(-194,30); colormap(gray);daspect([140/100,1,1]);
axis tight;
ax = gca;
ax.BoxStyle = 'full';
box(ax,'on'); 
camproj('perspective')
%% RA Smooth Wall-fiber block
clc;close all;clear;
cd V:\ABI\JZ\Fiber_DTI\Chapter6_finalSections\tr1\fiber_rgby\;
for i=1:100
imname=sprintf('cut_%04d.tif',i);
im=imread(imname);V(:,:,100-i+1,:)=im;
end
cd V:\ABI\pacedSheep01;
D = squeeze(V(:,end:-1:1,:,:));
h = vol3d('cdata',D,'texture','3D');
view(-194,30); colormap(gray);daspect([140/100,1,1]);
axis tight;
ax = gca;
ax.BoxStyle = 'full';
box(ax,'on'); 
camproj('perspective')
