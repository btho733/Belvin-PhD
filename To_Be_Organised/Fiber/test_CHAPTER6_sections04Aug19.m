%% BB
clear;close all;clc;

len=90;  %change this for new image
for i=1:len
infile=sprintf('cut_%04d.tif',i-1);%outfile=sprintf('cut_%05d.tif',i-500);                       %change this for new image
% cd V:\ABI\JZ\Fiber_2016\Work_2017\CentralSection50um3\raw_25um3\bgdCorrect\;
cd /hpc_atog/btho733/ABI/JZ/Fiber_2016/Work_2017/CentralSection50um3/raw_25um3/bgdCorrect;
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

cd /hpc_atog/btho733/ABI/pacedSheep01/Anisotropic;JR1=CoherenceFilter(red,struct('T',20,'dt',2,'Scheme','R'));



figure,
subplot(1,3,1),imagesc(squeeze(D1(:,70,:,1)));colormap(gray);daspect([1 1 1]);
subplot(1,3,2),imagesc(squeeze(red(:,70,:)));colormap(gray);daspect([1 1 1]);
subplot(1,3,3),imagesc(squeeze(JR1(:,70,:)));colormap(gray);daspect([1 1 1]);

figure,
subplot(1,3,1),imagesc(squeeze(D1(:,:,45,1)));colormap(gray);daspect([1 1 1]);
subplot(1,3,2),imagesc(squeeze(red(:,:,45)));colormap(gray);daspect([1 1 1]);
subplot(1,3,3),imagesc(squeeze(JR1(:,:,45)));colormap(gray);daspect([1 1 1]);

figure,
subplot(1,3,1),imagesc(squeeze(D1(40,:,:,1)));colormap(gray);daspect([1 1 1]);
subplot(1,3,2),imagesc(squeeze(red(40,:,:)));colormap(gray);daspect([1 1 1]);
subplot(1,3,3),imagesc(squeeze(JR1(40,:,:)));colormap(gray);daspect([1 1 1]);

% for i=1:90
%     imname=sprintf('cut_%05d.tif',91-i);
%     JRplane=uint8(JR1(:,:,i));
%     cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Chapter6_finalSections/tr2/06_New_AD_orig;
%     imwrite(JRplane,imname);
% end
figure,imagesc(squeeze(JR1(40,:,:)));colormap(gray);daspect([1 1 1]);
figure,imagesc(squeeze(JR1(:,70,:)));colormap(gray);daspect([1 1 1]);
figure,imagesc(squeeze(JR1(:,:,45)));colormap(gray);daspect([1 1 1]);
%%  RA Wall

clear;close all;clc;

len=100;  %change this for new image
for i=1:len
infile=sprintf('cut_%04d.tif',i-1);%outfile=sprintf('cut_%05d.tif',i-500);                       %change this for new image
% cd V:\ABI\JZ\Fiber_2016\Work_2017\CentralSection50um3\raw_25um3\bgdCorrect\;
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

cd /hpc_atog/btho733/ABI/pacedSheep01/Anisotropic;JR1=CoherenceFilter(red,struct('T',20,'dt',2,'Scheme','R'));

figure,
subplot(1,3,1),imagesc(squeeze(D1(:,50,:,1)));colormap(gray);daspect([1 1 1]);
subplot(1,3,2),imagesc(squeeze(red(:,50,:)));colormap(gray);daspect([1 1 1]);
subplot(1,3,3),imagesc(squeeze(JR1(:,50,:)));colormap(gray);daspect([1 1 1]);

figure,
subplot(1,3,1),imagesc(squeeze(D1(:,:,50,1)));colormap(gray);daspect([1 1 1]);
subplot(1,3,2),imagesc(squeeze(red(:,:,100)));colormap(gray);daspect([1 1 1]);
subplot(1,3,3),imagesc(squeeze(JR1(:,:,50)));colormap(gray);daspect([1 1 1]);

figure,
subplot(1,3,1),imagesc(squeeze(D1(70,:,:,1)));colormap(gray);daspect([1 1 1]);
subplot(1,3,2),imagesc(squeeze(red(70,:,:)));colormap(gray);daspect([1 1 1]);
subplot(1,3,3),imagesc(squeeze(JR1(70,:,:)));colormap(gray);daspect([1 1 1]);

figure,imagesc(squeeze(JR1(70,:,:)));colormap(gray);daspect([1 1 1]);
figure,imagesc(squeeze(JR1(:,50,:)));colormap(gray);daspect([1 1 1]);
figure,imagesc(squeeze(JR1(:,:,50)));colormap(gray);daspect([1 1 1]);
% for i=1:100
%     imname=sprintf('cut_%05d.tif',101-i);
%     JRplane=uint8(segw(:,:,i));
%     cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Chapter6_finalSections/tr1/09_New_segWhite_orig;
%     imwrite(JRplane,imname);
% end
%% Septum
clc;clear;close all;
offset=180;
for i=offset:239
    imname=sprintf('cut_%05d.tif',i);
    cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Chapter6_finalSections/s1/;
    im=imread(imname);
    imcut=im(65:164,7:106,:);
    D1(:,:,i-offset+1,:)=imcut;
end
% For Red plane

[Ny, Nx, Nz,k] = size(D1);
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI;
circular_block=squeeze(D1(:,:,:,1));

corrected_zplane=255*ones(Ny,Nx);upto=Nz-1;
for zplane=1:upto
    r=6;
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


cd /hpc_atog/btho733/ABI/pacedSheep01/Anisotropic;JR1=CoherenceFilter(red,struct('T',20,'dt',2,'Scheme','R'));
cd /hpc_atog/btho733/ABI/pacedSheep01/Anisotropic/functions/;showcs3(JR1);


figure,
subplot(1,3,1),imagesc(squeeze(D1(:,50,:,1)));colormap(gray);daspect([1 1 1]);
subplot(1,3,2),imagesc(squeeze(red(:,50,:)));colormap(gray);daspect([1 1 1]);
subplot(1,3,3),imagesc(squeeze(JR1(:,50,:)));colormap(gray);daspect([1 1 1]);

figure,
subplot(1,3,1),imagesc(squeeze(D1(:,:,30,1)));colormap(gray);daspect([1 1 1]);
subplot(1,3,2),imagesc(squeeze(red(:,:,30)));colormap(gray);daspect([1 1 1]);
subplot(1,3,3),imagesc(squeeze(JR1(:,:,30)));colormap(gray);daspect([1 1 1]);

figure,
subplot(1,3,1),imagesc(squeeze(D1(50,:,:,1)));colormap(gray);daspect([1 1 1]);
subplot(1,3,2),imagesc(squeeze(red(50,:,:)));colormap(gray);daspect([1 1 1]);
subplot(1,3,3),imagesc(squeeze(JR1(50,:,:)));colormap(gray);daspect([1 1 1]);

figure,imagesc(squeeze(JR1(50,:,:)));colormap(gray);daspect([1 1 1]);
figure,imagesc(squeeze(JR1(:,50,:)));colormap(gray);daspect([1 1 1]);
figure,imagesc(squeeze(JR1(:,:,30)));colormap(gray);daspect([1 1 1]);

figure,imagesc(squeeze(JR1(61,:,:)));colormap(gray);daspect([1 1 1]);
figure,imagesc(squeeze(JR1(:,74,:)));colormap(gray);daspect([1 1 1]);
figure,imagesc(squeeze(JR1(:,:,30)));colormap(gray);daspect([1 1 1]);
% for i=1:60
%     imname=sprintf('cut_%05d.tif',i);
%     JRplane=uint8(JR1(:,:,i));
%     cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Chapter6_finalSections/Septum_froms1_180to259/06_New_AD_orig;
%     imwrite(JRplane,imname);
% end

%% LA Wall
clc;clear;close all;
for i=1:60
    imname=sprintf('cut_%05d.tif',i);
    cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Chapter6_finalSections/s2/s2_seg2_segmented;
    im=imread(imname);
    imcut=im(:,80:160,:);
    D1(:,:,i,:)=imcut;
end
% For Red plane

[Ny, Nx, Nz,k] = size(D1);
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI;
circular_block=squeeze(D1(:,:,:,1));

corrected_zplane=255*ones(Ny,Nx);upto=Nz-1;
for zplane=1:upto
    r=6;
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


cd /hpc_atog/btho733/ABI/pacedSheep01/Anisotropic;JR1=CoherenceFilter(red,struct('T',20,'dt',2,'Scheme','R'));
cd /hpc_atog/btho733/ABI/pacedSheep01/Anisotropic/functions/;showcs3(JR1);


figure,
subplot(1,3,1),imagesc(squeeze(D1(:,50,:,1)));colormap(gray);daspect([1 1 1]);
subplot(1,3,2),imagesc(squeeze(red(:,50,:)));colormap(gray);daspect([1 1 1]);
subplot(1,3,3),imagesc(squeeze(JR1(:,50,:)));colormap(gray);daspect([1 1 1]);

figure,
subplot(1,3,1),imagesc(squeeze(D1(:,:,30,1)));colormap(gray);daspect([1 1 1]);
subplot(1,3,2),imagesc(squeeze(red(:,:,30)));colormap(gray);daspect([1 1 1]);
subplot(1,3,3),imagesc(squeeze(JR1(:,:,30)));colormap(gray);daspect([1 1 1]);

figure,
subplot(1,3,1),imagesc(squeeze(D1(114,:,:,1)));colormap(gray);daspect([1 1 1]);
subplot(1,3,2),imagesc(squeeze(red(114,:,:)));colormap(gray);daspect([1 1 1]);
subplot(1,3,3),imagesc(squeeze(JR1(114,:,:)));colormap(gray);daspect([1 1 1]);

figure,imagesc(squeeze(T(114,:,:)));colormap(jet);daspect([1 1 1]);
figure,imagesc(squeeze(T(:,50,:)));colormap(jet);daspect([1 1 1]);
figure,imagesc(squeeze(T(:,:,30)));colormap(jet);daspect([1 1 1]);

% for i=1:60
%     imname=sprintf('cut_%05d.tif',61-i);
%     JRplane=uint8(JR1(:,:,i));
%     cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Chapter6_finalSections/tr3_froms2/06_New_AD_orig;
%     imwrite(JRplane,imname);
% end