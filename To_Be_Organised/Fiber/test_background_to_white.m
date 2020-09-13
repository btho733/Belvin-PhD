% clear; close all; clc;
% startno=1;endno=250;
% for i=startno:endno
% imname=sprintf('Im_351to600_%04d.tif',i-1);    
% cd /hpc/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/small_section_from_rgboutpng_351to600;im=imread(imname);cd ..;
% im1=im(:,:,1);im2=im(:,:,2);im3=im(:,:,3);
% [y,x]=size(im1);
% bin=ones(y,x);
% bin((im1==0)&(im2==0)&(im3==0))=0;
% rgbname=sprintf('elevz_%04d.png',i-1);cd /hpc/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/small_section_from_rgboutpng_351to600/results_s1m3rho10k50/new_rgb_elevz_absz/;rgb=imread(rgbname);cd ..;
% out1=rgb(:,:,1);out2=rgb(:,:,2);out3=rgb(:,:,3);
% out1(bin==0)=255;out2(bin==0)=255;out3(bin==0)=255;
% out(:,:,1)=out1;out(:,:,2)=out2;out(:,:,3)=out3;
% cd /hpc/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/small_section_from_rgboutpng_351to600/results_s1m3rho10k50/new_rgb_elevz_absz/bgdCorrect;imwrite(out,rgbname);cd ..;
% end
clear; close all; clc;
startno=1;endno=250;
for i=startno:endno
imname=sprintf('Im_351to600_%04d.tif',i-1);    
cd /hpc/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/small_section_from_rgboutpng_351to600;im=imread(imname);cd ..;
im1=im(:,:,1);im2=im(:,:,2);im3=im(:,:,3);
[y,x]=size(im1);
bin=ones(y,x);
bin((im1==0)&(im2==0)&(im3==0))=0;
mask(:,:,i)=bin;
disp(i);
end
cd /hpc/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/small_section_from_rgboutpng_351to600;
save('mask_small_section_from_rgboutpng_351to600.mat','mask','-v7.3');
exit;