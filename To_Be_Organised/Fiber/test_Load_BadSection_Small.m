            %#################################
            % LOADING BadSection_Small Data      %
            %#################################
%% UNFiltered Original images 
clear;close all;clc;
cd V:\ABI\JZ\Fiber_DTI\Badsection_small\;
len=70;  %change this for new image
for i=1:len
infile=sprintf('cut_%04d.tif',i-1);                      
im=imread(infile);
D1(:,:,len-i+1,:)=im;

end
cd V:\ABI\pacedSheep01\Anisotropic\functions;
showcs3(single(D1));
%% Filtered Greyscale images 
clear;close all;clc;
cd V:\ABI\JZ\Fiber_DTI\Badsection_small\a101_zplanes_corrected\;
len=70;  %change this for new image
for i=1:len
infile=sprintf('z_%04d.tif',i-1);                      
im=imread(infile);
D1(:,:,len-i+1)=im;

end
cd V:\ABI\pacedSheep01\Anisotropic\functions;
showcs3(single(D1));
%% Contrast enhancement : Mathod 1
cd V:\ABI\JZ\Fiber_DTI\Badsection_small\a101_zplanes_corrected\;
for u=1:70
    p1=squeeze(D1(:,:,u));
    p1_RGB=uint8(ind2rgb(p1,255*summer(256)));
    name=sprintf('z_%04d.tif',u-1);
    cd V:\ABI\JZ\Fiber_DTI\Badsection_small\a105_filtered_rgb_froma101; imwrite(p1_RGB,name);cd ..;
end
%% Contrast enhanced RGB image : Method 1
clear;close all;clc;
cd V:\ABI\JZ\Fiber_DTI\Badsection_small\a106_contrast_enhanced_rgb_froma105\;
len=70;  %change this for new image
for i=1:len
infile=sprintf('z_%04d.tif',i-1);                     
im=imread(infile);
D1(:,:,len-i+1,:)=im;
end
cd V:\ABI\pacedSheep01\Anisotropic\functions;showcs3(single(D1));
%% Contrast enhanced RGB image : Method 2
clear;close all;clc;
cd V:\ABI\JZ\Fiber_DTI\Badsection_small\a103_zplanes_froma101_contrast_enhanced_imagej\;
len=70;  %change this for new image
for i=1:len
infile=sprintf('z_%04d.tif',i-1);                      
im=imread(infile);
% im=im(y1:y2,x1:x2,:);
D1(:,:,len-i+1)=im;

end
cd V:\ABI\pacedSheep01\Anisotropic\functions;
showcs3(single(D1));
%%
            %#################################
            % LOADING Orientations from 3DST %
            %#################################

%%  Loading 3DST of midSmallcut  : InPlane

% clear;close all;clc;
cd V:\ABI\JZ\Fiber_DTI\Badsection_small\a301_ST3D_results\s1m3rho5k15\revz;
len=70;  %change this for new image
for i=1:len
infile=sprintf('z_%04d.png',i-1);                     
im=imread(infile);
D4(:,:,i,:)=im;
end
cd V:\ABI\pacedSheep01\Anisotropic\functions;showcs3(single(D4));

%% Loading 3DST of midSmallcut : CrossPlane Y

% clear;close all;clc;
cd V:\ABI\JZ\Fiber_DTI\Badsection_small\a301_ST3D_results\s1m3rho5k15\revy;
len=70;  %change this for new image
for i=1:len
infile=sprintf('y_%04d.png',i-1);                     
im=imread(infile);
D5(:,:,i,:)=im;
end
cd V:\ABI\pacedSheep01\Anisotropic\functions;showcs3(single(D5));

%% Loading 3DST of midSmallcut  : CrossPlane X

% clear;close all;clc;
cd V:\ABI\JZ\Fiber_DTI\Badsection_small\a301_ST3D_results\s1m3rho5k15\revx;
len=70;  %change this for new image
for i=1:len
infile=sprintf('x_%04d.png',i-1);                     
im=imread(infile);
D6(:,:,i,:)=im;
end
cd V:\ABI\pacedSheep01\Anisotropic\functions;showcs3(single(D6));
%% <<<  LATER to be updated for Badsection_small from Badsection_Large  >>>  Cutting 3DST of midSmallcut from 3DST of Largeblock : InPlane
% 
% clear;close all;clc;
% cd V:\ABI\JZ\Fiber_DTI\Large_block_25um_filtered\a301_ST3D_results\s1m3rho5k15\negyx;
% len=100;  %change this for new image
% for i=1:len
% infile=sprintf('z_%04d.png',i-1);                     
% im=imread(infile);
% D1(:,:,len-i+1,:)=squeeze(im(101:250,101:400,:));
% end
% cd V:\ABI\pacedSheep01\Anisotropic\functions;showcs3(single(D1));
% 
% %% Cutting 3DST of midSmallcut from 3DST of Largeblock : CrossPlane Y
% 
% % clear;close all;clc;
% cd V:\ABI\JZ\Fiber_DTI\Large_block_25um_filtered\a301_ST3D_results\s1m3rho5k15\negzx;
% len=100;  %change this for new image
% for i=1:len
% infile=sprintf('y_%04d.png',i-1);                     
% im=imread(infile);
% D2(:,:,len-i+1,:)=squeeze(im(101:250,101:400,:));
% end
% cd V:\ABI\pacedSheep01\Anisotropic\functions;showcs3(single(D2));
% 
% %% Cutting 3DST of midSmallcut from 3DST of Largeblock : CrossPlane X
% 
% % clear;close all;clc;
% cd V:\ABI\JZ\Fiber_DTI\Large_block_25um_filtered\a301_ST3D_results\s1m3rho5k15\negzy;
% len=100;  %change this for new image
% for i=1:len
% infile=sprintf('x_%04d.png',i-1);                     
% im=imread(infile);
% D3(:,:,len-i+1,:)=squeeze(im(101:250,101:400,:));
% end
% cd V:\ABI\pacedSheep01\Anisotropic\functions;showcs3(single(D3));
