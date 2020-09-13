            %#################################
            % LOADING MidSmallCut Data      %
            %#################################
%% UNFiltered Original images 
clear;close all;clc;
cd V:\ABI\JZ\Fiber_DTI\MidSmallCut\;
len=100;  %change this for new image
for i=1:len
infile=sprintf('cut_%04d.tif',i-1);                      
im=imread(infile);
D1(:,:,len-i+1,:)=im;

end
cd V:\ABI\pacedSheep01\Anisotropic\functions;
showcs3(single(D1));
%% Filtered Greyscale images 
clear;close all;clc;
cd V:\ABI\JZ\Fiber_DTI\MidSmallCut\a101_filtered_zplanes;
len=100;  %change this for new image
for i=1:len
infile=sprintf('z_%04d.tif',i-1);                      
im=imread(infile);
% im=im(y1:y2,x1:x2,:);
D1(:,:,len-i+1)=im;
end
cd V:\ABI\pacedSheep01\Anisotropic\functions;
showcs3(single(D1));
%% Contrast enhanced RGB image : Method 1
clear;close all;clc;
cd V:\ABI\JZ\Fiber_DTI\MidSmallCut\a104_contrast_enhanced_from_a102\Reversed\;
len=100;  %change this for new image
for i=1:len
infile=sprintf('z_%04d.tif',i-1);                     
im=imread(infile);
D1(:,:,i,:)=im;
end
cd V:\ABI\pacedSheep01\Anisotropic\functions;showcs3(single(D1));
%% Contrast enhanced RGB image : Method 2
clear;close all;clc;
cd V:\ABI\JZ\Fiber_DTI\MidSmallCut\a105_rgb_froma103\reversed\;
len=100;  %change this for new image
for i=1:len
infile=sprintf('z_%04d.tif',i-1);                      
im=imread(infile);
D1(:,:,i,:)=im;

end
cd V:\ABI\pacedSheep01\Anisotropic\functions;
showcs3(single(D1));
%%
            %#################################
            % LOADING Orientations from 3DST %
            %#################################

%% Cutting 3DST of midSmallcut from 3DST of Largeblock : InPlane

clear;close all;clc;
cd V:\ABI\JZ\Fiber_DTI\Large_block_25um_filtered\a301_ST3D_results\s1m3rho5k15\negyx;
len=200;  %change this for new image
for i=1:len
infile=sprintf('z_%04d.png',i-1);                     
im=imread(infile);
D1(:,:,len-i+1,:)=squeeze(im(101:250,11:400,:));
end
cd V:\ABI\pacedSheep01\Anisotropic\functions;showcs3(single(D1));

%% Cutting 3DST of midSmallcut from 3DST of Largeblock : CrossPlane Y

% clear;close all;clc;
cd V:\ABI\JZ\Fiber_DTI\Large_block_25um_filtered\a301_ST3D_results\s1m3rho5k15\negzx;
len=200;  %change this for new image
for i=1:len
infile=sprintf('y_%04d.png',i-1);                     
im=imread(infile);
D2(:,:,len-i+1,:)=squeeze(im(101:250,11:400,:));
end
cd V:\ABI\pacedSheep01\Anisotropic\functions;showcs3(single(D2));

%% Cutting 3DST of midSmallcut from 3DST of Largeblock : CrossPlane X

% clear;close all;clc;
cd V:\ABI\JZ\Fiber_DTI\Large_block_25um_filtered\a301_ST3D_results\s1m3rho5k15\negzy;
len=200;  %change this for new image
for i=1:len
infile=sprintf('x_%04d.png',i-1);                     
im=imread(infile);
D3(:,:,len-i+1,:)=squeeze(im(101:250,11:400,:));
end
cd V:\ABI\pacedSheep01\Anisotropic\functions;showcs3(single(D3));
%%  Loading 3DST of midSmallcut  : InPlane

% clear;close all;clc;
% cd V:\ABI\JZ\Fiber_DTI\MidSmallCut\a301_ST3D_results\s1m3rho5k15\revz;
cd V:\ABI\JZ\Fiber_DTI\MidSmallCut\a303_ST3D_results_EVLargest\s1m3rho5k25\revz;% Largest EV
len=100;  %change this for new image
for i=1:len
infile=sprintf('z_%04d.png',i-1);                     
im=imread(infile);
D4(:,:,i,:)=im;
end
cd V:\ABI\pacedSheep01\Anisotropic\functions;showcs3(single(D4));

%% Loading 3DST of midSmallcut : CrossPlane Y

% clear;close all;clc;
% cd V:\ABI\JZ\Fiber_DTI\MidSmallCut\a301_ST3D_results\s1m3rho5k15\revy;
cd V:\ABI\JZ\Fiber_DTI\MidSmallCut\a303_ST3D_results_EVLargest\s1m3rho5k25\revy;% Largest EV
len=100;  %change this for new image
for i=1:len
infile=sprintf('y_%04d.png',i-1);                     
im=imread(infile);
D5(:,:,i,:)=im;
end
cd V:\ABI\pacedSheep01\Anisotropic\functions;showcs3(single(D5));

%% Loading 3DST of midSmallcut  : CrossPlane X

% clear;close all;clc;
% cd V:\ABI\JZ\Fiber_DTI\MidSmallCut\a301_ST3D_results\s1m3rho5k15\revx;
cd V:\ABI\JZ\Fiber_DTI\MidSmallCut\a303_ST3D_results_EVLargest\s1m3rho5k25\revx;% Largest EV
len=100;  %change this for new image
for i=1:len
infile=sprintf('x_%04d.png',i-1);                     
im=imread(infile);
D6(:,:,i,:)=im;
end
cd V:\ABI\pacedSheep01\Anisotropic\functions;showcs3(single(D6));
%% PCA sample
data=rand(100,10);  % artificial data set of 100 variables (genes) and 10 samples
    [W, pc] = pca(data'); pc=pc'; W=W';
    plot(pc(1,:),pc(2,:),'.'); 
    title('{\bf PCA} by princomp'); xlabel('PC 1'); ylabel('PC 2')