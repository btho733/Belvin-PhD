            %#################################
            % LOADING LargeBlock Data      %
            %#################################
%% UNFiltered Original images 
clear;close all;clc;
cd V:\ABI\JZ\Fiber_2016\Work_2017\CentralSection50um3\raw_25um3\bgdCorrect;
len=200;  %change this for new image
for i=1:len
infile=sprintf('cut_%04d.tif',i-1);                      
im=imread(infile);
D(:,:,len-i+1,:)=im;
D1=squeeze(D(:,:,:,1));D2=squeeze(D(:,:,:,2));D3=squeeze(D(:,:,:,3));
end
cd V:\ABI\pacedSheep01\Anisotropic\functions;
showcs3(single(Dnew));
%% Filtered Greyscale images 
% clear;close all;clc;
cd V:\ABI\JZ\Fiber_DTI\Large_block_25um_filtered\Zplanes\grey\;
len=200;  %change this for new image
for i=1:len
infile=sprintf('z_%04d.tif',i-1);                      
im=imread(infile);
Df(:,:,len-i+1)=im;

end
cd V:\ABI\pacedSheep01\Anisotropic\functions;
showcs3(single(Df));
% Converting Df to Dark background using the segmented mask(D1,D2,D3)
% Dark=Df;Dark((D1==255)&(D2==255)&(D3==255))=0;
% Saving Dark (Images with Dark background)
cd V:\ABI\JZ\Fiber_DTI\Large_block_25um_filtered\Zplanes\;
for u=1:200
    p1=squeeze(Df(:,:,u));
    p1_RGB=uint8(ind2rgb(p1,255*summer(256)));
    name=sprintf('z_%04d.tif',u-1);
%     cd grey; imwrite(p1,name);cd ..;
    cd rgb; imwrite(p1_RGB,name);cd ..;
end


% clear;close all;clc;
% cd V:\ABI\JZ\Fiber_DTI\MidSmallCut\a101_filtered_zplanes;
% len=100;  %change this for new image
% for i=1:len
% infile=sprintf('z_%04d.tif',i-1);                      
% im=imread(infile);
% % im=im(y1:y2,x1:x2,:);
% D1(:,:,len-i+1)=im;
% end
% cd V:\ABI\pacedSheep01\Anisotropic\functions;
% showcs3(single(D1));
% %% Contrast enhanced RGB image : Method 1
% clear;close all;clc;
% cd V:\ABI\JZ\Fiber_DTI\MidSmallCut\a104_contrast_enhanced_from_a102\Reversed\;
% len=100;  %change this for new image
% for i=1:len
% infile=sprintf('z_%04d.tif',i-1);                     
% im=imread(infile);
% D1(:,:,i,:)=im;
% end
% cd V:\ABI\pacedSheep01\Anisotropic\functions;showcs3(single(D1));
% %% Contrast enhanced RGB image : Method 2
% clear;close all;clc;
% cd V:\ABI\JZ\Fiber_DTI\MidSmallCut\a105_rgb_froma103\reversed\;
% len=100;  %change this for new image
% for i=1:len
% infile=sprintf('z_%04d.tif',i-1);                      
% im=imread(infile);
% D1(:,:,i,:)=im;
% 
% end
% cd V:\ABI\pacedSheep01\Anisotropic\functions;
% showcs3(single(D1));
%%
            %#################################
            % LOADING Orientations from 3DST %
            %#################################

%% Cutting 3DST of Largeblock : InPlane

% clear;close all;clc;
cd V:\ABI\JZ\Fiber_DTI\Large_block_25um_filtered\a301_ST3D_results\s1m3rho5k25\revz;
% cd V:\ABI\JZ\Fiber_DTI\Large_block_25um_filtered\a302_ST3D_results_EVmiddle\s1m3rho5k25\revz\;
% cd V:\ABI\JZ\Fiber_DTI\Large_block_25um_filtered\a303_ST3D_results_EVLargest\s1m3rho5k25\revz\;
len=200;  %change this for new image
for i=1:len
infile=sprintf('z_%04d.png',i-1);                     
im=imread(infile);
D1(:,:,i,:)=im;
end
cd V:\ABI\pacedSheep01\Anisotropic\functions;showcs3(single(D1));

%% Cutting 3DST of Largeblock : CrossPlane Y

% clear;close all;clc;
% cd V:\ABI\JZ\Fiber_DTI\Large_block_25um_filtered\a301_ST3D_results\s1m3rho5k15\negzx;
% cd V:\ABI\JZ\Fiber_DTI\Large_block_25um_filtered\a302_ST3D_results_EVmiddle\s1m3rho5k25\revy\;
cd V:\ABI\JZ\Fiber_DTI\Large_block_25um_filtered\a303_ST3D_results_EVLargest\s1m3rho5k25\revy\;
len=200;  %change this for new image
for i=1:len
infile=sprintf('y_%04d.png',i-1);                     
im=imread(infile);
D2(:,:,i,:)=im;
end
cd V:\ABI\pacedSheep01\Anisotropic\functions;showcs3(single(D2));

%% Cutting 3DST of Largeblock : CrossPlane X

% clear;close all;clc;
% cd V:\ABI\JZ\Fiber_DTI\Large_block_25um_filtered\a301_ST3D_results\s1m3rho5k15\negzy;
% cd V:\ABI\JZ\Fiber_DTI\Large_block_25um_filtered\a302_ST3D_results_EVmiddle\s1m3rho5k25\revx\;
cd V:\ABI\JZ\Fiber_DTI\Large_block_25um_filtered\a303_ST3D_results_EVLargest\s1m3rho5k25\revx\;
len=200;  %change this for new image
for i=1:len
infile=sprintf('x_%04d.png',i-1);                     
im=imread(infile);
D3(:,:,i,:)=im;
end
cd V:\ABI\pacedSheep01\Anisotropic\functions;showcs3(single(D3));

