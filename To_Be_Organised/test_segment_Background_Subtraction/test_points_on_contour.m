%%  Getting x-y coordinates from contour and draw a spline using those points.

%% Testing on sample data

% Getting x-y coordinates from contour 
%#####################################
clc;clear;close all;
z = peaks; % Load in a dataset that all matlab users have
x = 1:size(peaks, 2); y = 1:size(peaks, 1);
figure;pcolor(x, y, z); % Plot the data
hold on;shading flat
[cc, hh] = contour(x, y, z, [0 10^5], 'k'); % Overlay contour line
hold on;h = plot(cc(1,:), cc(2, :), 'w.'); % These are the x-y coordinates through the contour

% Now draw a spline using those points
% ####################################
figure;pcolor(x, y, z);  % make the same figure again
xy=cc(:,73:140); % choose all the points making a continuous curve
spcv=cscvn(xy);points=fnplt(spcv,'w',2);
hold on;plot(points(1,:),points(2,:),'w.','LineWidth',1.6);

%% Real testing on my data
clear;close all;clc;
for imno=593:660
cd /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/v1_cutsection_from_S/scale16;

imgname=sprintf('Scut_%05d.png',imno);
Img=imread(imgname);
cd /hpc/btho733/ABI/matlab_central/DRLSE_v0/;

demo_1
cd /hpc_atog/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction;

figure;imagesc(im); % Plot the data
hold on;  [cc,hh]=contour(phi, [0 0], 'r','LineWidth',2); % Overlay contour line
str=['Final zero level contour, ', num2str(iter_outer*iter_inner+iter_refine), ' iterations'];
title(str);
hold on;h = plot(cc(1,:), cc(2, :), 'w.'); % These are the x-y coordinates through the contour


% imno=590;
% Now draw a spline using those points
% ####################################
han=figure;imagesc(im); % make the same figure again
ucc=unique(cc','rows','stable');
ucct=ucc';
xy=ucct(:,5:end); % choose all the points making a continuous curve, omitting certain data points
spcv=cscvn(xy);points=fnplt(spcv,'w',2);
hold on;plot(points(1,:),points(2,:),'w.','LineWidth',1.6);
cd /hpc/btho733/ABI/matlab_central/DRLSE_v0/seg/;saveas(han,sprintf('FIG_%d.tif',imno));cd ..;
end
%%
mask=poly2mask(points(1,:),points(2,:),325,375);
figure,imagesc(mask);
NextInitialLSF=double(mask);
NextInitialLSF(mask==0)=c0;NextInitialLSF(mask==1)=-c0;
figure,imagesc(NextInitialLSF);

%% DRLSE for subsequent slices
for imno=591:610;
cd /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/v1_cutsection_from_S/scale16;
% Img=imread('gourd.bmp');
imgname=sprintf('Scut_%05d.png',imno);
Img=imread(imgname);
cd /hpc/btho733/ABI/matlab_central/DRLSE_v0;
% Img=imresize(Img,0.25);
Img=double(Img(:,:,1));

% parameter setting
timestep=1;  % time step
mu=0.2/timestep;  % coefficient of the distance regularization term R(phi)
iter_inner=10;
iter_outer=10;
lambda=5; % coefficient of the weighted length term L(phi)
alfa=-3;  % coefficient of the weighted area term A(phi)
epsilon=1.5; % papramater that specifies the width of the DiracDelta function

sigma=.8;    % scale parameter in Gaussian kernel
G=fspecial('gaussian',15,sigma); % Caussian kernel
Img_smooth=conv2(Img,G,'same');  % smooth image by Gaussiin convolution
[Ix,Iy]=gradient(Img_smooth);
f=Ix.^2+Iy.^2;
g=1./(1+f);  % edge indicator function.

% initialize LSF as binary step function
% c0=2;
% initialLSF =NextInitialLSF;
% % generate the initial region R0 as two rectangles
% initialLSF(145:205,170:245)=-c0; 
% initialLSF(25:35,40:50)=-c0;
phi=NextInitialLSF;

figure(1);
mesh(-phi);   % for a better view, the LSF is displayed upside down
hold on;  contour(phi, [0,0], 'r','LineWidth',2);
title('Initial level set function');
view([-80 35]);
cd /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/uu1_normalised_from_d_originals;

figure(2);
imname=sprintf('cut_%05d.png',imno);
im=imresize(imread(imname),0.25);
cd /hpc/btho733/ABI/matlab_central/DRLSE_v0;
imagesc(im); axis off; axis equal; colormap(gray); hold on;  contour(phi, [0,0], 'r');
title('Initial zero level contour');
pause(0.5);

potential=2;  
if potential ==1
    potentialFunction = 'single-well';  % use single well potential p1(s)=0.5*(s-1)^2, which is good for region-based model 
elseif potential == 2
    potentialFunction = 'double-well';  % use double-well potential in Eq. (16), which is good for both edge and region based models
else
    potentialFunction = 'double-well';  % default choice of potential function
end  

% start level set evolution
for n=1:iter_outer
    phi = drlse_edge(phi, g, lambda, mu, alfa, epsilon, timestep, iter_inner, potentialFunction);    
    if mod(n,2)==0
        figure(2);
        imagesc(im); axis off; axis equal; colormap(gray); hold on;  contour(phi, [0,0], 'r');
    end
end

% refine the zero level contour by further level set evolution with alfa=0
alfa=0;
iter_refine = 10;
phi = drlse_edge(phi, g, lambda, mu, alfa, epsilon, timestep, iter_inner, potentialFunction);

finalLSF=phi;
figure(2);

imagesc(im);
% axis off; axis equal; colormap(gray); 
hold on;  contour(phi, [0,0], 'r','LineWidth',2);
% hold on;  contour(phi, [0,0], 'r');
str=['Final zero level contour, ', num2str(iter_outer*iter_inner+iter_refine), ' iterations'];
title(str);

figure;
mesh(-finalLSF); % for a better view, the LSF is displayed upside down
hold on;  contour(phi, [0,0], 'r','LineWidth',2);
view([-80 35]);
str=['Final level set function, ', num2str(iter_outer*iter_inner+iter_refine), ' iterations'];
title(str);
axis on;
[nrow, ncol]=size(Img);
axis([1 ncol 1 nrow -5 5]);
set(gca,'ZTick',[-3:1:3]);
set(gca,'FontSize',14)


% Second section

cd /hpc_atog/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction;
% close all;
figure;imagesc(im); % Plot the data
hold on;  [cc,hh]=contour(phi, [0 0], 'r','LineWidth',2); % Overlay contour line
str=['Final zero level contour, ', num2str(iter_outer*iter_inner+iter_refine), ' iterations'];
title(str);
hold on;h = plot(cc(1,:), cc(2, :), 'w.'); % These are the x-y coordinates through the contour

% Now draw a spline using those points
% ####################################
han=figure;imagesc(im); % make the same figure again
ucc=unique(cc','rows','stable');
ucct=ucc';
xy=ucct(:,5:end); % choose all the points making a continuous curve, omitting certain data points
spcv=cscvn(xy);points=fnplt(spcv,'w',2);
hold on;plot(points(1,:),points(2,:),'w.','LineWidth',1.6);


cd /hpc/btho733/ABI/matlab_central/DRLSE_v0/seg/;saveas(han,sprintf('FIG_%d.tif',imno));cd ..;

mask=poly2mask(points(1,:),points(2,:),325,375);
% figure,imagesc(mask);
NextInitialLSF=double(mask);
NextInitialLSF(mask==0)=c0;NextInitialLSF(mask==1)=-c0;
% figure,imagesc(NextInitialLSF);
close all;
end

