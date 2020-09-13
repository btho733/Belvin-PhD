clc;clear; close all;
len=1064;
for i=1:len
imname=sprintf('z_%04d.tif',i);outname=sprintf('z_%04d.tif',i);
% cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/small_section_from_rgboutpng_351to600/results_s1m3rho5k25/rgby;
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/Final_Fiber_Set_rgbz/a01_unsegmented;
im=imread(imname);
mname=sprintf('rgb_%05d.png',i);
cd /hpc_atog/btho733/ABI/pacedSheep01/Dataset_Scale4/After_Amira/rgbseg_outputpng;m=imread(mname);
mask=ones(2490,4140);
m1=m(:,:,1);m2=m(:,:,2);m3=m(:,:,3);
mask(m1==0 & m2==0 & m3==0)=0;
im1=im(:,:,1);im2=im(:,:,2);im3=im(:,:,3);
im1(mask==0)=0;im2(mask==0)=0;im3(mask==0)=0;
seg(:,:,1)=im1;seg(:,:,2)=im2;seg(:,:,3)=im3;
% segr=imresize(seg,0.2);
% cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/small_section_from_rgboutpng_351to600/results_s1m3rho5k25/Final_351to600_namecorrected/scale0/;
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/Final_Fiber_Set_rgbz/a02_segmented_using_Masks_rgboutput_png/;
imwrite(seg,outname,'tif');
% cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/small_section_from_rgboutpng_351to600/results_s1m3rho5k25/rgby_seg_scale5;
% imwrite(segr,outname,'tif');
disp(i);
end

%% Renaming

clc;clear;close all;
len=125;offset=0;
parfor i=1:len
oldname=sprintf('z_%04d.png',i-1);
newname=sprintf('z_%04d.tif',i+offset);
% cd V:\ABI\JZ\Fiber_DTI\Whole_Atria\small_section_from_rgboutpng_Par200\results_from76\rgby;
% cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/Final_Fiber_Set_rgby/unsegmented/png;
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/small_section_from_rgboutpng_Par200/results_from_1/rgbz;
im=imread(oldname);
% cd V:\ABI\JZ\Fiber_DTI\Whole_Atria\small_section_from_rgboutpng_Par200\results_from76\rgby\renamed;
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/small_section_from_rgboutpng_Par200/results_from_1/rgbz/tif;
imwrite(im,newname,'tif');
end

%% Z Visualisation using showcs3
clc;clear;close all;
parfor i= 1:212
    cd V:\ABI\JZ\Fiber_DTI\Whole_Atria\Final_Fiber_Set_rgbx\a03_s5_nointerp\noflip;
    imname=sprintf('s5x_nointerp_%04d.tif',i);
    im=imread(imname);
    V(:,:,i,:)=im;
end
cd V:\ABI\pacedSheep01\Anisotropic\functions\;showcs3(single(V(:,:,end:-1:1,:)))
%% Y Visualisation using showcs3
clc;clear;close all;
parfor i= 1:212
    cd V:\ABI\JZ\Fiber_DTI\Whole_Atria\Final_Fiber_Set_rgby\a03_s5_nointerp\noflip;
    imname=sprintf('s5_nointerp_%04d.tif',i);
    im=imread(imname);
    V(:,:,i,:)=im;
end
cd V:\ABI\pacedSheep01\Anisotropic\functions\;showcs3(single(V(:,:,end:-1:1,:)))
%% Highres_Y Visualisation using showcs3
clc;clear;close all;
offset=0;
parfor i= 1:1060
    cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/Final_Fiber_Set_rgbx/a02_segmented_using_Masks_rgboutput_png;
%     cd V:\ABI\JZ\Fiber_DTI\Whole_Atria\Final_Fiber_Set_rgby\a02_segmented_using_Masks_rgboutput_png;
    imname=sprintf('x_%04d.tif',i+offset);
    im=imread(imname);
    V(:,:,i,:)=im;
    disp(i);
end
disp('done');
% figure,imagesc(squeeze(V(1350,end:-1:1,:,:)))
% cd V:\ABI\pacedSheep01\Anisotropic\functions\;showcs3(single(V(:,:,end:-1:1,:)))

%% 
clear;clc;close all;

cd V:\ABI\JZ\Fiber_DTI\Whole_Atria\Final_Fiber_Set_rgby\a02_segmented_using_Masks_rgboutput_png\
im=imread('y_0600.tif');
figure,imagesc(im);hold on;
[x,y]=getpts();
x=round(x);y=round(y);
% x=[2595;2631];y=[1374;1738];
% x=[256;306];y=[3453;3370];
m=(y(2)-y(1))/(x(2)-x(1));
b=y(1)-m*x(1);
xcords=min(x):max(x);
clear x;clear y;
ycords=round(m.*xcords+b);
cords=[xcords',ycords'];ucords=unique(cords,'rows','stable');
x=ucords(:,1);y=ucords(:,2);
plot(x,y,'w*');
imhsv=rgb2hsv(im);
h=imhsv(:,:,1);
angle=180*h-90; % converting from [0,1] to [-90,90]
yaxis=diag(angle(y,x));
xaxis=1:length(xcords);
figure,plot(xaxis,yaxis);
% figure,plot(xaxis,smooth(yaxis,3));
% figure,plot(xaxis,sgolayfilt(yaxis,3,11));
%%  Recreating the Interesting location close to LA roof (RGBZ)
% First load all rgbz highres images to V; Then run this
%  Later, Investigate in  all 3 views-rgbz,rgbx and rgby 
% showcs3(single(V(501:1000,2801:3500,450:-1:201,:)))  
im=squeeze(V(501:1000,2801:3500,400,:));
figure,imagesc(squeeze(V(501:1000,2801:3500,400,:)));
x=[172;401];y=[316;421];
m=(y(2)-y(1))/(x(2)-x(1));
b=y(1)-m*x(1);
xcords=min(x):max(x);
clear x;clear y;
ycords=round(m.*xcords+b);
cords=[xcords',ycords'];ucords=unique(cords,'rows','stable');
x=ucords(:,1);y=ucords(:,2);
plot(x,y,'w*');
imhsv=rgb2hsv(im);
h=imhsv(:,:,1);
angle=180*h-90; % converting from [0,1] to [-90,90]
yaxis=diag(angle(y,x));
xaxis=1:length(xcords);
figure,plot(xaxis,yaxis);ylim([-90,90]);
% figure,plot(xaxis,smooth(yaxis,3));
% figure,plot(xaxis,sgolayfilt(yaxis,3,11));


figure,imagesc(squeeze(V(801:1176,2801:3300,390,:)))
figure,imagesc(squeeze(V(801:1176,2801:3100,390,:)));



%% CT vertical angle
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/Final_Fiber_Set_rgby/e01_All_yplanes/;
  plane=695;
  imname=sprintf('y_%04d.tif',plane);
  im=imread(imname);
  figure,imagesc(im);hold on;
 
    
    x=[820;884];y=[731;739];
  m=(y(2)-y(1))/(x(2)-x(1));
b=y(1)-m*x(1);
xcords=min(x):max(x);
clear x;clear y;
ycords=round(m.*xcords+b);
cords=[xcords',ycords'];ucords=unique(cords,'rows','stable');
x=ucords(:,1);y=ucords(:,2);
plot(x,y,'w*');
imhsv=rgb2hsv(im);
h=imhsv(:,:,1);
angle=180*h-90; % converting from [0,1] to [-90,90]
yaxis=diag(angle(y,x));
xaxis=1:length(xcords);
figure,plot(xaxis,yaxis);ylim([-90,90]);
% set(gca,'xdir','reverse');

%% BB

close all;
  cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/Final_Fiber_Set_rgby/e01_All_yplanes/;
  plane=212*5;
  imname=sprintf('y_%04d.tif',plane);
  im=imread(imname);
  figure,imagesc(im);hold on;
   
   x=[1831;2365];y=[428;431];
  m=(y(2)-y(1))/(x(2)-x(1));
b=y(1)-m*x(1);
xcords=min(x):max(x);
clear x;clear y;
ycords=round(m.*xcords+b);
cords=[xcords',ycords'];ucords=unique(cords,'rows','stable');
x=ucords(:,1);y=ucords(:,2);
plot(x,y,'w*');
imhsv=rgb2hsv(im);
h=imhsv(:,:,1);
angle=180*h-90; % converting from [0,1] to [-90,90]
yaxis=diag(angle(y,x));
xaxis=1:length(xcords);
figure,plot(xaxis,yaxis);ylim([-90,90]);
%% Vertical bundles in Septum Vs  BB 
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/Final_Fiber_Set_rgby/e01_All_yplanes/;
  plane=225*5;
  imname=sprintf('y_%04d.tif',plane);
  im=imread(imname);
  figure,imagesc(im);hold on;
x=[1627;2438];
y=[395;440];
  m=(y(2)-y(1))/(x(2)-x(1));
b=y(1)-m*x(1);
xcords=min(x):max(x);
clear x;clear y;
ycords=round(m.*xcords+b);
cords=[xcords',ycords'];ucords=unique(cords,'rows','stable');
x=ucords(:,1);y=ucords(:,2);
plot(x,y,'w*');
imhsv=rgb2hsv(im);
h=imhsv(:,:,1);
angle=180*h-90; % converting from [0,1] to [-90,90]
yaxis=diag(angle(y,x));
xaxis=1:length(xcords);
figure,plot(xaxis,yaxis);ylim([-90,90]);
% set(gca,'xdir','reverse');
%%   Sudden angle shift in Lower Septum #1
 cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/Final_Fiber_Set_rgby/e01_All_yplanes/;
  plane=225*5;
  imname=sprintf('y_%04d.tif',plane);
  im=imread(imname);
  figure,imagesc(im);hold on;
x=[1850;2130];y=[643;671];
  m=(y(2)-y(1))/(x(2)-x(1));
b=y(1)-m*x(1);
xcords=min(x):max(x);
clear x;clear y;
ycords=round(m.*xcords+b);
cords=[xcords',ycords'];ucords=unique(cords,'rows','stable');
x=ucords(:,1);y=ucords(:,2);
plot(x,y,'w*');
imhsv=rgb2hsv(im);
h=imhsv(:,:,1);
angle=180*h-90; % converting from [0,1] to [-90,90]
yaxis=diag(angle(y,x));
xaxis=1:length(xcords);
figure,plot(xaxis,yaxis);ylim([-90,90]);
% set(gca,'xdir','reverse');
%%   Sudden angle shift in Lower Septum #2
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/Final_Fiber_Set_rgby/e01_All_yplanes/;
  plane=231*5;
  imname=sprintf('y_%04d.tif',plane);
  im=imread(imname);
  figure,imagesc(im);hold on;
 
    x=[1812;2062];y=[626;648];
  m=(y(2)-y(1))/(x(2)-x(1));
b=y(1)-m*x(1);
xcords=min(x):max(x);
clear x;clear y;
ycords=round(m.*xcords+b);
cords=[xcords',ycords'];ucords=unique(cords,'rows','stable');
x=ucords(:,1);y=ucords(:,2);
plot(x,y,'w*');
imhsv=rgb2hsv(im);
h=imhsv(:,:,1);
angle=180*h-90; % converting from [0,1] to [-90,90]
yaxis=diag(angle(y,x));
xaxis=1:length(xcords);
figure,plot(xaxis,yaxis);ylim([-90,90]);
% set(gca,'xdir','reverse');

%% Superior Left Septum vertical angle
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/Final_Fiber_Set_rgby/e01_All_yplanes/;
  plane=173*5;
  imname=sprintf('y_%04d.tif',plane);
  im=imread(imname);
  figure,imagesc(im);hold on;
 
    
    x=[2267;2324];y=[455;434];
  m=(y(2)-y(1))/(x(2)-x(1));
b=y(1)-m*x(1);
xcords=min(x):max(x);
clear x;clear y;
ycords=round(m.*xcords+b);
cords=[xcords',ycords'];ucords=unique(cords,'rows','stable');
x=ucords(:,1);y=ucords(:,2);
plot(x,y,'w*');
imhsv=rgb2hsv(im);
h=imhsv(:,:,1);
angle=180*h-90; % converting from [0,1] to [-90,90]
yaxis=diag(angle(y,x));
xaxis=1:length(xcords);
figure,plot(xaxis,yaxis);ylim([-90,90]);
% set(gca,'xdir','reverse');

%%  LA Free Wall.....Recreating the first angle shift result (RGBY)
% First load all rgby highres images to V; Then run this

im=squeeze(V(1350,:,end:-1:1,:));
% s5 slice #270
x=[256;306];y=[3453;3370];
m=(y(2)-y(1))/(x(2)-x(1));
b=y(1)-m*x(1);
xcords=min(x):max(x);
clear x;clear y;
ycords=round(m.*xcords+b);
cords=[xcords',ycords'];ucords=unique(cords,'rows','stable');
x=ucords(:,1);y=ucords(:,2);
figure,imagesc(im);hold on;
plot(x,y,'w');view(-90,90);
imhsv=rgb2hsv(im);
h=imhsv(:,:,1);
angle=180*h-90; % converting from [0,1] to [-90,90]
yaxis=diag(angle(y,x));
xaxis=1:length(xcords);
figure,plot(xaxis,yaxis);ylim([-90,90]);
% set(gca,'xdir','reverse');

% The LA view of this interesting portion showcs3(single(V(:,3001:4000,950:-1:750,:)))
% 3D view of the exact location           showcs3(single(V(1251:1450,3301:3500,950:-1:750,:)))
%% LA Wall

close all;
  cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/Final_Fiber_Set_rgby/e01_All_yplanes/;
  plane=212*5;
  imname=sprintf('y_%04d.tif',plane);
  im=imread(imname);
  figure,imagesc(im);hold on;
   
   x=[3709;3775];y=[505;498];
  m=(y(2)-y(1))/(x(2)-x(1));
b=y(1)-m*x(1);
xcords=min(x):max(x);
clear x;clear y;
ycords=round(m.*xcords+b);
cords=[xcords',ycords'];ucords=unique(cords,'rows','stable');
x=ucords(:,1);y=ucords(:,2);
plot(x,y,'w*');
imhsv=rgb2hsv(im);
h=imhsv(:,:,1);
angle=180*h-90; % converting from [0,1] to [-90,90]
yaxis=diag(angle(y,x));
xaxis=1:length(xcords);
figure,plot(xaxis,yaxis);ylim([-90,90]);
%% LA wall septal side #1
close all;
  cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/Final_Fiber_Set_rgby/e01_All_yplanes/;
  plane=306*5;
  imname=sprintf('y_%04d.tif',plane);
  im=imread(imname);
  figure,imagesc(im);hold on;
 
   x=[2500;2670];y=[615;591];
    x=round(x);y=round(y);
  m=(y(2)-y(1))/(x(2)-x(1));
b=y(1)-m*x(1);
xcords=min(x):max(x);
clear x;clear y;
ycords=round(m.*xcords+b);
cords=[xcords',ycords'];ucords=unique(cords,'rows','stable');
x=ucords(:,1);y=ucords(:,2);
plot(x,y,'w*');
imhsv=rgb2hsv(im);
h=imhsv(:,:,1);
angle=180*h-90; % converting from [0,1] to [-90,90]
yaxis=diag(angle(y,x));
xaxis=1:length(xcords);
figure,plot(xaxis,yaxis);ylim([-90,90]);
%% LA wall septal side #2
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/Final_Fiber_Set_rgby/e01_All_yplanes/;
  plane=310*5;
  imname=sprintf('y_%04d.tif',plane);
  im=imread(imname);
  figure,imagesc(im);hold on;
  
y=[636;627];x=[2498;2649];
  m=(y(2)-y(1))/(x(2)-x(1));
b=y(1)-m*x(1);
xcords=min(x):max(x);
clear x;clear y;
ycords=round(m.*xcords+b);
cords=[xcords',ycords'];ucords=unique(cords,'rows','stable');
x=ucords(:,1);y=ucords(:,2);
plot(x,y,'w*');
imhsv=rgb2hsv(im);
h=imhsv(:,:,1);
angle=180*h-90; % converting from [0,1] to [-90,90]
yaxis=diag(angle(y,x));
xaxis=1:length(xcords);
figure,plot(xaxis,yaxis);ylim([-90,90]);
% set(gca,'xdir','reverse');
%%  LA Roof (Between PVs) vertical angles 
 cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/Final_Fiber_Set_rgby/e01_All_yplanes/;
  plane=93*5;
  imname=sprintf('y_%04d.tif',plane);
  im=imread(imname);
  figure,imagesc(im);hold on;

  x=[2914;2931];y=[402;347];
  m=(y(2)-y(1))/(x(2)-x(1));
b=y(1)-m*x(1);
xcords=min(x):max(x);
clear x;clear y;
ycords=round(m.*xcords+b);
cords=[xcords',ycords'];ucords=unique(cords,'rows','stable');
x=ucords(:,1);y=ucords(:,2);
plot(x,y,'w*');
imhsv=rgb2hsv(im);
h=imhsv(:,:,1);
angle=180*h-90; % converting from [0,1] to [-90,90]
yaxis=diag(angle(y,x));
xaxis=1:length(xcords);
figure,plot(xaxis,yaxis);ylim([-90,90]);
% set(gca,'xdir','reverse');
%% Posterior LA 
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/Final_Fiber_Set_rgby/e01_All_yplanes/;
  plane=422;% s5 slice #84
  imname=sprintf('y_%04d.tif',plane);
  im=imread(imname);
  figure,imagesc(im);hold on;
 
    x=[2544;2653];
y=[586;600];
  m=(y(2)-y(1))/(x(2)-x(1));
b=y(1)-m*x(1);
xcords=min(x):max(x);
clear x;clear y;
ycords=round(m.*xcords+b);
cords=[xcords',ycords'];ucords=unique(cords,'rows','stable');
x=ucords(:,1);y=ucords(:,2);
plot(x,y,'w*');
imhsv=rgb2hsv(im);
h=imhsv(:,:,1);
angle=180*h-90; % converting from [0,1] to [-90,90]
yaxis=diag(angle(y,x));
xaxis=1:length(xcords);
figure,plot(xaxis,yaxis);ylim([-90,90]);
% set(gca,'xdir','reverse');
%%   In -plane organisation in valve plane
close all;
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/Final_Fiber_Set_rgby/d02_segmented_iso1/;
  plane=194*5;
  imname=sprintf('d02_overlaid_iso0_%04d.tif',plane);
  im=imread(imname);
  figure,imagesc(im);hold on;
 %%
    [x,y]=getpts();
    x=round(x);y=round(y);
  m=(y(2)-y(1))/(x(2)-x(1));
b=y(1)-m*x(1);
xcords=min(x):max(x);
clear x;clear y;
ycords=round(m.*xcords+b);
cords=[xcords',ycords'];ucords=unique(cords,'rows','stable');
x=ucords(:,1);y=ucords(:,2);
plot(x,y,'w*');
imhsv=rgb2hsv(im);
h=imhsv(:,:,1);
angle=180*h-90; % converting from [0,1] to [-90,90]
yaxis=diag(angle(y,x));
xaxis=1:length(xcords);
figure,plot(xaxis,yaxis);ylim([-90,90]);
% set(gca,'xdir','reverse');
%% Interesting section No. 3 (vertical angles in RGBY)

showcs3(single(V(900:1250,1701:2200,800:-1:500,:)));
figure,imagesc(squeeze(V(1100,1701:2400,800:-1:500,:)));view(-90,90);

%% Interesting section No. 4 (vertical angles in RGBY)
showcs3(single(V(1400:1700,2301:2800,725:-1:525,:)));
figure,imagesc(squeeze(V(1556,2301:2800,725:-1:525,:)));view(-90,90);

%% Interesting section No. 5 (vertical angles in RGBY)
showcs3(single(V(1400:1700,3050:3250,820:-1:720,:)));
figure,imagesc(squeeze(V(1556,2301:2800,725:-1:525,:)));view(-90,90);

%% RGBX 
showcs3(single(squeeze(V(1500:1650,3000:3200,900:-1:700,:))))
%% Saving all y-planes
for i=1:2490
    yplane=squeeze(V(i,:,end:-1:1,:));
    name=sprintf('y_%04d.tif',i);
    rotyplane=rot90(yplane);
    cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/Final_Fiber_Set_rgby/e01_All_yplanes/;
    imwrite(rotyplane,name);
    disp(i);
end
%% Angle Tool
close all;
  cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/Final_Fiber_Set_rgby/e01_All_yplanes/;
  plane=245*5;
  imname=sprintf('y_%04d.tif',plane);
  im=imread(imname);
  figure,imagesc(im);hold on;
%%   
    [x,y]=getpts();
    x=round(x);y=round(y);
  m=(y(2)-y(1))/(x(2)-x(1));
b=y(1)-m*x(1);
xcords=min(x):max(x);
clear x;clear y;
ycords=round(m.*xcords+b);
cords=[xcords',ycords'];ucords=unique(cords,'rows','stable');
x=ucords(:,1);y=ucords(:,2);
plot(x,y,'w*');
imhsv=rgb2hsv(im);
h=imhsv(:,:,1);
angle=180*h-90; % converting from [0,1] to [-90,90]
yaxis=diag(angle(y,x));
xaxis=1:length(xcords);
figure,plot(xaxis,yaxis);ylim([-90,90]);
% set(gca,'xdir','reverse');