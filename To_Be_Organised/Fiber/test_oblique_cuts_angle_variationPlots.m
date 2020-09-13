%% Angle Tool
close all;
  cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Angle_Variation_across_AtrialWall/oblique/;
%   plane=245*5;
%   imname=sprintf('y_%04d.tif',plane);
  im1=imread('hflip_BB_oblique.tif');
  im=squeeze(im1(:,:,1:3));
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
% set(gca,'ydir','normal');

%%  Oblique #1

close all;
 cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Angle_Variation_across_AtrialWall/oblique;
%   plane=245*5;
%   imname=sprintf('y_%04d.tif',plane);
  im1=imread('d04_0001.tif-Oblique-Slice-Ortho-Slice.tif');
  im=squeeze(im1(:,:,1:3));
  figure,imagesc(im);hold on;
 x=[1298;1326];y=[519;484];
  m=(y(2)-y(1))/(x(2)-x(1));
b=y(1)-m*x(1);
xcords=min(x):max(x);
clear x;clear y;
ycords=round(m.*xcords+b);
cords=[xcords',ycords'];ucords=unique(cords,'rows','stable');
x=ucords(:,1);y=ucords(:,2);
plot(x,y,'w');
imhsv=rgb2hsv(im);
h=imhsv(:,:,1);
angle=180*h-90; % converting from [0,1] to [-90,90]
yaxis=diag(angle(y,x));
xaxis=1:length(xcords);
set(gca,'ydir','normal');
figure,plot(xaxis,yaxis);ylim([-90,90]);
%% oblique #2  Septum
close all;
  cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Angle_Variation_across_AtrialWall/oblique/;
%   plane=245*5;
%   imname=sprintf('y_%04d.tif',plane);
  im1=imread('hflip_BB_oblique.tif');
  im=squeeze(im1(:,:,1:3));
  figure,imagesc(im);hold on;

  x=[850;947];y=[355;403];
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
% set(gca,'ydir','normal');
%% oblique #2  BB
close all;
%   cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Angle_Variation_across_AtrialWall/oblique/;
cd V:\ABI\JZ\Fiber_DTI\Angle_Variation_across_AtrialWall\oblique;
%   plane=245*5;
%   imname=sprintf('y_%04d.tif',plane);
  im1=imread('hflip_BB_oblique.tif');
  im=squeeze(im1(:,:,1:3));
  figure,imagesc(im);hold on;

  x=[956;1034];y=[206;272];
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
% set(gca,'ydir','normal');
 %%  Septum test tegion #3
close all;
  cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Chapter6_finalSections/s1/02_fiber
% cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Chapter6_finalSections/s1/fiber_largesection/;
  plane=195;%167
  imname=sprintf('fiber_%04d.tif',plane);
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
figure,plot(xaxis,yaxis,'m.');ylim([-90,90]);
% set(gca,'xdir','reverse');

 %%  Septum test region #1
close all;
  cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Chapter6_finalSections/tr1/fiber_rgby/
  plane=15;
  imname=sprintf('cut_%04d.tif',plane);
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

%%  RASmoothWall test region #2
close all;
  cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Chapter6_finalSections/tr1/fiber_rgby/
  plane=75;
  imname=sprintf('cut_%04d.tif',plane);
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

%%  BB test region #3
close all;
  cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Chapter6_finalSections/tr2/fiber_rgby/
  plane=20;
  imname=sprintf('cut_%04d.tif',plane);
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

%% LA Wall
close all;
  cd V:\ABI\JZ\Fiber_DTI\Chapter6_finalSections\tr3_froms2\testcut1_fiber_rgby;%/hpc_atog/btho733/ABI/JZ/Fiber_DTI/Chapter6_finalSections/tr2/fiber_rgby/
  plane=10;
  imname=sprintf('cut_%04d.tif',plane);
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

