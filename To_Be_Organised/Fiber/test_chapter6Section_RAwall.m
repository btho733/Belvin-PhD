clear;clc;
close all;

oy=70;ox=51;oz=50;side=50;
d=floor(side/2);

% Adding to path the algorithms folder. This is not required if it is
% permanently included in the Matlab path.
if isempty(regexp(path,['algorithms' pathsep], 'once'))
p1=genpath('/hpc_atog/btho733/ABI/matlab_central/FM_Tests/test6');
    addpath(p1);    

p2=genpath('/hpc_atog/btho733/ABI/matlab_central/FM_Tests/test2');
    addpath(p2); 
end
cd /hpc_atog/btho733/ABI/matlab_central/FM_Tests/test6/fm2_matlab-master/examples;

% s=smooth3(V,'gaussian',[51,51,51],1.5);
% s=smooth3(map,'gaussian',[3,3,3],0.65);
% s=smooth3(map,'gaussian',[7,7,7],1);
% s=smooth3(map,'gaussian',[11,11,11],3);
% s=smooth3(map,'gaussian',[3,3,3],3);
% s=smooth3(map,'gaussian',[7,7,7],3);% 

% thr=0.8;
% map=s;map(s>thr)=1;map(s<thr+0.01)=0;

% Parameters:
sat =0; % Between 0 and 1.

% Load map and adapt it
% map = imread('../data/map.bmp');

map=importdata('../data/geo4_JR1_LS.mat');
% map = flipdim(map,1); % To change the Y coordinates.

% These points can be set with ginput or other input methods. They have to
% be integers at last.
start =[oy;ox;oz];%[70,1,50];%[70;51;50];%[70;51;1];%[57;69;100];%[57;69;100];%[75;50;50];%[72;58;52];%[72;58;52];%[12;8;65];%[15;57;60];
goal =[10;96;98];%[20;41;1];%[10;96;98];[6;10;100];%[47;78;1];%[5;97;3];%[5;97;3];[145;6;3];% [100;94;65];%[28;27;65];

% Plotting map
% figure(1);
% hold on;
% imagesc(map);
% colormap gray(256);
% axis xy;
% axis image;
% axis off;
% plot(start(1), start(2), 'rx', 'MarkerSize', 15);
% plot(goal(1), goal(2), 'k*', 'MarkerSize', 15);

% Changing order of start and goal because of the (x,y) - (row,col)
 % correspondence.
start = [start(1); start(2);start(3)];
goal = [goal(1); goal(2);goal(3)];
W = double(map);
% Parameters setting
options.nb_iter_max = Inf;
options.Tmax        = sum(size(W));
options.end_points  = goal;


[T,~] = perform_fast_marching(W, start, options);
% Plotting times-of-arrival map.

% figure,imagesc(T(:,:,98));
% figure,imagesc(T(:,:,1));colormap jet;

A1 = unique(T);
out = A1(end-1);

T(T>out)=out;
Tnorm=T;
% colormap jet;
% axis xy;
% axis image;
% axis off;
% cd V:\ABI\pacedSheep01\Anisotropic\functions;
cd /hpc_atog/btho733/ABI/pacedSheep01/Anisotropic/functions;
showcs3(single(Tnorm));

% Table computation

dist=sqrt((d*sqrt(2))^2+d^2);top=oz+d;bottom=oz-d;
topcount=0;bottomcount=0;
mA=map(oy-d,ox-d,top);
if(mA==1)
    TA=Tnorm(oy-d,ox-d,top);VA=dist*0.025/TA;
else
    disp('mA=0 ');
    topcount=topcount+1;
end

mB=map(oy+d,ox-d,top);
if(mB==1)
    TB=Tnorm(oy+d,ox-d,top);VB=dist*0.025/TB;
else
    disp('mB=0 ');
    topcount=topcount+1;
end

mC=map(oy+d,ox+d,top);
if(mC==1)
    TC=Tnorm(oy+d,ox+d,top);VC=dist*0.025/TC;
else
    disp('mC=0 ');
    topcount=topcount+1;
end

mD=map(oy-d,ox+d,top);
if(mD==1)
    TD=Tnorm(oy-d,ox+d,top);VD=dist*0.025/TD;
else
    disp('mD=0 ');
    topcount=topcount+1;
end

    figure,imagesc(Tnorm(:,:,top));title('Top');colormap(jet);
    hold on; plot([ox,ox-d,ox-d,ox+d,ox+d],[oy,oy-d,oy+d,oy+d,oy-d],'w*');


mE=map(oy-d,ox-d,bottom);
if(mE==1)
    TE=Tnorm(oy-d,ox-d,bottom);VE=dist*0.025/TE;
else
    disp('mE=0 ');
    bottomcount=bottomcount+1;
end

mF=map(oy+d,ox-d,bottom);
if(mF==1)
    TF=Tnorm(oy+d,ox-d,bottom);VF=dist*0.025/TF;
else
    disp('mF=0 ');
    bottomcount=bottomcount+1;
end

mG=map(oy+d,ox+d,bottom);
if(mG==1)
    TG=Tnorm(oy+d,ox+d,bottom);VG=dist*0.025/TG;
else
    disp('mG=0 ');
    bottomcount=bottomcount+1;
end

mH=map(oy-d,ox+d,bottom);
if(mH==1)
    TH=Tnorm(oy-d,ox+d,bottom);VH=dist*0.025/TH;
else
    disp('mH=0 ');
    bottomcount=bottomcount+1;
end

    figure,imagesc(Tnorm(:,:,bottom));title('bottom');colormap(jet);
    hold on; plot([ox,ox-d,ox-d,ox+d,ox+d],[oy,oy-d,oy+d,oy+d,oy-d],'w*');

%% colorbar;
gg=unique(T);
rem=mod(length(gg),2);
if(rem==0)

rgg=reshape(gg,length(gg)/2,2);
figure, imagesc(gg);colormap(jet);colorbar;
else
gg2=gg(2:end);
rgg=reshape(gg2,length(gg2)/2,2);
figure, imagesc(gg);colormap(jet);colorbar;
end
