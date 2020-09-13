%% After rounding
%%
clc;clear;
close all;

side=30;
d=floor(side/2);

%Table computation

dist=sqrt((d*sqrt(2))^2+d^2);
j=round([0.386921361	0.40541386	0.375989485	0.3775	0.44600154	0.458294861	0.427160327	0.445179265],2);
    TA=j(1);TB=j(2);TC=j(3);TD=j(4);TE=j(5);TF=j(6);TG=j(7);TH=j(8);
    VA=dist*0.025/TA;

    VB=dist*0.025/TB;

    VC=dist*0.025/TC;
   
    VD=dist*0.025/TD;

    VE=dist*0.025/TE;
% else
%     disp('mE=0 ');
%     bottomcount=bottomcount+1;
% end

    VF=dist*0.025/TF;

    VG=dist*0.025/TG;

    VH=dist*0.025/TH;
% else
%     disp('mH=0 ');
%     bottomcount=bottomcount+1;
% end
result=round([j;VA VB VC VD VE VF VG VH],2);
%     figure,imagesc(Tnorm(:,:,bottom));title('bottom');colormap(jet);
%     hold on; plot([ox,15,20,70,62],[oy,98,155,110,62],'w*');

%% Ratios Computation
clc;clear;
close all;

oy=100;ox=41;oz=30;side=50;
d=floor(side/2);

for plane=1:60
% cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Chapter6_finalSections/s2/s2_seg2_segmented/Levelset_outs_0.001/;
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Chapter6_finalSections/LAwall_from_S2/03_seg;
imname=sprintf('cut_%04d.tif',plane);
i=imread(imname);
l=logical(i);
map(:,:,plane)=l;
end
%  clearvars -except map;
% cd /hpc_atog/btho733/ABI/matlab_central/FM_Tests/test6/fm2_matlab-master/examples;

% s=smooth3(V,'gaussian',[51,51,51],1.5);
% s=smooth3(map,'gaussian',[3,3,3],0.65);
% s=smooth3(map,'gaussian',[7,7,7],1);
% s=smooth3(map,'gaussian',[11,11,11],3);
% s=smooth3(map,'gaussian',[3,3,3],3);
% s=smooth3(map,'gaussian',[7,7,7],3);% 

% thr=0.8;70
% map=s;map(s>thr)=1;map(s<thr+0.01)=0;
% Adding to path the algorithms folder. This is not required if it is
% permanently included in the Matlab path.
p1=genpath('/hpc_atog/btho733/ABI/matlab_central/FM_Tests/test6');
    addpath(p1);    

p2=genpath('/hpc_atog/btho733/ABI/matlab_central/FM_Tests/test2');
    addpath(p2); 

% Parameters:
sat =0; % Between 0 and 1.
% oz=60;
% figure,imagesc(map(:,:,oz));hold on;
% [xp,yp]=getpts();
%     ox=round(xp);oy=round(yp);
% Load map and adapt it
% map = imread('../data/map.bmp');
% Changing order of start and goal because of the (x,y) - (row,col)
 % correspondence.
start =[oy;ox;oz];%[105;40;30];%[yp;xp;start_prop];%[20;76;60];%[181;3;23];
goal = [8;8;1];%[46;72;23];
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
% Tnorm=T;
Tnorm=(2.2523.*T)/out;


% Inpainting, ST computation and Anisotropy table

clear T;
T=Tnorm;
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
rho =40;
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

%%
clc;clear;
close all;

oy=100;ox=41;oz=30;side=30;
d=floor(side/2);

for plane=1:60
% cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Chapter6_finalSections/s2/s2_seg2_segmented/Levelset_outs_0.001/;
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Chapter6_finalSections/LAwall_from_S2/03_seg;
imname=sprintf('cut_%04d.tif',plane);
i=imread(imname);
l=logical(i);
map(:,:,plane)=l;
end
%  clearvars -except map;
% cd /hpc_atog/btho733/ABI/matlab_central/FM_Tests/test6/fm2_matlab-master/examples;

% s=smooth3(V,'gaussian',[51,51,51],1.5);
% s=smooth3(map,'gaussian',[3,3,3],0.65);
% s=smooth3(map,'gaussian',[7,7,7],1);
% s=smooth3(map,'gaussian',[11,11,11],3);
% s=smooth3(map,'gaussian',[3,3,3],3);
% s=smooth3(map,'gaussian',[7,7,7],3);% 

% thr=0.8;
% map=s;map(s>thr)=1;map(s<thr+0.01)=0;
% Adding to path the algorithms folder. This is not required if it is
% permanently included in the Matlab path.
p1=genpath('/hpc_atog/btho733/ABI/matlab_central/FM_Tests/test6');
    addpath(p1);    

p2=genpath('/hpc_atog/btho733/ABI/matlab_central/FM_Tests/test2');
    addpath(p2); 

% Parameters:
sat =0; % Between 0 and 1.
% oz=60;
% figure,imagesc(map(:,:,oz));hold on;
% [xp,yp]=getpts();
%     ox=round(xp);oy=round(yp);
% Load map and adapt it
% map = imread('../data/map.bmp');
% Changing order of start and goal because of the (x,y) - (row,col)
 % correspondence.
start =[oy;ox;oz];%[105;40;30];%[yp;xp;start_prop];%[20;76;60];%[181;3;23];
goal = [8;8;1];%[46;72;23];
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
Tnorm=T;
Tnorm=(2.2523.*T)/out;
% colormap jet;
% axis xy;
% axis image;
% axis off;
cd /hpc_atog/btho733/ABI/pacedSheep01/Anisotropic/functions;
% cd V:\ABI\pacedSheep01\Anisotropic\functions;
showcs3(single(Tnorm(:,end:-1:1,:)));

%Table computation

dist=sqrt((d*sqrt(2))^2+d^2);top=oz+d;bottom=oz-d;
topcount=0;bottomcount=0;
mA=map(95,15,top);
if(mA==1)
    TA=Tnorm(95,15,top);
    VA=dist*0.025/TA;
else
    disp('mA=0 ');
    topcount=topcount+1;
end

mB=map(155,20,top);
if(mB==1)
    TB=Tnorm(155,20,top);VB=dist*0.025/TB;
else
    disp('mB=0 ');
    topcount=topcount+1;
end

mC=map(110,70,top);
if(mC==1)
    TC=Tnorm(110,70,top);VC=dist*0.025/TC;
else
    disp('mC=0 ');
    topcount=topcount+1;
end

mD=map(60,60,top);
if(mD==1)
    TD=Tnorm(60,60,top);VD=dist*0.025/TD;
else
    disp('mD=0 ');
    topcount=topcount+1;
end

    figure,imagesc(Tnorm(:,:,top));title('Top');colormap(jet);
    hold on; plot([ox,15,20,70,60],[oy,95,155,110,60],'w*');


mE=map(98,15,bottom);
if(mE==1)
    TE=Tnorm(98,15,bottom);VE=dist*0.025/TE;
else
    disp('mE=0 ');
    bottomcount=bottomcount+1;
end

mF=map(155,20,bottom);
if(mF==1)
    TF=Tnorm(155,20,bottom);VF=dist*0.025/TF;
else
    disp('mF=0 ');
    bottomcount=bottomcount+1;
end

mG=map(110,70,bottom);
if(mG==1)
    TG=Tnorm(110,70,bottom);    
    VG=dist*0.025/TG;
else
    disp('mG=0 ');
    bottomcount=bottomcount+1;
end

mH=map(62,62,bottom);
if(mH==1)
    TH=Tnorm(62,62,bottom);VH=dist*0.025/TH;
else
    disp('mH=0 ');
    bottomcount=bottomcount+1;
end

    figure,imagesc(Tnorm(:,:,bottom));title('bottom');colormap(jet);
    hold on; plot([ox,15,20,70,62],[oy,98,155,110,62],'w*');
%%
clc;clear;
close all;

oy=100;ox=41;oz=30;side=30;
d=floor(side/2);

for plane=1:60
% cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Chapter6_finalSections/s2/s2_seg2_segmented/Levelset_outs_0.001/;
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Chapter6_finalSections/LAwall_from_S2/03_seg;
imname=sprintf('cut_%04d.tif',plane);
i=imread(imname);
l=logical(i);
map(:,:,plane)=l;
end
%  clearvars -except map;
% cd /hpc_atog/btho733/ABI/matlab_central/FM_Tests/test6/fm2_matlab-master/examples;

% s=smooth3(V,'gaussian',[51,51,51],1.5);
% s=smooth3(map,'gaussian',[3,3,3],0.65);
% s=smooth3(map,'gaussian',[7,7,7],1);
% s=smooth3(map,'gaussian',[11,11,11],3);
% s=smooth3(map,'gaussian',[3,3,3],3);
% s=smooth3(map,'gaussian',[7,7,7],3);% 

% thr=0.8;
% map=s;map(s>thr)=1;map(s<thr+0.01)=0;
% Adding to path the algorithms folder. This is not required if it is
% permanently included in the Matlab path.
p1=genpath('/hpc_atog/btho733/ABI/matlab_central/FM_Tests/test6');
    addpath(p1);    

p2=genpath('/hpc_atog/btho733/ABI/matlab_central/FM_Tests/test2');
    addpath(p2); 

% Parameters:
sat =0; % Between 0 and 1.
% oz=60;
% figure,imagesc(map(:,:,oz));hold on;
% [xp,yp]=getpts();
%     ox=round(xp);oy=round(yp);
% Load map and adapt it
% map = imread('../data/map.bmp');
% Changing order of start and goal because of the (x,y) - (row,col)
 % correspondence.
start =[oy;ox;oz];%[105;40;30];%[yp;xp;start_prop];%[20;76;60];%[181;3;23];
goal = [8;8;1];%[46;72;23];
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
Tnorm=T;
% Tnorm=(2.2523.*T)/out;
% colormap jet;
% axis xy;
% axis image;
% axis off;
cd /hpc_atog/btho733/ABI/pacedSheep01/Anisotropic/functions;
% cd V:\ABI\pacedSheep01\Anisotropic\functions;
showcs3(single(Tnorm(:,end:-1:1,:)));

%Table computation

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
gg=unique(Tnorm);
rem=mod(length(gg),2);
if(rem==0)
rgg=reshape(gg,length(gg)/2,2);
figure, imagesc(gg);colormap(jet);colorbar;
else
gg2=gg(2:end);
rgg=reshape(gg2,length(gg2)/2,2);
figure, imagesc(gg);colormap(jet);colorbar;
end
