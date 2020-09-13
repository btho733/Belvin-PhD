%% Ratios computation
% Fast Marching
clc;clear;
close all;

offset=180;
oy=50;ox=51;oz=31;side=50;
d=floor(side/2);

for plane=offset:239
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Chapter6_finalSections/Septum_froms1_180to259/03_seg;
imname=sprintf('cut_%04d.tif',plane);
i=imread(imname);
l=logical(i);
map(:,:,plane-offset+1)=l;
end
%  clearvars -except map start_prop plane offset;
% cd /hpc_atog/btho733/ABI/matlab_central/FM_Tests/test6/fm2_matlab-master/examples;

% s=smooth3(V,'gaussian',[51,51,51],1.5);
s=smooth3(map,'gaussian',[3,3,3],0.65);
% s=smooth3(map,'gaussian',[7,7,7],1);
% s=smooth3(map,'gaussian',[11,11,11],3);
% s=smooth3(map,'gaussian',[3,3,3],3);
% s=smooth3(map,'gaussian',[7,7,7],3);% 

thr=0.8;
map=s;map(s>thr)=1;map(s<thr+0.01)=0;

% Adding to path the algorithms folder. This is not required if it is
% permanently included in the Matlab path.
p1=genpath('/hpc_atog/btho733/ABI/matlab_central/FM_Tests/test6');
    addpath(p1);    

p2=genpath('/hpc_atog/btho733/ABI/matlab_central/FM_Tests/test2');
    addpath(p2); 

% Parameters:
sat =0; % Between 0 and 1.
% figure,imagesc(map(:,:,start_prop));hold on;
% [xp,yp]=getpts();
%     xp=round(xp);yp=round(yp);
% Load map and adapt it
% map = imread('../data/map.bmp');
% Changing order of start and goal because of the (x,y) - (row,col)
 % correspondence.
start =[oy;ox;oz];%[95;110;93];%[yp;xp;start_prop];%[100;127;92];%[119;98;1];%
goal = [1;48;60];%[59;36;plane-offset+1];%[46;72;23];%
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


% Inpainting, ST computation and Anisotropy table

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
%% Cut section
clc;clear;
close all;

offset=180;
oy=50;ox=51;oz=31;side=59;
d=floor(side/2);

for plane=offset:239
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Chapter6_finalSections/Septum_froms1_180to259/03_seg;
imname=sprintf('cut_%04d.tif',plane);
i=imread(imname);
l=logical(i);
map(:,:,plane-offset+1)=l;
end
%  clearvars -except map start_prop plane offset;
% cd /hpc_atog/btho733/ABI/matlab_central/FM_Tests/test6/fm2_matlab-master/examples;

% s=smooth3(V,'gaussian',[51,51,51],1.5);
s=smooth3(map,'gaussian',[3,3,3],0.65);
% s=smooth3(map,'gaussian',[7,7,7],1);
% s=smooth3(map,'gaussian',[11,11,11],3);
% s=smooth3(map,'gaussian',[3,3,3],3);
% s=smooth3(map,'gaussian',[7,7,7],3);% 

thr=0.8;
map=s;map(s>thr)=1;map(s<thr+0.01)=0;

% Adding to path the algorithms folder. This is not required if it is
% permanently included in the Matlab path.
p1=genpath('/hpc_atog/btho733/ABI/matlab_central/FM_Tests/test6');
    addpath(p1);    

p2=genpath('/hpc_atog/btho733/ABI/matlab_central/FM_Tests/test2');
    addpath(p2); 

% Parameters:
sat =0; % Between 0 and 1.
% figure,imagesc(map(:,:,start_prop));hold on;
% [xp,yp]=getpts();
%     xp=round(xp);yp=round(yp);
% Load map and adapt it
% map = imread('../data/map.bmp');
% Changing order of start and goal because of the (x,y) - (row,col)
 % correspondence.
start =[oy;ox;oz];%[95;110;93];%[yp;xp;start_prop];%[100;127;92];%[119;98;1];%
goal = [1;48;60];%[59;36;plane-offset+1];%[46;72;23];%
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
showcs3(single(Tnorm(:,:,end:-1:1)));
% 
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

%% Cutting
clc;clear;close all;

for i=1:350
imname=sprintf('cut_%04d.tif',i);fibname=sprintf('fiber_%04d.tif',i);segname=sprintf('cut_%04d.png',i);
cd V:\ABI\JZ\Fiber_DTI\Chapter6_finalSections\s1\01_corrected;
im=imread(imname);
imcut=im(65:164,7:116);
cd V:\ABI\JZ\Fiber_DTI\Chapter6_finalSections\Septum_froms1_180to259\01_corrected;
imwrite(imcut,imname);
cd V:\ABI\JZ\Fiber_DTI\Chapter6_finalSections\s1\02_fiber;
fib=imread(fibname);
fibcut=fib(65:164,7:116,:);
cd V:\ABI\JZ\Fiber_DTI\Chapter6_finalSections\Septum_froms1_180to259\02_fiber;
imwrite(fibcut,fibname);
cd V:\ABI\JZ\Fiber_DTI\Chapter6_finalSections\s1\Levelset_outs_neg1\rem_outlier_bright1.5;
seg=imread(segname);
segcut=seg(65:164,7:116);
cd V:\ABI\JZ\Fiber_DTI\Chapter6_finalSections\Septum_froms1_180to259\03_seg;
imwrite(segcut,imname);
end

%% Full
clc;clear;
% close all;
offset=165;start_prop=92;
for plane=offset:350
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Chapter6_finalSections/s1/Levelset_outs_neg1/rem_outlier_bright1.5/;
imname=sprintf('cut_%04d.png',plane);
i=imread(imname);
l=logical(i);
map(:,:,plane-offset+1)=l;
end
 clearvars -except map start_prop plane offset;
% cd /hpc_atog/btho733/ABI/matlab_central/FM_Tests/test6/fm2_matlab-master/examples;

% Adding to path the algorithms folder. This is not required if it is
% permanently included in the Matlab path.
p1=genpath('/hpc_atog/btho733/ABI/matlab_central/FM_Tests/test6');
    addpath(p1);    

p2=genpath('/hpc_atog/btho733/ABI/matlab_central/FM_Tests/test2');
    addpath(p2); 

% Parameters:
sat =0; % Between 0 and 1.
% figure,imagesc(map(:,:,start_prop));hold on;
% [xp,yp]=getpts();
%     xp=round(xp);yp=round(yp);
% Load map and adapt it
% map = imread('../data/map.bmp');
% Changing order of start and goal because of the (x,y) - (row,col)
 % correspondence.
start =[95;110;93];%[yp;xp;start_prop];%[100;127;92];%[119;98;1];%
goal = [59;36;plane-offset+1];%[46;72;23];%
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
Tnorm=(2.2523.*T)/out;
% colormap jet;
% axis xy;
% axis image;
% axis off;
cd /hpc_atog/btho733/ABI/pacedSheep01/Anisotropic/functions;
showcs3(single(Tnorm(:,:,end:-1:1)));
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

