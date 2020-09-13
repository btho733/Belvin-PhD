clc;close all;clear;
cd V:\ABI\JZ\Fiber_DTI\Propagation_Tool\;
p=imread('Synthet1_ST_2D_scale5.tif');
figure,imagesc(p);
hsv=rgb2hsv(p);
h=hsv(:,:,1);

ori=180.*(h)-90; % converting from [0,1] to [-90,90]
vf=80;vt=80;



xs=150;ys=200;theta=90;%ori(ys,xs);
phi=30;

centre=theta;
% xnew(1)=xs+vf*cosd(centre+phi);
% ynew(1)=ys-vf*sind(centre+phi);
xnew(1)=xs+vf*cosd(centre);
ynew(1)=ys-vf*sind(centre);
% xnew(3)=xs+vf*cosd(centre-phi);
% ynew(3)=ys-vf*sind(centre-phi);

centre=theta-90;
% xnew(4)=xs+vt*cosd(centre+phi);
% ynew(4)=ys-vt*sind(centre+phi);
xnew(2)=xs+vt*cosd(centre);
ynew(2)=ys-vt*sind(centre);
% xnew(6)=xs+vt*cosd(centre-phi);
% ynew(6)=ys-vt*sind(centre-phi);

centre=theta+180;
% xnew(7)=xs+vf*cosd(centre+phi);
% ynew(7)=ys-vf*sind(centre+phi);
xnew(3)=xs+vf*cosd(centre);
ynew(3)=ys-vf*sind(centre);
% xnew(9)=xs+vf*cosd(centre-phi);
% ynew(9)=ys-vf*sind(centre-phi);

centre=theta+90;
% xnew(10)=xs+vt*cosd(centre+phi);
% ynew(10)=ys-vt*sind(centre+phi);
xnew(4)=xs+vt*cosd(centre);
ynew(4)=ys-vt*sind(centre);
% xnew(12)=xs+vt*cosd(centre-phi);
% ynew(12)=ys-vt*sind(centre-phi);

propgeo=2.*ones(303,450);
p1=squeeze(p(:,:,1));p2=squeeze(p(:,:,2));p3=squeeze(p(:,:,3));
propgeo(p1==0 & p2==0 & p3==0 & ori==-90)=0;

startmask=propgeo; % At this stage, propgeo is the start mask

propgeo(ys,xs)=1;

xy=[xnew' ynew'];
l=length(xy);
xy(l+1,:)=xy(1,:);   
spcv=cscvn(xy');              
p=fnplt(spcv);
up=unique(p','rows','stable');
% hold on;plot(up(:,1),up(:,2),'r.','LineWidth',1.6);
BW=poly2mask(up(:,1),up(:,2),303,450);
% figure,imagesc(ori);colormap(jet);title('Fiber orientation map');colorbar;
propgeo=2.*zeros(303,450);
propgeo(BW==1)=1;figure,imagesc(propgeo);colormap(jet);%title('propagation on geometry ');
hold on;plot(p(1,:),p(2,:),'w-','Marker','o');
% BW(BW==1)=255;
% sBW=imgaussfilt(uint8(BW),[8,1]);figure,imagesc(sBW);
% r=imresize(BW,0.25);figure,imagesc(r);

% figure,imagesc(ori);colormap(jet);colorbar;hold on;plot(p(1,:),p(2,:),'r-','Marker','o');xlim([1 450]);ylim([1 303]);set(gca,'YDir','reverse')
cd V:\ABI\matlab_central\curvature_ANd_Normals\;
vertices=up;
N=LineNormals2D(vertices);
xx=[vertices(:,1)-1*N(:,1) vertices(:,1) vertices(:,1)+1*N(:,1)]';
yy=[vertices(:,2)-1*N(:,2) vertices(:,2) vertices(:,2)+1*N(:,2)]';
figure,plot(p(1,:),p(2,:),'r-','Marker','o');hold on; plot(xx,yy);    
    
%% Normals test script
close all;
clear;close all;
for imno=592
    close all;
imname2=sprintf('Scut_%05d.png',imno+1);pointname=sprintf('c_%05d.mat',imno);orname2=sprintf('cut_%05d.png',imno+1);
cd /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/v1_cutsection_from_S/scale16;
% cd V:\ABI\pacedSheep01\test__segment_Background_Subtraction\v1_cutsection_from_S\scale16;
im2=imread(imname2);cd ..;
cd /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/v1_cutsection_from_S/scale16/coords_from_program_without_fit;
% cd V:\ABI\pacedSheep01\test__segment_Background_Subtraction\v1_cutsection_from_S\scale16\coords_from_program_without_fit;
points=importdata(pointname);
% load('testdata');
cd  /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/uu1_normalised_from_d_originals/;
% cd V:\ABI\pacedSheep01\test__segment_Background_Subtraction\uu1_normalised_from_d_originals;
or2=imread(orname2);cd ..;
or2=imresize(or2,0.25);
cd /hpc_atog/btho733/ABI/matlab_central/curvature_ANd_Normals;
% cd V:\ABI\matlab_central\curvature_ANd_Normals\;
vertices=unique(points','rows','stable'); N=LineNormals2D(vertices);
xx=[vertices(:,1)-5*N(:,1) vertices(:,1)-4*N(:,1) vertices(:,1)-3*N(:,1) vertices(:,1)-2*N(:,1) vertices(:,1)-1*N(:,1) vertices(:,1) vertices(:,1)+1*N(:,1) vertices(:,1)+2*N(:,1) vertices(:,1)+3*N(:,1) vertices(:,1)+4*N(:,1) vertices(:,1)+5*N(:,1)]';
yy=[vertices(:,2)-5*N(:,2) vertices(:,2)-4*N(:,2) vertices(:,2)-3*N(:,2) vertices(:,2)-2*N(:,2) vertices(:,2)-1*N(:,2) vertices(:,2) vertices(:,2)+1*N(:,2) vertices(:,2)+2*N(:,2) vertices(:,2)+3*N(:,2) vertices(:,2)+4*N(:,2) vertices(:,2)+5*N(:,2)]';
hold on; plot(xx,yy);
end
%%
clear all;clc;
close all;
Im_no=501;  % CHANGE ONLY THIS LINE 
Im_name=sprintf('s4_%05d.png',Im_no);
cd V:\ABI\pacedSheep01\test__segment_Background_Subtraction\d_images2dcoarsesegment_s4;
x1=1541;x2=2590;y1=876;y2=1345;z1=45;z2=46;
im=imread(Im_name);
imcut=im(y1:y2,x1:x2,1);
im1=im(:,:,1);im2=im(:,:,2);im3=im(:,:,3);
im1(1:end,1:x1)=0;im1(1:y1,x1:end)=0;im1(y1:y2,x2:end)=0;im1(y2:end,x1:end)=0;
im(:,:,1)=im1;
im2(1:end,1:x1)=0;im2(1:y1,x1:end)=0;im2(y1:y2,x2:end)=0;im2(y2:end,x1:end)=0;
im(:,:,2)=im2;

figure,imagesc(im);
title([' Image no.  ',num2str(Im_no)]);
cd V:\ABI\matlab_central\active_contour_without_edge\c\c2;
c=importdata('c_00501.mat');
spcv=cscvn(c');                 
p=fnplt(spcv);
vertices=unique(p','rows','stable');
hold on;plot(vertices(:,1),vertices(:,2),'w*');

cd V:\ABI\matlab_central\curvature_ANd_Normals\;
N=LineNormals2D(vertices);

xx=[vertices(:,1)-5*N(:,1) vertices(:,1)-4*N(:,1) vertices(:,1)-3*N(:,1) vertices(:,1)-2*N(:,1) vertices(:,1)-1*N(:,1) vertices(:,1) vertices(:,1)+1*N(:,1) vertices(:,1)+2*N(:,1) vertices(:,1)+3*N(:,1) vertices(:,1)+4*N(:,1) vertices(:,1)+5*N(:,1)]';
yy=[vertices(:,2)-5*N(:,2) vertices(:,2)-4*N(:,2) vertices(:,2)-3*N(:,2) vertices(:,2)-2*N(:,2) vertices(:,2)-1*N(:,2) vertices(:,2) vertices(:,2)+1*N(:,2) vertices(:,2)+2*N(:,2) vertices(:,2)+3*N(:,2) vertices(:,2)+4*N(:,2) vertices(:,2)+5*N(:,2)]';
hold on; plot(xx,yy);

%%

clear;clc;close all;
num=10000;
x=40+sin(0:1/num:2*pi)*10;
y=40+cos(0:1/num:2*pi)*10;
figure,plot(x,y,'r.');xlim([0 100]);ylim([0 100]);hold on;
vertices=[x' y']; cd V:\ABI\matlab_central\curvature_ANd_Normals;
N=LineNormals2D(vertices);
xx=[vertices(:,1)-1*N(:,1) vertices(:,1)]';
yy=[vertices(:,2)-1*N(:,2) vertices(:,2)]';
plot(xx(:,1:500:end),yy(:,1:500:end));
xx=xx';yy=yy';
x1=xx(:,2);y1=yy(:,2);
x2=xx(:,1);y2=yy(:,1);
angletan=(atan2((y2-y1),(x2-x1))).*180/pi;
% alpha(1)=180;
% for i=2:length(x)-1
%     den(i)=(x(i-1)-x(i))*(x(i-1)-x(i+1));   
%     t1(i)=y(i-1)*(x(i)-x(i+1))/den(i); 
%     t2(i)=y(i)*(2*x(i)-x(i-1)-x(i+1))/den(i);
%     t3(i)=y(i+1)*(x(i)-x(i-1))/den(i); 
%     t(i)=acosd(t1(i)+t2(i)+t3(i));
%     alpha(i)=90+t(i);
% %     clear t1;clear t2;clear t3;clear den;
% end
% alpha(length(x))=180;
% alpha=alpha';
% t=t';
% t1=t1';
% t2=t2';
% t3=t3';

%%
clear;close all;
delta= [0;30;45;90;135;150;180;225;240;270;-45;-30];
beta=90.*ones(12,1);
x=0;y=0;
dl=3;dt=1;
xL=dl.*(cosd(delta).*cosd(beta));
xT=dt.*(sind(delta).*sind(beta));
yL=dl.*(cosd(delta).*sind(beta));
yT=dt.*(sind(delta).*cosd(beta));
hor=xL+xT;
ver=yL+yT;


%% Saving sample code
h=figure;imagesc(sobtarget);title(['Image no   ' ,num2str(targetno)]);
% hold on;plot(coords(:,1),coords(:,2),'w*');
% newpoints=[xnew' ynew'];
% coords=unique(round(newpoints),'rows','stable');
% coords1=sgolayfilt(coords(1:end,:),2,11);
% xy=coords';spcv=cscvn(xy);points=fnplt(spcv,'w',1);
% xy4plot=coords1';spcv4plot=cscvn(xy4plot);points4plot=fnplt(spcv4plot,'w',1);
hold on;plot(points4plot(1,:),points4plot(2,:),'w','LineWidth',1.6);
cd ur_figures/;saveas(h,sprintf('FIG_1_%d.tif',targetno));cd ..;

% For Dynamically varying colorbar
%#################################
% figure, imagesc(squeeze(A(:,:,1,1))); colorbar    % For Dynamically varying colorbar
% axis tight manual;                                % For Dynamically varying colorbar
% set(gca,'NextPlot','replacechildren');            % For Dynamically varying colorbar
writerObj = VideoWriter('vid.avi');
writerObj.FrameRate=7;
open(writerObj);
for i=1:n
    
   imagesc(TD(:,:,i,1));
   title(['Frame:' num2str(i)]);
   colorbar; set(gca, 'clim', [minval maxval]);    % For Fixed colorbar over the entire movie
   M(:,:,:,i)=getframe(gcf);
   writeVideo(writerObj,M(:,:,:,i));

end 

close(writerObj);

%% 

clear;clc;close all;
beta_angle=90;points=10;
xstart=150;ystart=150;
col=300;row=300;
dl=2;dt=1;
x=xstart+sin(0:1/points:2*pi)*5;
y=ystart+cos(0:1/points:2*pi)*5;
x=x';y=y';
% figure,plot(x,y,'r.');xlim([0 col]);ylim([0 row]);hold on;pause;
BW=poly2mask(x,y,row,col);
figure,imagesc(BW);colormap(jet);hold on;plot(x,y,'w.');xlim([0 col]);ylim([0 row]);
for i=1:160
cd V:\ABI\matlab_central\curvature_ANd_Normals;
N=LineNormals2D([x y]);
x1=x;x2=x-1*N(:,1);
y1=y;y2=y-1*N(:,2);
alpha=atan2d((y2-y1),(x2-x1));
beta=-beta_angle.*ones(length(alpha),1);
delta=beta-alpha;

xL=dl.*(cosd(delta).*cosd(beta));
xT=dt.*(sind(delta).*sind(beta));
yL=dl.*(cosd(delta).*sind(beta));
yT=dt.*(sind(delta).*cosd(beta));
hor=xL+xT;
ver=yL+yT;
xnew=x+hor;ynew=y+ver;
plot(xnew,ynew,'w.');hold on;
% BW=poly2mask(xnew,ynew,row,col);
% imagesc(BW);colormap(jet)
x=xnew;y=ynew;
disp(i);
pause;
end


%% sgolay filtered version managing more number of points

clear;clc;close all;
beta_angle=90;points=1000;
xstart=150;ystart=150;
dl=2;dt=1;
x=xstart+sin(0:1/points:2*pi)*20;
y=ystart+cos(0:1/points:2*pi)*20;
x=x';y=y';
figure,plot(x,y,'r.');xlim([0 450]);ylim([0 450]);hold on;pause;
for i=1:100
cd V:\ABI\matlab_central\curvature_ANd_Normals;
N=LineNormals2D([x y]);
x1=x;x2=x-1*N(:,1);
y1=y;y2=y-1*N(:,2);
alpha=atan2d((y2-y1),(x2-x1));
beta=beta_angle.*ones(length(alpha),1);
delta=beta-alpha;

xL=dl.*(cosd(delta).*cosd(beta));
xT=dt.*(sind(delta).*sind(beta));
yL=dl.*(cosd(delta).*sind(beta));
yT=dt.*(sind(delta).*cosd(beta));
hor=xL+xT;
ver=yL+yT;
xnew=x+hor;ynew=y+ver;
xy=[xnew ynew;xnew(1) ynew(1)];
sgf=sgolayfilt(xy,2,101);
plot(sgf(:,1),sgf(:,2),'b');hold on;
x=sgf(:,1);y=sgf(:,2);
pause;
end

%% %% With geometry synthet1 (No variation of Beta)


clear;clc;close all;
beta_angle=48.33;points=10000;
xstart=172;ystart=177;
dl=2;dt=1;
cd V:\ABI\JZ\Fiber_DTI\Propagation_Tool\;
p=imread('Synthet1_ST_2D_scale5.tif');
% figure,imagesc(p);
% hsv=rgb2hsv(p);
% h=hsv(:,:,1);

% ori=180.*(h)-90;

x=xstart+sin(0:1/points:2*pi)*5;
y=ystart+cos(0:1/points:2*pi)*5;
x=x';y=y';
figure,imagesc(p);colormap(jet);hold on;
plot(x,y,'r.');xlim([0 450]);ylim([0 303]);hold on;pause;
for i=1:100
cd V:\ABI\matlab_central\curvature_ANd_Normals;
N=LineNormals2D([x y]);
x1=x;x2=x-1*N(:,1);
y1=y;y2=y-1*N(:,2);
alpha=atan2d((y2-y1),(x2-x1));
beta=-beta_angle.*ones(length(alpha),1);
delta=beta-alpha;


xL=dl.*(cosd(delta).*cosd(beta));
xT=dt.*(sind(delta).*sind(beta));
yL=dl.*(cosd(delta).*sind(beta));
yT=dt.*(sind(delta).*cosd(beta));
hor=xL+xT;
ver=yL+yT;
xnew=x+hor;ynew=y+ver;
plot(xnew,ynew,'b.');hold on;
x=xnew;y=ynew;
pause;
end

%% With geometry synthet1 (simple variation of Beta)
clear;clc;close all;


points=500;
xstart=299;ystart=166;
dl=2;dt=1;
cd V:\ABI\JZ\Fiber_DTI\Propagation_Tool\;
p=imread('Synthet1_ST_2D_scale5.tif');
% figure,imagesc(p);
hsv=rgb2hsv(p);
h=hsv(:,:,1);

ori=180.*(h)-90;

x=xstart+sin(0:1/points:2*pi)*5;
y=ystart+cos(0:1/points:2*pi)*5;
x=x';y=y';
figure,imagesc(ori);colormap(jet);hold on;
plot(x,y,'r.');xlim([0 450]);ylim([0 303]);hold on;pause;
for i=1:650
cd V:\ABI\matlab_central\curvature_ANd_Normals;
N=LineNormals2D([x y]);
x1=x;x2=x-1*N(:,1);
y1=y;y2=y-1*N(:,2);
alpha=atan2d((y2-y1),(x2-x1));
beta=-diag(ori(round(y),round(x)));%-48.33.*ones(length(alpha),1);
delta=beta-alpha;


xL=dl.*(cosd(delta).*cosd(beta));
xT=dt.*(sind(delta).*sind(beta));
yL=dl.*(cosd(delta).*sind(beta));
yT=dt.*(sind(delta).*cosd(beta));
hor=xL+xT;
ver=yL+yT;
xnew=x+hor;ynew=y+ver;
plot(xnew,ynew,'b.');hold on;
x=xnew;y=ynew;
pause;
end

%% With geometry synthet2 (curvilinear variation of Beta)
clear;clc;close all;
points=500;
xstart=145;ystart=73;
dl=2;dt=1;
cd V:\ABI\JZ\Fiber_DTI\Propagation_Tool\;
p=imread('Synthet2_ST_2D_scale3.tif');
% figure,imagesc(p);
[row,col,~]=size(p);
hsv=rgb2hsv(p);
h=hsv(:,:,1);

ori=180.*(h)-90;
figure,imagesc(ori);colormap(jet);hold on;
x=xstart+sin(0:1/points:2*pi)*5;
y=ystart+cos(0:1/points:2*pi)*5;
x=x';y=y';

plot(x,y,'r.');xlim([0 col]);ylim([0 row]);hold on;pause;
for i=1:150
cd V:\ABI\matlab_central\curvature_ANd_Normals;
N=LineNormals2D([x y]);
x1=x;x2=x-1*N(:,1);
y1=y;y2=y-1*N(:,2);
alpha=atan2d((y2-y1),(x2-x1));
beta=-diag(ori(round(y),round(x)));%-48.33.*ones(length(alpha),1);
delta=beta-alpha;

xL=dl.*(cosd(delta).*cosd(beta));
xT=dt.*(sind(delta).*sind(beta));
yL=dl.*(cosd(delta).*sind(beta));
yT=dt.*(sind(delta).*cosd(beta));
hor=xL+xT;
ver=yL+yT;
xnew=x+hor;ynew=y+ver;
plot(xnew,ynew,'b.');hold on;
x=xnew;y=ynew;
pause;
end

%% With All the validation checks(tissue/non-tissue ? Outside/Inside of booundary etc)
clear;clc;close all;
points=100;


dl=2;dt=1;
cd V:\ABI\JZ\Fiber_DTI\Propagation_Tool\;
p=imread('Synthet1_ST_2D_scale5.tif');xstart=228;ystart=214;
% p=imread('Synthet2_ST_2D_scale3.tif');xstart=145;ystart=73;
% figure,imagesc(p);
[row,col,~]=size(p);
hsv=rgb2hsv(p);
h=hsv(:,:,1);

ori=180.*(h)-90;
figure,imagesc(ori);colormap(jet);hold on;

propgeo=ones(row,col);
p1=squeeze(p(:,:,1));p2=squeeze(p(:,:,2));p3=squeeze(p(:,:,3));
propgeo(p1==0 & p2==0 & p3==0 & ori==-90)=0;
propgeo(1:2,:)=0;propgeo(row-1:row,:)=0;
propgeo(:,1:2)=0;propgeo(:,col-1:col)=0;

x=xstart+sin(0:1/points:2*pi)*3;
y=ystart+cos(0:1/points:2*pi)*3;
x=x';y=y';


plot(x,y,'r.');xlim([0 col]);ylim([0 row]);hold on;pause;
for i=1:650
cd V:\ABI\matlab_central\curvature_ANd_Normals;
N=LineNormals2D([x y]);
x1=x;x2=x-1*N(:,1);
y1=y;y2=y-1*N(:,2);
alpha=atan2d((y2-y1),(x2-x1));
beta=-diag(ori(round(y),round(x)));%-48.33.*ones(length(alpha),1);
delta=beta-alpha;

xL=dl.*(cosd(delta).*cosd(beta));
xT=dt.*(sind(delta).*sind(beta));
yL=dl.*(cosd(delta).*sind(beta));
yT=dt.*(sind(delta).*cosd(beta));
hor=xL+xT;
ver=yL+yT;
xnew=x+hor;ynew=y+ver;
plot(xnew,ynew,'b.');hold on;
Eliminator(:,1)=round(xnew);
Eliminator(:,2)=round(ynew);
Eliminator(:,3)=diag(propgeo(Eliminator(:,2),Eliminator(:,1)));
x=xnew(all(Eliminator,2),:);
y=ynew(all(Eliminator,2),:);
clearvars -except x y ori propgeo dl dt;
pause;
end


%% 
spcv=cscvn(xy'); % doing spline interpolation
spcords=fnplt(spcv); % Making spline coords
uniqcords=unique(spcords','rows','stable'); % Taking unique spline coords
figure,plot(uniqcords(:,1),uniqcords(:,2));
sgf=sgolayfilt(xy,2,11);figure,plot(sgf(:,1),sgf(:,2));


%%  24 May 2018

clear;clc;close all;
beta_angle=50;points=100;
dl=2;dt=1;
xstart=100;ystart=100;
col=200;row=200;
x=xstart+sin(0:1/points:2*pi)*50;
y=ystart+cos(0:1/points:2*pi)*50;
x=x';y=y';
BW=poly2mask(x,y,row,col);
figure,imagesc(BW);colormap(jet);set(gca,'Ydir','reverse');
hold on;plot(x,y,'w.');xlim([0 col]);ylim([0 row]);
[heapy,heapx]=find(BW);
% factor1=cos(beta_angle/28.64785);

for i=1:100
cd V:\ABI\matlab_central\curvature_ANd_Normals;
N=LineNormals2D([x y]);
x1=x;x2=x-1.*N(:,1);
y1=y;y2=y+1.*N(:,2);
alpha=atan2d((y2-y1),(x2-x1));
beta=beta_angle.*ones(length(alpha),1);
delta=ones(length(x),1);
% for t=1:length(beta)
%     if(alpha(t)>0)
%     delta(t)=abs(beta(t)-alpha(t));
%     else
%     delta(t)=-abs(beta(t)-alpha(t));
%     end
% end
%    delta=beta-alpha;
if ((beta_angle>45 && beta_angle<135)||(beta_angle>-135 && beta_angle<-45))
    delta=beta-alpha;
end
if ((beta_angle<=45 && beta_angle>=-45)||(beta_angle<=-135 && beta_angle>=-180)||(beta_angle<=180 && beta_angle>=135))
    delta=alpha-beta;
end
% for t=1:length(beta)
%     if((alpha(t)<=45 && alpha(t)>=-45)||(alpha(t)<=-135 && alpha(t)>=-180)||(alpha(t)>=135 && alpha(t)<=180))
%         delta(t)=beta(t)-alpha(t);
%     else
%         delta(t)=alpha(t)-beta(t);
%     end
% end
xL=dl.*(cosd(delta).*cosd(beta));
xT=dt.*(sind(delta).*sind(beta));
yL=dl.*(cosd(delta).*sind(beta));
yT=dt.*(sind(delta).*cosd(beta));
hor=xL+xT;
ver=yL+yT;
disp('Reached 2');
xnew=x+hor;ynew=y-ver;
[in,on]=inpolygon(xnew,ynew,x,y);
disp('Reached 3');
xnew_corrected=xnew((~in));%|(~on));
ynew_corrected=ynew((~in));%|(~on));
BWnew=poly2mask(xnew_corrected,ynew_corrected,row,col);

[newheapy,newheapx]=find(BWnew);
piledheap=[newheapy newheapx;heapy heapx];
piledheap_uniq=unique(piledheap,'rows','stable'); % Taking unique coords
disp('Reached 4');
for m=1:length(piledheap_uniq)
    BWnew(piledheap_uniq(m,1),piledheap_uniq(m,2))=1;
end
disp('Reached 5');
heapy=piledheap_uniq(:,1);
heapx=piledheap_uniq(:,2);
% BWnew2=poly2mask(piledheap_uniq(:,2),piledheap_uniq(:,1),row,col);
% (piledheap_uniq(1:end-2,1),piledheap_uniq(1:end-2,2))=1;
imagesc(BWnew);colormap(jet);
x=xnew_corrected(1:1:end);y=ynew_corrected(1:1:end);
disp(i);
pause;
end

%% 27 June 2018 Debugging above code


clear;clc;close all;
beta_angle=40;points=100;
dl=2;dt=1;
xstart=100;ystart=100;
col=200;row=200;
x=xstart+sin(0:1/points:2*pi)*50;
y=ystart+cos(0:1/points:2*pi)*50;
x=[x';x(1)];y=[y';y(1)];
BW=poly2mask(x,y,row,col);
figure,imagesc(BW);colormap(jet);set(gca,'Ydir','reverse');
for i=1:50
vertices=[x y];
cd V:\ABI\matlab_central\curvature_ANd_Normals;
N=LineNormals2D([x y]);
d=1;
xx=[vertices(1:d:end,1)-20*N(1:d:end,1) vertices(1:d:end,1)-19*N(1:d:end,1) vertices(1:d:end,1)-18*N(1:d:end,1) vertices(1:d:end,1)-17*N(1:d:end,1) vertices(1:d:end,1)-16*N(1:d:end,1) vertices(1:d:end,1)-15*N(1:d:end,1) vertices(1:d:end,1)-14*N(1:d:end,1) vertices(1:d:end,1)-13*N(1:d:end,1) vertices(1:d:end,1)-12*N(1:d:end,1) vertices(1:d:end,1)-11*N(1:d:end,1) vertices(1:d:end,1)-10*N(1:d:end,1) vertices(1:d:end,1)-9*N(1:d:end,1) vertices(1:d:end,1)-8*N(1:d:end,1) vertices(1:d:end,1)-7*N(1:d:end,1) vertices(1:d:end,1)-6*N(1:d:end,1) vertices(1:d:end,1)-5*N(1:d:end,1) vertices(1:d:end,1)-4*N(1:d:end,1) vertices(1:d:end,1)-3*N(1:d:end,1) vertices(1:d:end,1)-2*N(1:d:end,1) vertices(1:d:end,1)-1*N(1:d:end,1) vertices(1:d:end,1) vertices(1:d:end,1)+1*N(1:d:end,1) vertices(1:d:end,1)+2*N(1:d:end,1) vertices(1:d:end,1)+3*N(1:d:end,1) vertices(1:d:end,1)+4*N(1:d:end,1) vertices(1:d:end,1)+5*N(1:d:end,1) vertices(1:d:end,1)+6*N(1:d:end,1) vertices(1:d:end,1)+7*N(1:d:end,1) vertices(1:d:end,1)+8*N(1:d:end,1) vertices(1:d:end,1)+9*N(1:d:end,1) vertices(1:d:end,1)+10*N(1:d:end,1) vertices(1:d:end,1)+11*N(1:d:end,1) vertices(1:d:end,1)+12*N(1:d:end,1) vertices(1:d:end,1)+13*N(1:d:end,1) vertices(1:d:end,1)+14*N(1:d:end,1) vertices(1:d:end,1)+15*N(1:d:end,1) vertices(1:d:end,1)+16*N(1:d:end,1) vertices(1:d:end,1)+17*N(1:d:end,1) vertices(1:d:end,1)+18*N(1:d:end,1) vertices(1:d:end,1)+19*N(1:d:end,1) vertices(1:d:end,1)+20*N(1:d:end,1)]';
yy=[vertices(1:d:end,2)-20*N(1:d:end,2) vertices(1:d:end,2)-19*N(1:d:end,2) vertices(1:d:end,2)-18*N(1:d:end,2) vertices(1:d:end,2)-17*N(1:d:end,2) vertices(1:d:end,2)-16*N(1:d:end,2) vertices(1:d:end,2)-15*N(1:d:end,2) vertices(1:d:end,2)-14*N(1:d:end,2) vertices(1:d:end,2)-13*N(1:d:end,2) vertices(1:d:end,2)-12*N(1:d:end,2) vertices(1:d:end,2)-11*N(1:d:end,2) vertices(1:d:end,2)-10*N(1:d:end,2) vertices(1:d:end,2)-9*N(1:d:end,2) vertices(1:d:end,2)-8*N(1:d:end,2) vertices(1:d:end,2)-7*N(1:d:end,2) vertices(1:d:end,2)-6*N(1:d:end,2) vertices(1:d:end,2)-5*N(1:d:end,2) vertices(1:d:end,2)-4*N(1:d:end,2) vertices(1:d:end,2)-3*N(1:d:end,2) vertices(1:d:end,2)-2*N(1:d:end,2) vertices(1:d:end,2)-1*N(1:d:end,2) vertices(1:d:end,2) vertices(1:d:end,2)+1*N(1:d:end,2) vertices(1:d:end,2)+2*N(1:d:end,2) vertices(1:d:end,2)+3*N(1:d:end,2) vertices(1:d:end,2)+4*N(1:d:end,2) vertices(1:d:end,2)+5*N(1:d:end,2) vertices(1:d:end,2)+6*N(1:d:end,2) vertices(1:d:end,2)+7*N(1:d:end,2) vertices(1:d:end,2)+8*N(1:d:end,2) vertices(1:d:end,2)+9*N(1:d:end,2) vertices(1:d:end,2)+10*N(1:d:end,2) vertices(1:d:end,2)+11*N(1:d:end,2) vertices(1:d:end,2)+12*N(1:d:end,2) vertices(1:d:end,2)+13*N(1:d:end,2) vertices(1:d:end,2)+14*N(1:d:end,2) vertices(1:d:end,2)+15*N(1:d:end,2) vertices(1:d:end,2)+16*N(1:d:end,2) vertices(1:d:end,2)+17*N(1:d:end,2) vertices(1:d:end,2)+18*N(1:d:end,2) vertices(1:d:end,2)+19*N(1:d:end,2) vertices(1:d:end,2)+20*N(1:d:end,2)]';
x1=x;x2=x-1.*N(:,1);
y1=y;y2=y+1.*N(:,2);
alpha=atan2d((y2-y1),(x2-x1));
% hold on; plot(xx,yy,'g.','LineWidth',1);
hold on;plot(x,y,'w.');xlim([0 col]);ylim([0 row]);
beta=beta_angle.*ones(length(alpha),1);
delta=beta-alpha;
% delta=alpha-beta;
% delta=ones(length(x),1);
% for t=1:length(beta)
%     if((alpha(t)<=45 && alpha(t)>=0))%||(alpha(t)<=-135 && alpha(t)>=-180)||(alpha(t)>=135 && alpha(t)<=180))
%         delta(t)=alpha(t)-beta(t);
%     else
%         delta(t)=beta(t)-alpha(t);
%     end
% end

xL=dl.*(cosd(delta).*cosd(beta));
xT=dt.*(sind(delta).*sind(beta));
yL=dl.*(cosd(delta).*sind(beta));
yT=dt.*(sind(delta).*cosd(beta));
hor=xL+xT;
% ver=yL+yT;
ver=-sind(beta-delta)-(dl-1).*(cosd(delta).*sind(beta));
xnew=x+hor;ynew=y+ver;

% plot(xnew,ynew,'m.');
% [in,on]=inpolygon(xnew,ynew,x,y);
% ynew(in|on)=y(in|on)-ver(in|on);
plot(xnew,ynew,'g.');
% BWnew=poly2mask(xnew,ynew,row,col);
% imagesc(BWnew);colormap(jet);
% xnew_corrected=xnew((~in));%|(~on));
% ynew_corrected=ynew((~in));%|(~on));
clear x;clear y;
x=xnew;y=ynew;
clear xnew;clear ynew;
pause;
end

%% 29 June 2018 <<Working code : Propagating region>>


clear;clc;close all;
beta_angle=240;points=100;
dl=3;dt=1;minvel=min(dl,dt);maxvel=max(dl,dt);
xstart=100;ystart=100;
col=200;row=200;
x=xstart+sin(0:1/points:2*pi)*20;
y=ystart+cos(0:1/points:2*pi)*20;
x=[x';x(1)];y=[y';y(1)];
BW=poly2mask(x,y,row,col);
figure,imagesc(BW);colormap(jet);set(gca,'Ydir','reverse');
hold on;plot(x,y,'w.');xlim([0 col]);ylim([0 row]);
[heapy,heapx]=find(BW);
pause;
for i=1:100
vertices=[x y];
cd V:\ABI\matlab_central\curvature_ANd_Normals;
N=LineNormals2D([x y]);
d=1;
xx=[vertices(1:d:end,1)-20*N(1:d:end,1) vertices(1:d:end,1)-19*N(1:d:end,1) vertices(1:d:end,1)-18*N(1:d:end,1) vertices(1:d:end,1)-17*N(1:d:end,1) vertices(1:d:end,1)-16*N(1:d:end,1) vertices(1:d:end,1)-15*N(1:d:end,1) vertices(1:d:end,1)-14*N(1:d:end,1) vertices(1:d:end,1)-13*N(1:d:end,1) vertices(1:d:end,1)-12*N(1:d:end,1) vertices(1:d:end,1)-11*N(1:d:end,1) vertices(1:d:end,1)-10*N(1:d:end,1) vertices(1:d:end,1)-9*N(1:d:end,1) vertices(1:d:end,1)-8*N(1:d:end,1) vertices(1:d:end,1)-7*N(1:d:end,1) vertices(1:d:end,1)-6*N(1:d:end,1) vertices(1:d:end,1)-5*N(1:d:end,1) vertices(1:d:end,1)-4*N(1:d:end,1) vertices(1:d:end,1)-3*N(1:d:end,1) vertices(1:d:end,1)-2*N(1:d:end,1) vertices(1:d:end,1)-1*N(1:d:end,1) vertices(1:d:end,1) vertices(1:d:end,1)+1*N(1:d:end,1) vertices(1:d:end,1)+2*N(1:d:end,1) vertices(1:d:end,1)+3*N(1:d:end,1) vertices(1:d:end,1)+4*N(1:d:end,1) vertices(1:d:end,1)+5*N(1:d:end,1) vertices(1:d:end,1)+6*N(1:d:end,1) vertices(1:d:end,1)+7*N(1:d:end,1) vertices(1:d:end,1)+8*N(1:d:end,1) vertices(1:d:end,1)+9*N(1:d:end,1) vertices(1:d:end,1)+10*N(1:d:end,1) vertices(1:d:end,1)+11*N(1:d:end,1) vertices(1:d:end,1)+12*N(1:d:end,1) vertices(1:d:end,1)+13*N(1:d:end,1) vertices(1:d:end,1)+14*N(1:d:end,1) vertices(1:d:end,1)+15*N(1:d:end,1) vertices(1:d:end,1)+16*N(1:d:end,1) vertices(1:d:end,1)+17*N(1:d:end,1) vertices(1:d:end,1)+18*N(1:d:end,1) vertices(1:d:end,1)+19*N(1:d:end,1) vertices(1:d:end,1)+20*N(1:d:end,1)]';
yy=[vertices(1:d:end,2)-20*N(1:d:end,2) vertices(1:d:end,2)-19*N(1:d:end,2) vertices(1:d:end,2)-18*N(1:d:end,2) vertices(1:d:end,2)-17*N(1:d:end,2) vertices(1:d:end,2)-16*N(1:d:end,2) vertices(1:d:end,2)-15*N(1:d:end,2) vertices(1:d:end,2)-14*N(1:d:end,2) vertices(1:d:end,2)-13*N(1:d:end,2) vertices(1:d:end,2)-12*N(1:d:end,2) vertices(1:d:end,2)-11*N(1:d:end,2) vertices(1:d:end,2)-10*N(1:d:end,2) vertices(1:d:end,2)-9*N(1:d:end,2) vertices(1:d:end,2)-8*N(1:d:end,2) vertices(1:d:end,2)-7*N(1:d:end,2) vertices(1:d:end,2)-6*N(1:d:end,2) vertices(1:d:end,2)-5*N(1:d:end,2) vertices(1:d:end,2)-4*N(1:d:end,2) vertices(1:d:end,2)-3*N(1:d:end,2) vertices(1:d:end,2)-2*N(1:d:end,2) vertices(1:d:end,2)-1*N(1:d:end,2) vertices(1:d:end,2) vertices(1:d:end,2)+1*N(1:d:end,2) vertices(1:d:end,2)+2*N(1:d:end,2) vertices(1:d:end,2)+3*N(1:d:end,2) vertices(1:d:end,2)+4*N(1:d:end,2) vertices(1:d:end,2)+5*N(1:d:end,2) vertices(1:d:end,2)+6*N(1:d:end,2) vertices(1:d:end,2)+7*N(1:d:end,2) vertices(1:d:end,2)+8*N(1:d:end,2) vertices(1:d:end,2)+9*N(1:d:end,2) vertices(1:d:end,2)+10*N(1:d:end,2) vertices(1:d:end,2)+11*N(1:d:end,2) vertices(1:d:end,2)+12*N(1:d:end,2) vertices(1:d:end,2)+13*N(1:d:end,2) vertices(1:d:end,2)+14*N(1:d:end,2) vertices(1:d:end,2)+15*N(1:d:end,2) vertices(1:d:end,2)+16*N(1:d:end,2) vertices(1:d:end,2)+17*N(1:d:end,2) vertices(1:d:end,2)+18*N(1:d:end,2) vertices(1:d:end,2)+19*N(1:d:end,2) vertices(1:d:end,2)+20*N(1:d:end,2)]';
x1=x;x2=x-1.*N(:,1);
y1=y;y2=y+1.*N(:,2);
alpha=atan2d((y2-y1),(x2-x1));
% hold on; plot(xx,yy,'g.','LineWidth',1);
hold on;plot(x,y,'w.');xlim([0 col]);ylim([0 row]);
beta=beta_angle.*ones(length(alpha),1);
delta=beta-alpha;
xL=dl.*(cosd(delta).*cosd(beta));
xT=dt.*(sind(delta).*sind(beta));
% yL=dl.*(cosd(delta).*sind(beta));
% yT=dt.*(sind(delta).*cosd(beta));
hor=xL+xT;
% ver=yT-yL;
ver=-minvel.*sind(beta-delta)-(maxvel-minvel).*(cosd(delta).*sind(beta));%-(dl).*sind(alpha);
% sin(A-B)=sinA cosB - cosA sinB;
xnew=x+hor;ynew=y+ver; 

% plot(xnew,ynew,'m.');

% ynew(in|on)=y(in|on)-ver(in|on);
plot(xnew,ynew,'g.');
% BWnew=poly2mask(xnew,ynew,row,col);
% imagesc(BWnew);colormap(jet);
% xnew_corrected=xnew((~in));%|(~on));
% ynew_corrected=ynew((~in));%|(~on));

[in,on]=inpolygon(xnew,ynew,x,y);
xnew_corrected=xnew((~in));%|(~on));
ynew_corrected=ynew((~in));%|(~on));
BWnew=poly2mask(xnew_corrected,ynew_corrected,row,col);

[newheapy,newheapx]=find(BWnew);
piledheap=[newheapy newheapx;heapy heapx];
piledheap_uniq=unique(piledheap,'rows','stable'); % Taking unique coords
% disp('Reached 4');
for m=1:length(piledheap_uniq)
    BWnew(piledheap_uniq(m,1),piledheap_uniq(m,2))=1;
end
% disp('Reached 5');
heapy=piledheap_uniq(:,1);
heapx=piledheap_uniq(:,2);
% BWnew2=poly2mask(piledheap_uniq(:,2),piledheap_uniq(:,1),row,col);
% (piledheap_uniq(1:end-2,1),piledheap_uniq(1:end-2,2))=1;
imagesc(BWnew);colormap(jet);
x=xnew_corrected(1:1:end);y=ynew_corrected(1:1:end);
disp(i);
pause;
clear xnew;clear ynew;

end

%%

clear;clc;close all;
cd /hpc/btho733/ABI/JZ/Fiber_DTI/Propagation_Tool/sections/largesection/filtered/s1/;
im=imread('cut_0050.tif');
im=fatsave;
lab_he = rgb2lab(im);
ab = lab_he(:,:,2:3);
nrows = size(ab,1);
ncols = size(ab,2);
ab = reshape(ab,nrows*ncols,2);nColors = 2;

[cluster_idx, cluster_center] = kmeans(ab,nColors,'distance','sqEuclidean', ...
                                      'Start','plus','Replicates',3);
                                  
   pixel_labels = reshape(cluster_idx,nrows,ncols);
% figure,imshow(pixel_labels,[]); title('image labeled by cluster index');   

segmented_images = cell(1,3);
rgb_label = repmat(pixel_labels,[1 1 3]);

for k = 1:nColors
    color = im;
    color(rgb_label ~= k) = 255;
    segmented_images{k} = color;
end

figure,imagesc(segmented_images{1}); title('objects in cluster 1');

figure,imagesc(segmented_images{2}); title('objects in cluster 2');
figure,imagesc(im);
