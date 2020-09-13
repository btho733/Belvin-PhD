%% 29 June 2018 <<Working code : Propagating region>>


clear;clc;close all;
beta_angle=0;points=100;
dl=2;dt=1;minvel=min(dl,dt);maxvel=max(dl,dt);
xstart=100;ystart=100;
col=200;row=200;
x=xstart+sin(0:1/points:2*pi)*20;
y=ystart+cos(0:1/points:2*pi)*20;
% x=x';y=y';
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
hold on;plot(x,y,'w.');xlim([0 col]);ylim([0 row]);pause
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

% BWnew=poly2mask(xnew,ynew,row,col);
% imagesc(BWnew);colormap(jet);
% xnew_corrected=xnew((~in));%|(~on));
% ynew_corrected=ynew((~in));%|(~on));

[in,on]=inpolygon(xnew,ynew,x,y);
xnew_corrected=xnew((~in));%&(~on));
ynew_corrected=ynew((~in));%&(~on));

x=[xnew_corrected;xnew_corrected(1)];y=[ynew_corrected;ynew_corrected(1)];
BWnew=poly2mask(x,y,row,col);

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

plot(x,y,'g.');
disp(i);
pause;
clear xnew;clear ynew;

end


%% Propagation region +  With All the validation checks(tissue/non-tissue ? Outside/Inside of booundary etc)
clear;clc;close all;

points=1000;dl=2;dt=1;minvel=min(dl,dt);maxvel=max(dl,dt);

cd V:\ABI\JZ\Fiber_DTI\Propagation_Tool\;
% p=imread('Synthet1_ST_2D_scale5.tif');xstart=294;ystart=157;%2,1
p=imread('Synthet2_ST_2D_scale3.tif');xstart=200;ystart=202;%3,1;200,202
%  p=imread('Synthet3_ST_2D_scale10.tif');xstart=546;ystart=344;
[row,col,~]=size(p);

% Orientation map of the image is stored in hue plane of the ST output(p)
% Beta_angle(the curvature information) at each point of propagating
% wavefront is derived later from this Orientation map. The line below convert the
% angles from [0,1] to [-90,90].
hsb=rgb2hsv(p);h=hsb(:,:,1);ori=180.*(h)-90;
figure,imagesc(ori);colormap(jet);hold on;

propgeo=ones(row,col);
p1=squeeze(p(:,:,1));p2=squeeze(p(:,:,2));p3=squeeze(p(:,:,3));
propgeo(p1==0 & p2==0 & p3==0 & ori==-90)=0;
propgeo(1:2,:)=0;propgeo(row-1:row,:)=0;
propgeo(:,1:2)=0;propgeo(:,col-1:col)=0;
x=xstart+sin(0:1/points:2*pi)*3;
y=ystart+cos(0:1/points:2*pi)*3;
x=[x';x(1)];y=[y';y(1)];
hold on;plot(x,y,'w.');xlim([0 col]);ylim([0 row]);

BW=poly2mask(x,y,row,col);% figure,imagesc(BW);colormap(jet);set(gca,'Ydir','reverse');
[heapy,heapx]=find(BW);
%%
figure;
for i=1:500
cd V:\ABI\matlab_central\curvature_ANd_Normals;
N=LineNormals2D([x y]);
x1=x;x2=x-1.*N(:,1);
y1=y;y2=y+1.*N(:,2);
disp('reached 1');
% hold on;
% plot(x,y,'w.');xlim([0 col]);ylim([0 row]);
alpha=atan2d((y2-y1),(x2-x1));
disp('reached 2');
beta=diag(ori(round(y),round(x)));%-48.33.*ones(length(alpha),1);
%May need a NEGATIVE sign for beta.
delta=beta-alpha;
xL=dl.*(cosd(delta).*cosd(beta));
xT=dt.*(sind(delta).*sind(beta));
hor=xL+xT;
ver=-minvel.*sind(beta-delta)-(maxvel-minvel).*(cosd(delta).*sind(beta));
xnew=x+hor;ynew=y+ver;
% plot(xnew,ynew,'g.');

disp('reached 3');
Eliminator(:,1)=round(xnew);
Eliminator(:,2)=round(ynew);
Eliminator(:,3)=diag(propgeo(Eliminator(:,2),Eliminator(:,1)));
xf=xnew(all(Eliminator,2),:);
yf=ynew(all(Eliminator,2),:);

disp('reached 4');
[in,on]=inpolygon(xf,yf,x,y);
xnew_corrected=xf((~in));%|(~on));
ynew_corrected=yf((~in));%|(~on));

disp('reached 5');
x=[xnew_corrected;xnew_corrected(1)];y=[ynew_corrected;ynew_corrected(1)];
BWnew=poly2mask(x,y,row,col);
[newheapy,newheapx]=find(BWnew);
piledheap=[newheapy newheapx;heapy heapx];
piledheap_uniq=unique(piledheap,'rows','stable'); % Taking unique coords
for m=1:length(piledheap_uniq)
    BWnew(piledheap_uniq(m,1),piledheap_uniq(m,2))=1;
end
disp('reached 6');
BWnew(p1==0 & p2==0 & p3==0 & ori==-90)=0;
heapy=piledheap_uniq(:,1);
heapx=piledheap_uniq(:,2);
disp(i);
imagesc(BWnew);
pause;
clear xnew;clear ynew;clear xf;clear yf;clear Eliminator;

end