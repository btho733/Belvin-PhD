%%  26 June 2018 --- Trying out Bruce' Normal eqn

clear;clc;close all;
beta_angle=0;points=100;
dl=2;dt=1;
xstart=100;ystart=100;
col=200;row=200;
x=xstart+sin(0:1/points:2*pi)*10;
y=ystart+cos(0:1/points:2*pi)*10;
x=x';y=y';
BW=poly2mask(x,y,row,col);
figure,imagesc(BW);colormap(jet);set(gca,'Ydir','normal');
hold on;plot(x,y,'w.');xlim([0 col]);ylim([0 row]);
[heapy,heapx]=find(BW);
for i=1:100
cd /hpc_atog/btho733/ABI/matlab_central/curvature_ANd_Normals;
% N=LineNormals2D([x y]);
% x1=x;x2=x-1*N(:,1);
% y1=y;y2=y-1*N(:,2);
% alpha=atan2d((y2-y1),(x2-x1));
for n=2:length(x)-1
    den(n)=(x(n-1)-x(n))*(x(n-1)-x(n+1));
    costheta(n)=((x(n)-x(n+1))*y(n-1)/den(n))+((2*x(n)-x(n-1)-x(n+1))*y(n)/den(n))+((x(n)-x(n-1))*y(n+1)/den(n));
    
    if(costheta(n)>1)
        costheta(n)=1;
    end
    if(costheta(n)<-1)
        costheta(n)=-1;
    end
    alph(n)=90+acosd(costheta(n));
end
alph(1)=alph(2);alph(length(x))=alph(length(x)-1);
beta=beta_angle.*ones(length(alph),1);
delta=beta-alph;
xL=dl.*(cosd(delta).*cosd(beta));
xT=dt.*(sind(delta).*sind(beta));
yL=dl.*(cosd(delta).*sind(beta));
yT=dt.*(sind(delta).*cosd(beta));
hor=xL+xT;
ver=yL+yT;
xnew=x+hor;ynew=y+ver;
[in,on]=inpolygon(xnew,ynew,x,y);
xnew_corrected=xnew((~in));%|(~on));
ynew_corrected=ynew((~in));%|(~on));
BWnew=poly2mask(round(xnew_corrected),round(ynew_corrected),row,col);

[newheapy,newheapx]=find(BWnew);
piledheap=[newheapy newheapx;heapy heapx];
piledheap_uniq=unique(piledheap,'rows','stable'); % Taking unique coords
for m=1:length(piledheap_uniq)
    BWnew(piledheap_uniq(m,1),piledheap_uniq(m,2))=1;
end
heapy=piledheap_uniq(:,1);
heapx=piledheap_uniq(:,2);
% BWnew2=poly2mask(piledheap_uniq(:,2),piledheap_uniq(:,1),row,col);
% (piledheap_uniq(1:end-2,1),piledheap_uniq(1:end-2,2))=1;
imagesc(BWnew);colormap(jet);
if(length(xnew_corrected)>20000)
x=xnew_corrected(1:1000:end,:);y=ynew_corrected(1:1000:end,:);
else
   x=xnew_corrected(1:2:end,:);y=ynew_corrected(1:2:end,:); 
end
disp(i);
pause;
clearvars -except i x y beta_angle dl dt row col heapy heapx;
end