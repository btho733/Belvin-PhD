%% Curvature as reciprocal radius of the local fitting circle
clear;clc;close all;
xy=importdata('inputall.mat');
x=xy(:,1);y=xy(:,2);
dx  = gradient(x);
ddx = gradient(dx);
dy  = gradient(y);
ddy = gradient(dy);
thr=4.2;
thr1=3.85;
num   = dx .* ddy - ddx .* dy;
denom = dx .* dx + dy .* dy;
denom = sqrt(denom);
denom = denom .* denom .* denom;
curvature = num ./ denom;
curvature(denom < 0) = NaN;
th=curvature+4;
ax=x(th>thr);ay=y(th>thr);
bx=x(th<=thr1);by=y(th<=thr1);
cd uu1_normalised_from_d_originals; im560=imread('cut_00560.png');cd ..;
figure,imagesc(im560);hold on;plot(ax,ay,'w*');%,bx,by,'r*');

%% Curvature as  (4*the area of the triangle formed by the three points )/ product of its three sides
clear;clc;close all;
xy=importdata('inputall.mat');thr=.5;
x=xy(:,1);y=xy(:,2);
for i=2:length(xy)-1
x1=xy(i-1,1);y1=xy(i-1,2);
x2=xy(i,1);y2=xy(i,2);
x3=xy(i+1,1);y3=xy(i+1,2);
K(i) = 2*abs((x2-x1).*(y3-y1)-(x3-x1).*(y2-y1)) ./ ...
  sqrt(((x2-x1).^2+(y2-y1).^2)*((x3-x1).^2+(y3-y1).^2)*((x3-x2).^2+(y3-y2).^2));
end
figure,plot(1:1283,K);

ax=x(K>thr);ay=y(K>thr);
bx=x(K<=thr);by=y(K<=thr);
cd uu1_normalised_from_d_originals; im560=imread('cut_00560.png');cd ..;
figure,imagesc(im560);hold on;plot(ax,ay,'w*');

%%

clear;clc;close all;
xy=importdata('inputall.mat');thr=.5;
x=xy(:,1);y=xy(:,2);
mx = mean(x); my = mean(y);
 X = x - mx; Y = y - my; % Get differences from means
 dx2 = mean(X.^2); dy2 = mean(Y.^2); % Get variances
 t = [X,Y]\(X.^2-dx2+Y.^2-dy2)/2; % Solve least mean squares problem
 a0 = t(1); b0 = t(2); % t is the 2 x 1 solution array [a0;b0]
 r = sqrt(dx2+dy2+a0^2+b0^2); % Calculate the radius
 a = a0 + mx; b = b0 + my; % Locate the circle's center
 curv = 1/r; % Get the curvature
figure,plot(1:length(curv),curv);
%% by moving avg of angles

clear;clc;close all;
cd uu1_normalised_from_d_originals; im560=imread('cut_00560.png');cd ..;
xy=importdata('inputall.mat');
[~,ang560]=imgradient(im560(:,:,1));
p1=diag(ang560(xy(:,2),xy(:,1)));
m1=movavg(p1,21);plot(1:1284,m1,'b');figure,plot(1:1284,p1,'r');
[pk,lc] = findpeaks(m1,1:1284);
[peak,locs]=findpeaks(m1,'MinPeakHeight',50);
%hold on;plot(locs,peak,'x')
figure,imagesc(im560);hold on;plot(xy(:,1),xy(:,2),'g*');

%% test script for radiusofcurv function

clear;clc;close all;
cd uu1_normalised_from_d_originals; im560=imread('cut_00560.png');cd ..;
xy=importdata('inputall.mat');
x=xy(:,1);y=xy(:,2);
for i=2:length(x)-1
    xset=[x(i-1);x(i);x(i+1)];
    yset=[y(i-1);y(i);y(i+1)];
[r(i),a(i),b(i)]=radiusofcurv(xset,yset);
end
rplot=(1./r)';

figure,plot(1:length(rplot),rplot);
m1=movavg(rplot,3);
m1(m1<0.7)=0;figure,plot(1:length(m1),m1,'b');
