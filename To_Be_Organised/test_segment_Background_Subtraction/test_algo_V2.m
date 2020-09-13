% edge tracking
clc;clear;
current=560;next=561;
currentname=sprintf('s4_%05d.png',current);
currentsobname=sprintf('sobel_%05d.png',current);
cd d_images2dcoarsesegment_s4; im=imread(currentname); cd ..;
cd ua_newalgo_sobel;sob=imread(currentsobname);cd ..;
controlpts=importdata('controlpts.mat');
sob1=sob(:,:,1);sob2=sob(:,:,2);sob3=sob(:,:,3);
im1=im(1:1300,2051:3550,1);im2=im(1:1300,2051:3550,2);im3=im(1:1300,2051:3550,3);
[gr1,ang1]=imgradient(im1);[~,ang2]=imgradient(im2);[~,ang3]=imgradient(im3);
ang(:,:,1)=ang1;ang(:,:,2)=ang2;ang(:,:,3)=ang3;
angu=uint8(ang);
x1=544;y1=500;
% x1=973;y1=891;
% x1=544;y1=500;
%792;968;%321;985
%973;891;%1259;364;%1291;425;
k=0;stop=0;
input=[0 0];
dist=3;
%x1=zeros(100,1);y1=zeros(100,1);
while stop~=1   
    actual_angle=ang1(y1,x1);
    pred_angle=PredictedAngle(gr1,x1,y1,actual_angle,2,5); 
    if(abs(actual_angle-pred_angle)<80)
        [x2,y2]=LocationAfterMoveAtAngle(x1,y1,pred_angle,dist);
    else
        [x2,y2]=LocationAfterMoveAtAngle(x1,y1,actual_angle,dist);
    end
    ys=[-1 -1 -1;-0 0 0;1 1 1];         % to find neighbours of (x2,y2) : y coordinates
    newy2=y2+ys;
    xs=[-1 0 1;-1 0 1;-1 0 1];          % to find neighbours of (x2,y2) : x coordinates
    newx2=x2+xs;
    coords=[newx2(:,1) newy2(:,1) ;newx2(:,2) newy2(:,2) ;newx2(:,3) newy2(:,3) ]; % Making the matrix to look like the co-ordinate locations
    
    [x3,y3]=BestEdgeNeighbor(coords,sob,angu,gr1,actual_angle,ang1);
    Lia = ismember(input(:,1:2),[x3 y3],'rows');
   anglecontinuity=[ang1(y1,x1) ang1(y3,x3)];
    check=IsStopRequired(x1,y1,x3,y3,Lia,gr1,anglecontinuity);
    if check==0
        k=k+1;
        input(k,1)=x1;
        input(k,2)=y1; 
        input(k,3)=pred_angle;
        input(k,4)=actual_angle;
        x1=x3;y1=y3;
        disp(['Added input point ',num2str(k)]);
    elseif check >1
        stop=1;
        disp(['Error code:  ', num2str(check)]);
        
    else    
        pointsbehind=1;
        coordsbehind=input(k-pointsbehind:k,1:2);
        [x3,y3]=BestEdgeNeighbor(coordsbehind,sob,angu,gr1,actual_angle,ang1);
        Lia1=ismember(input(:,1:2),[x3 y3],'rows');
        k=find(Lia1==1);
        input=input(1:k,:); 
        pivotangle=input(k-1,3);kernelsize=20;anglestep=2;
        [All_neighborcontrolpts,stop] = NearestControlPoint(x3,y3,pivotangle,controlpts,kernelsize,anglestep);
        [x1,y1]=BestEdgeNeighbor(All_neighborcontrolpts,sob,angu,gr1,actual_angle,ang1);
              
     end
%     values=diag(gr(coords(:,1),coords(:,2)));    % find corresponding gradient values
%     newcoords=coords(max(find(values==max(values))),:); % find locations in 3x3 neighbourhood with max gradient.
    %There may be more than one  locations. In that case , choose the last in the coords table(This is doubtful. may have to change later ?)
    
end

figure,imagesc(sob);hold on;
plot(input(:,1),input(:,2),'w-');%,controlpts(:,1),controlpts(:,2),'g*');

% figure,imagesc(sob);hold on;plot(controlpts(:,1),controlpts(:,2),'g*');

%%
clc;clear;close all;
current=560;
currentsobname=sprintf('sobel_%05d.png',current);
cd ua_newalgo_sobel;sob=imread(currentsobname);cd ..;
input1=importdata('input1.mat');
input2=importdata('input2.mat');
input3=importdata('input3.mat');
input4=importdata('input4.mat');
input5=importdata('input5.mat');
input6=importdata('input6.mat');
inputall=importdata('inputall.mat');

% 
% 
input5_filtered=sgolayfilt(input5,2,15);
[a,gof1]=createFit(input5_filtered(:,1),input5_filtered(:,2))
yh5=a(445:520);

input6_filtered=sgolayfilt(input6(1:end,:),2,41);
[b,gof2]=createFit(input6_filtered(1:end,1),input6_filtered(1:end,2))
% [b,gof2]=createFit(input6(:,1),input6(:,2))
yh6=b(1175:1284);

% figure,imagesc(sob);
figure,imagesc(sob);hold on;

plot(input1(1:235,1),input1(1:235,2),'w-',input1(245:end-5,1),input1(245:end-5,2),'b-');hold on;
plot(input2(:,1),input2(:,2),'r-');
plot(input3(:,1),input3(:,2),'g-');
plot(input4(:,1),input4(:,2),'g-');
plot(input5(1:end,1),input5(1:end,2),'w*');
plot(input6(:,1),input6(:,2),'w*');

hold on;
plot((445:520),ceil(yh5),'r-',(1175:1284),ceil(yh6),'b-');


figure,imagesc(sob);hold on;
plot(inputall(:,1),inputall(:,2),'w-');

bw1=poly2mask(inputall(1:end,1),inputall(1:end,2),1300,1500);
figure,imagesc(bw1)

current=560;
currentname=sprintf('s4_%05d.png',current);
cd d_images2dcoarsesegment_s4; im=imread(currentname); cd ..;
im1=im(1:1300,2051:3550,1);im2=im(1:1300,2051:3550,2);im3=im(1:1300,2051:3550,3);
im1(bw1==1)=255;im2(bw1==1)=255;im3(bw1==1)=255;
seg_im(:,:,1)=im1;seg_im(:,:,2)=im2;seg_im(:,:,3)=im3;
figure,imagesc(seg_im);
%%


% Script for testing continuous 3rd dimension planes(target  means next plane)
%input=onpu;
targetsobname=sprintf('sobel_%05d.png',target);
targetimname=sprintf('s4_%05d.png',target);
%cd ua_newalgo_sobel; im=imread('sobel_00560.png'); cd ..;
cd d_images2dcoarsesegment_s4; targetim=imread(targetimname); cd ..;
cd ua_newalgo_sobel; targetsob=imread(targetsobname); cd ..;

% Making unsigned angle
%cd d_images2dcoarsesegment_s4; im=imread('s4_00560.png'); cd ..;
targetim1=targetim(1:1300,2051:3550,1);
targetim2=targetim(1:1300,2051:3550,2);
targetim3=targetim(1:1300,2051:3550,3);
[gr1,ang1]=imgradient(targetim1);[gr2,ang2]=imgradient(targetim2);[gr3,ang3]=imgradient(targetim3);
ang(:,:,1)=ang1;ang(:,:,2)=ang2;ang(:,:,3)=ang3;
targetangu=uint8(ang);
figure,imagesc(targetangu);
figure,imagesc(ang1)

%script for testing isedge

l=length(input);
a=zeros(l,2);
for i=1:l
    [a(i,1),a(i,2)]=IsEdge(input(i,1),input(i,2),targetsob,targetangu);
end

u=a(:,1);
v=a(:,2);
f=find(u&v);
goodx=input(f,1);
goody=input(f,2);
figure,imagesc(targetsob);hold on;
plot(goodx,goody,'w*');title(['Image no    ' num2str(target)]);
length(goodx)/l

