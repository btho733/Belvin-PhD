% edge tracking
clc;clear;close all;
imno=560;target=561;
imname=sprintf('s4_%05d.png',imno);
sobname=sprintf('sobel_%05d.png',imno);
%cd ua_newalgo_sobel; im=imread('sobel_00560.png'); cd ..;
cd d_images2dcoarsesegment_s4; im=imread(imname); cd ..;
cd ua_newalgo_sobel;sob=imread(sobname);cd ..;
sob1=sob(:,:,1);sob2=sob(:,:,2);sob3=sob(:,:,3);
im1=im(1:1300,2051:3550,1);
[gr,ang1]=imgradient(im1);
x11=837;
%837;475;%878;453;%880;451;%882;449;%884;446;%910;411;%911;408;%910;407;%921;387;
%925;%914;%913;%912;%911;%896;%879;%871;%873;%875;%853;%544;%1251;%305;%544;%858;%1288;%309;%1277;%1197;%1033;%995;%;%508;%550;%448;%331;%309;%233;%330;%435;%;%571;%1157;%1293;%1015;
y11=475;
%381;%399;%403;%406;%409;%433;%451;%458;%457;%457;%464;%500;%194;%958;%500;%956;%452;%968;%391;%157;%77;%175;%403;%409;%490;%814;%869;%887;%889;%993;%1085;%1096;%1114;%677;%422;%145;
k=0;
input=[0 0];
allangles=zeros(100,1);
x1=zeros(100,1);y1=zeros(100,1);
for i=1:25000
x1(i)=x11;y1(i)=y11;   % Now v r @ point(x11,y11)

actual_angle=ang1(y11,x11);
angle=PredictedAngle(gr,x11,y11,actual_angle,2,5);%ang1(y11,x11);%(atan2(ux(i),uy(i))+pi/2)*(180/pi);    % Angle at (x11,y11)
allangles(i)=angle;dist=3;
[x2,y2]=LocationAfterMoveAtAngle(x1(i),y1(i),angle,dist);
ys=[-1 -1 -1;-0 0 0;1 1 1];         % to find neighbours of (x2,y2) : y coordinates
newy2=y2+ys;
xs=[-1 0 1;-1 0 1;-1 0 1];          % to find neighbours of (x2,y2) : x coordinates
newx2=x2+xs;
coords=[newy2(:,1) newx2(:,1);newy2(:,2) newx2(:,2);newy2(:,3) newx2(:,3)]; % Making the matrix to look like the co-ordinate locations
values=diag(gr(coords(:,1),coords(:,2)));    % find corresponding gradient values
% values1=diag(sob1(coords(:,1),coords(:,2)));
% values2=diag(sob2(coords(:,1),coords(:,2)));
% values3=diag(sob3(coords(:,1),coords(:,2)));
% newcoords=coords(max(find((values1>100)&(values2>60)&(values3<100))),:);
newcoords=coords(max(find(values==max(values))),:); % find locations in 3x3 neighbourhood with max gradient. 
                      %There may be more than one  locations. In that case , choose the last in the coords table(This is doubtful. may have to change later ?)
y11=newcoords(1,1); % Set the new co-ordinates to act as initial point for next iteration
x11=newcoords(1,2);
% y11=y2;x11=x2;
B=[x1(i) y1(i)];
Lia = ismember(input,B,'rows');
grin(i)=gr(y1(i),x1(i)); % check the gradient at current location (if gr>10)and add currrent location (not next location) to final mask input table
if (grin(i)>10)&&(abs(y11-y1(i))<20)&&(abs(x11-x1(i))<20)&&(sum(Lia)==0) % make sure that x and y do not change drastically in one step
    k=k+1;
    input(k,1)=x1(i);
    input(k,2)=y1(i);
else
    break;
end
end

figure,imagesc(sob);hold on;
plot(input(:,1),input(:,2),'w-');hold on;title(['Image no   ' num2str(imno)]);

% Script for testing IsEdge function
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
%figure,imagesc(targetangu);

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

%%
% edge tracking
clc;clear;close all;
imno=560;target=561;
imname=sprintf('s4_%05d.png',imno);
sobname=sprintf('sobel_%05d.png',imno);
%cd ua_newalgo_sobel; im=imread('sobel_00560.png'); cd ..;
cd d_images2dcoarsesegment_s4; im=imread(imname); cd ..;
cd ua_newalgo_sobel;sob=imread(sobname);cd ..;
sob1=sob(:,:,1);sob2=sob(:,:,2);sob3=sob(:,:,3);
im1=im(1:1300,2051:3550,1);
[gr,ang1]=imgradient(im1);
inputall=importdata('inputall.mat');
% x11=837;
% %837;475;%878;453;%880;451;%882;449;%884;446;%910;411;%911;408;%910;407;%921;387;
% %925;%914;%913;%912;%911;%896;%879;%871;%873;%875;%853;%544;%1251;%305;%544;%858;%1288;%309;%1277;%1197;%1033;%995;%;%508;%550;%448;%331;%309;%233;%330;%435;%;%571;%1157;%1293;%1015;
% y11=475;
% %381;%399;%403;%406;%409;%433;%451;%458;%457;%457;%464;%500;%194;%958;%500;%956;%452;%968;%391;%157;%77;%175;%403;%409;%490;%814;%869;%887;%889;%993;%1085;%1096;%1114;%677;%422;%145;
% k=0;
% input=[0 0];
% allangles=zeros(100,1);
% x1=zeros(100,1);y1=zeros(100,1);
% for i=1:25000
% x1(i)=x11;y1(i)=y11;   % Now v r @ point(x11,y11)
% 
% actual_angle=ang1(y11,x11);
% angle=PredictedAngle(gr,x11,y11,actual_angle,2,5);%ang1(y11,x11);%(atan2(ux(i),uy(i))+pi/2)*(180/pi);    % Angle at (x11,y11)
% allangles(i)=angle;dist=3;
% [x2,y2]=LocationAfterMoveAtAngle(x1(i),y1(i),angle,dist);
% ys=[-1 -1 -1;-0 0 0;1 1 1];         % to find neighbours of (x2,y2) : y coordinates
% newy2=y2+ys;
% xs=[-1 0 1;-1 0 1;-1 0 1];          % to find neighbours of (x2,y2) : x coordinates
% newx2=x2+xs;
% coords=[newy2(:,1) newx2(:,1);newy2(:,2) newx2(:,2);newy2(:,3) newx2(:,3)]; % Making the matrix to look like the co-ordinate locations
% values=diag(gr(coords(:,1),coords(:,2)));    % find corresponding gradient values
% % values1=diag(sob1(coords(:,1),coords(:,2)));
% % values2=diag(sob2(coords(:,1),coords(:,2)));
% % values3=diag(sob3(coords(:,1),coords(:,2)));
% % newcoords=coords(max(find((values1>100)&(values2>60)&(values3<100))),:);
% newcoords=coords(max(find(values==max(values))),:); % find locations in 3x3 neighbourhood with max gradient. 
%                       %There may be more than one  locations. In that case , choose the last in the coords table(This is doubtful. may have to change later ?)
% y11=newcoords(1,1); % Set the new co-ordinates to act as initial point for next iteration
% x11=newcoords(1,2);
% % y11=y2;x11=x2;
% B=[x1(i) y1(i)];
% Lia = ismember(input,B,'rows');
% grin(i)=gr(y1(i),x1(i)); % check the gradient at current location (if gr>10)and add currrent location (not next location) to final mask input table
% if (grin(i)>10)&&(abs(y11-y1(i))<20)&&(abs(x11-x1(i))<20)&&(sum(Lia)==0) % make sure that x and y do not change drastically in one step
%     k=k+1;
%     input(k,1)=x1(i);
%     input(k,2)=y1(i);
% else
%     break;
% end
% end
% 
figure,imagesc(sob);hold on;
plot(inputall(:,1),inputall(:,2),'w*');hold on;title(['Image no   ' num2str(imno)]);

% Script for testing IsEdge function
input=inputall;
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
%figure,imagesc(targetangu);

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
plot(inputall(:,1),inputall(:,2),'w*');hold on;title(['Image no   ' num2str(target)]);
figure,imagesc(targetsob);hold on;
plot(goodx,goody,'w*');title(['Image no    ' num2str(target)]);
length(goodx)/l

figure,imagesc(targetangu);colormap(jet);hold on;
plot(inputall(:,1),inputall(:,2),'w*');hold on;title(['Image no   ' num2str(target)]);


%% Making of ud_coords

clear;clc;close all;
inputall=importdata('input2_cut.mat');
for imno=560:590
    close all;
targetno=imno+1;anglethresh=1;
imname=sprintf('s4_%05d.png',imno);cd d_images2dcoarsesegment_s4; im=imread(imname); cd ..;
targetname=sprintf('s4_%05d.png',targetno);cd d_images2dcoarsesegment_s4; target=imread(targetname); cd ..;
sobname=sprintf('sobel_%05d.png',targetno);cd ua_newalgo_sobel;sobtarget=imread(sobname);cd ..;
sob1=sobtarget(:,:,1);
%sob2=sobtarget(:,:,2);sob3=sobtarget(:,:,3);
im1=im(1:1300,2051:3550,1);[grim,angim]=imgradient(im1);implot=diag(angim(inputall(:,2),inputall(:,1)));
target1=target(1:1300,2051:3550,1);target2=target(1:1300,2051:3550,2);target3=target(1:1300,2051:3550,3);
[grtarget,ang1]=imgradient(target1);[~,ang2]=imgradient(target2);[~,ang3]=imgradient(target3);
angtarget(:,:,1)=ang1;angtarget(:,:,2)=ang2;angtarget(:,:,3)=ang3;
targetangu=uint8(angtarget);
targetplot=diag(ang1(inputall(:,2),inputall(:,1)));
inputall(:,3)=implot;inputall(:,4)=targetplot;
inputall(:,5)=abs(inputall(:,3)-inputall(:,4));
Lia=zeros(length(inputall),1);
Lia(inputall(:,5)<anglethresh)=1;
inputallx=inputall(:,1);inputally=inputall(:,2);
x1=inputallx(Lia==1);x0=inputallx(Lia==0);
y1=inputally(Lia==1);y0=inputally(Lia==0);
% figure,imagesc(sobtarget);title(['Image no   ' num2str(targetno)]);hold on;plot(inputallx,inputally,'w-');hold on;plot(x0,y0,'r*');

for i=1:length(inputall)
    inputall(i,6)=PredictedAngle(grim,inputallx(i),inputally(i),inputall(i,3),2,5);
    inputall(i,7)=PredictedAngle(grtarget,inputallx(i),inputally(i),inputall(i,4),2,5);
end

inputall(:,8)=abs(inputall(:,6)-inputall(:,7));
Dir_sum=zeros(length(inputall),1);
for i=1:length(inputall)
    [BasedonActual{i}]=inORout(inputallx(i),inputally(i),inputall(i,4),sobtarget,1,2 ); %inORout based on actual angle
    [BasedonPredicted{i}]=inORout(inputallx(i),inputally(i),inputall(i,7),sobtarget,1,2 ); %inORout based on Pred_angle
    Dir_sum(i)=BasedonActual{1,i}{2,1}+BasedonPredicted{1,i}{2,1};
end

xout=inputallx(Dir_sum==2);xin=inputallx(Dir_sum==0);xon=inputallx(Dir_sum==1);
yout=inputally(Dir_sum==2);yin=inputally(Dir_sum==0);yon=inputally(Dir_sum==1);
% figure,imagesc(sobtarget);title(['Image no   ' num2str(targetno)]);hold on;plot(xout,yout,'w*');hold on;plot(xin,yin,'r*');hold on;plot(xon,yon,'g*');

xnew=inputallx;ynew=inputally;
for i=1:length(inputall)
    if(Dir_sum(i)~=1)&&(inputall(i,5)>=inputall(i,8))
      coords_in=BasedonActual{1,i}{1,1};
      [ xnew(i),ynew(i) ] = BestEdgeNeighbor2(coords_in,sobtarget,targetangu,grtarget); 
    end
    if(Dir_sum(i)~=1)&&(inputall(i,5)<inputall(i,8))
        coords_in=BasedonPredicted{1,i}{1,1};
      [ xnew(i),ynew(i) ] = BestEdgeNeighbor2(coords_in,sobtarget,targetangu,grtarget);  
    end
    if(Dir_sum(i)==1)
      coords_in=BasedonActual{1,i}{1,1};
      [ xnew(i),ynew(i) ] = BestEdgeNeighbor2(coords_in,sobtarget,targetangu,grtarget); 
    end
end
finaltest=diag(sob1(ynew,xnew));
xnew(finaltest<20)=[];ynew(finaltest<20)=[];
clear inputall;
inputall1=[xnew ynew];
inputall=unique(inputall1,'rows','stable');
c_name=sprintf('c_%05d.mat',targetno);
cd us0_coords_for_inputall_trila5_mat;
% save(c_name,'inputall');
cd ..;

figure,imagesc(sobtarget);title(['New points- Image no   ' ,num2str(targetno), 'Redline(old) White line(new)']);hold on;plot(xnew,ynew,'w*');hold on;plot(inputallx,inputally,'r*');
clearvars -except inputall imno;

end

%% Plotting 560 and 660 to compare

clc;clear;close all;
targetno=560;
sobname=sprintf('sobel_%05d.png',targetno);cd ua_newalgo_sobel;sobtarget=imread(sobname);cd ..;
c_name=sprintf('c_%05d.mat',targetno);cd ud_coords;coords=importdata(c_name);cd ..;
figure,imagesc(sobtarget);hold on;plot(coords(:,1),coords(:,2),'w*');
c_name=sprintf('c_%05d.mat',561);cd ud_coords;coords1=importdata(c_name);cd ..;
figure,imagesc(sobtarget);hold on;plot(coords1(:,1),coords1(:,2),'w*');
%% Splines
clc;clear;close all;
for targetno=561:590
close all;
sobname=sprintf('sobel_%05d.png',targetno);cd ua_newalgo_sobel;sobtarget=imread(sobname);cd ..;
c_name=sprintf('c_%05d.mat',targetno);cd us0_coords_for_inputall_trila5_mat;coords=importdata(c_name);cd ..;
% cd ul_coords_for_inputall_trial1_mat;coords2=importdata(c_name);cd ..;
coords=sgolayfilt(coords(1:end,:),3,17);
h=figure;imagesc(sobtarget);title(['Image no   ' ,num2str(targetno)]);
% hold on;plot(coords(:,1),coords(:,2),'w*');

hold on;xy=coords';spcv=cscvn(xy);points=fnplt(spcv,'w',2);
% hold on;xy2=coords2';spcv2=cscvn(xy2);points2=fnplt(spcv2,'w',2);
hold on;plot(points(1,:),points(2,:),'w','LineWidth',1.6);
% hold on;plot(points2(1,:),points2(2,:),'w','LineWidth',1.6);

cd us1_spline_figures_from_us0/;saveas(h,sprintf('FIG_1_%d.tif',targetno));cd ..;

end
%% Splines_for_inputall_using uj_coordsnew;
clc;clear;close all;
for targetno=561:580
sobname=sprintf('sobel_%05d.png',targetno);cd ua_newalgo_sobel;sobtarget=imread(sobname);cd ..;
c_name=sprintf('c_%05d.mat',targetno);cd uj_coordsnew;coords=importdata(c_name);cd ..;
% coords=sgolayfilt(coords(1:end,:),2,7);
h=figure;imagesc(sobtarget);
% hold on;plot(coords(:,1),coords(:,2),'w*');
hold on;xy=coords';fnplt(cscvn(xy),'w',2)
cd uk0_Without_sgolayfilt_splines_fitted_using_uj_coords/;saveas(h,sprintf('FIG_1_%d.tif',targetno));cd ..;
close all;
end

%% Making figures in ue_figures_from_ud
clear;close all;clc;
for targetno=561:660;
sobname=sprintf('sobel_%05d.png',targetno);cd ua_newalgo_sobel;sobtarget=imread(sobname);cd ..;
c_name=sprintf('c_%05d.mat',targetno);cd ud_coords;coords=importdata(c_name); cd ..;
h=figure;
imagesc(sobtarget);hold on;plot(coords(:,1),coords(:,2),'w*');
cd ue_figures_from_ud/;saveas(h,sprintf('FIG_%d.tif',targetno));cd ..;
close all;
end
%% Making figures in uf_figure_evolutionof_surface


clear;close all;clc;coords=importdata('inputall_trial4.mat');
coords=coords(1:280,:);
for targetno=561:660;
sobname=sprintf('sobel_%05d.png',targetno);cd ua_newalgo_sobel;sobtarget=imread(sobname);cd ..;
% c_name=sprintf('c_%05d.mat',targetno);cd ud_coords;

% cd ..;
h=figure;
imagesc(sobtarget);title(['Initial points on Image no   ' ,num2str(targetno)]);hold on;plot(coords(1:10:end,1),coords(1:10:end,2),'w*');
cd ur1_figure_evolution_of_surface;saveas(h,sprintf('FIG_%d.tif',targetno));cd ..;
close all;
end
%%

coords_filtered=sgolayfilt(coords(1:end,:),3,79);
[b,gof2]=createFit(coords_filtered(1:end,1),coords_filtered(1:end,2));
% [b,gof2]=createFit(input6(:,1),input6(:,2))
yh6=b(560:1016);

% figure,imagesc(sob);
figure,imagesc(sobtarget);hold on;



hold on;
plot((560:1016),ceil(yh6),'w-');
%% Making uq2_angle_figures;
clc;clear;close all;
for imno=561:660
imname=sprintf('s4_%05d.png',imno);cd d_images2dcoarsesegment_s4; im=imread(imname); cd ..;
im1=im(501:1300,2051:2850,1);[grim,angim]=imgradient(im1);
h=figure;
imagesc(angim);colormap(jet);
cd uq2_angle_figures;saveas(h,sprintf('FIG_%d.tif',imno));cd ..;
close all;

end
%% Making angu

clc;clear;close all;
imno=561;
imname=sprintf('s4_%05d.png',imno);cd d_images2dcoarsesegment_s4; im=imread(imname); cd ..;
im1=im(1:1300,2051:3550,1);im2=im(1:1300,2051:3550,2);im3=im(1:1300,2051:3550,3);
[gr1,ang1]=imgradient(im1);[~,ang2]=imgradient(im2);[~,ang3]=imgradient(im3);
angtarget(:,:,1)=ang1;angtarget(:,:,2)=ang2;angtarget(:,:,3)=ang3;
targetangu=uint8(angtarget);
[gra,angu]=imgradient(ang1);
figure,imagesc(gra);colormap(jet);

