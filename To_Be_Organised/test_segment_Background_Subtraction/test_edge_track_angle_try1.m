% edge tracking
clc;clear;close all;
imno=560;target=560;
imname=sprintf('s4_%05d.png',imno);
sobname=sprintf('sobel_%05d.png',imno);
%cd ua_newalgo_sobel; im=imread('sobel_00560.png'); cd ..;
cd d_images2dcoarsesegment_s4; im=imread(imname); cd ..;
cd ua_newalgo_sobel;sob=imread(sobname);cd ..;
sob1=sob(:,:,1);sob2=sob(:,:,2);sob3=sob(:,:,3);
im1=im(1:1300,2051:3550,1);
[gr,ang1]=imgradient(im1);
x11=1178;
%1279;395;
%837;475;%878;453;%880;451;%882;449;%884;446;%910;411;
%925;%914;%913;%912;%911;%896;%879;%871;%873;%875;%853;%544;%1251;%305;%544;%858;%1288;%309;%1277;%1197;%1033;%995;%;%508;%550;%448;%331;%309;%233;%330;%435;%;%571;%1157;%1293;%1015;
y11=644;
%381;%399;%403;%406;%409;%433;%451;%458;%457;%457;%464;%500;%194;%958;%500;%956;%452;%968;%391;%157;%77;%175;%403;%409;%490;%814;%869;%887;%889;%993;%1085;%1096;%1114;%677;%422;%145;
k=0;
input=[0 0];
allangles=zeros(100,1);
x1=zeros(100,1);y1=zeros(100,1);
flag=0;
for i=1:25000
%x1(i)=x11;y1(i)=y11;   % Now v r @ point(x11,y11)

actual_angle=ang1(y11,x11);angle_now=actual_angle;
if(flag==0)
    angle=PredictedAngle(gr,x11,y11,actual_angle,2,5);%ang1(y11,x11);%(atan2(ux(i),uy(i))+pi/2)*(180/pi);    % Angle at (x11,y11)
    
% elseif((abs(angle_now-angle_previous))<60)
%     angle=angle_now;
% else angle=allangles(k-1);x11=x22;y11=y22;flag=0;
end
% if((flag==1)&&(abs(angle_now-angle_previous)<60))
%     angle=angle_now;
% end
% if((flag==1)&&(abs(angle_now-angle_previous)>60))
%     angle=PredictedAngle(gr,x22,y22,actual_angle,2,5);
% end

dist=3;
[x2,y2]=LocationAfterMoveAtAngle(x11,y11,angle,dist);
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

x22=x11;y22=y11;
y11=newcoords(1,1); % Set the new co-ordinates to act as initial point for next iteration
x11=newcoords(1,2);
% y11=y2;x11=x2;
B=[x22 y22];
Lia = ismember(input,B,'rows');
%grin(i)=gr(y1(i),x1(i)); % check the gradient at current location (if gr>10)and add currrent location (not next location) to final mask input table
%if (grin(i)>10)&&(abs(y11-y1(i))<20)&&(abs(x11-x1(i))<20)&&(sum(Lia)==0) % make sure that x and y do not change drastically in one step
if (sum(Lia)==0)
    k=k+1;
    input(k,1)=x22;
    input(k,2)=y22;
    allangles(k)=angle;
    angle_previous=allangles(k);
else
    k=find(Lia>0,1,'first');
    input=input(1:k,:);
    %x1=x1(1:k,:);y1=y1(1:k,:);
    allangles=allangles(1:k,:);
     angle_previous=allangles(k);
   flag=1;

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

