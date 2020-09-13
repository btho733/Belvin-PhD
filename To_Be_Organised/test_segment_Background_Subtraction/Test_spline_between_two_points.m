%% Test spline between two points (Showing sudden phase changes blocking the use of this approach)
clc;close all;clear;
imno=640;
imname=sprintf('s4_%05d.png',imno);cd d_images2dcoarsesegment_s4; im=imread(imname); cd ..;

sobname=sprintf('sobel_%05d.png',imno);cd ua_newalgo_sobel;sob=imread(sobname);cd ..;
c_name=sprintf('c_%05d.mat',imno);cd ud_coords/;coords=importdata(c_name);cd ..;


im1=im(1:1300,2051:3550,1);[grim,angim]=imgradient(im1);coords(:,3)=diag(angim(coords(:,2),coords(:,1)));
coords_cut=[coords(12:42,:)];%;coords(43:45,:)%coords;%
%coords_cut=sgolayfilt(coords(1:end,:),3,11);
% coords_cut=coords_cut(17:end,:);
figure,imagesc(sob);hold on;plot(coords_cut(:,1),coords_cut(:,2),'b*');
diff=1;
splinecoords=[0 0];
for i=1:diff:length(coords_cut)-1
    xdiff=abs(coords_cut(i,1)-coords_cut(i+1,1));
    ydiff=abs(coords_cut(i,2)-coords_cut(i+1,2));
    if(xdiff<ydiff)
        x1=coords_cut(i,1);y1=coords_cut(i,2);dy1=tand(coords_cut(i,3)-45);
        x2=coords_cut(i+1,1);y2=coords_cut(i+1,2);dy2=tand(coords_cut(i+1,3)+45);
        A=[1/(x1*x1*x1) x1/(x1*x1*x1) x1*x1/(x1*x1*x1) x1*x1*x1/(x1*x1*x1);1/(x2*x2*x2) x2/(x2*x2*x2) x2*x2/(x2*x2*x2) x2*x2*x2/(x2*x2*x2);0/(3*x1*x1) 1/(3*x1*x1) 2*x1/(3*x1*x1) 3*x1*x1/(3*x1*x1);0/(3*x2*x2) 1/(3*x2*x2) 2*x2/(3*x2*x2) 3*x2*x2/(3*x2*x2)];
        B=[y1/(x1*x1*x1);y2/(x2*x2*x2);dy1/(3*x1*x1);dy2/(3*x2*x2)];
%         A=[1 x1 x1*x1 x1*x1*x1;1 x2 x2*x2 x2*x2*x2;0 1 2*x1 3*x1*x1;0 1 2*x2 3*x2*x2];
%         B=[y1;y2;dy1;dy2];
        sol=linsolve(A,B);
        a=sol(1);b=sol(2);c=sol(3);d=sol(4);
        if(x1<x2)
            newx=(x1:x2)';
        else
            newx=(x2:x1)';
            newx=flipud(newx);
        end
        newy=a+b.*newx+c.*newx.*newx+d.*newx.*newx.*newx;
    else
        x1=coords_cut(i,1);y1=coords_cut(i,2);dy1=tand(coords_cut(i,3));
        x2=coords_cut(i+1,1);y2=coords_cut(i+1,2);dy2=tand(coords_cut(i+1,3));
        A=[1/(y1*y1*y1) y1/(y1*y1*y1) y1*y1/(y1*y1*y1) y1*y1*y1/(y1*y1*y1);1/(y2*y2*y2) y2/(y2*y2*y2) y2*y2/(y2*y2*y2) y2*y2*y2/(y2*y2*y2);0/(3*y1*y1) 1/(3*y1*y1) 2*y1/(3*y1*y1) 3*y1*y1/(3*y1*y1);0/(3*y2*y2) 1/(3*y2*y2) 2*y2/(3*y2*y2) 3*y2*y2/(3*y2*y2)];
        B=[x1/(y1*y1*y1);x2/(y2*y2*y2);dy1/(3*y1*y1);dy2/(3*y2*y2)];
%         A=[1 y1 y1*y1 y1*y1*y1;1 y2 y2*y2 y2*y2*y2;0 1 2*y1 3*y1*y1;0 1 2*y2 3*y2*y2];
%         B=[x1;x2;dy1;dy2];
        sol=linsolve(A,B);
        a=sol(1);b=sol(2);c=sol(3);d=sol(4);
        if(y1<y2)
            newy=(y1:y2)';
        else
            newy=(y2:y1)';
            newy=flipud(newy);
        end
        newx=a+b.*newy+c.*newy.*newy+d.*newy.*newy.*newy;
    end
    clc;
    splinecoords=[splinecoords;newx newy];
end
splinecoords=splinecoords(2:end,:);

% hold on;fnplt(cscvn(splinecoords'),'w',1);
hold on;plot(splinecoords(:,1),splinecoords(:,2),'w-');clc;
%figure,plot(1:length(coords),coords(:,3)); title('Plot of Angles');
% figure,imagesc(angim);colormap(jet)

%%

clc;close all;clear;
imno=590;
imname=sprintf('s4_%05d.png',imno);cd d_images2dcoarsesegment_s4; im=imread(imname); cd ..;

sobname=sprintf('sobel_%05d.png',imno);cd ua_newalgo_sobel;sob=imread(sobname);cd ..;
c_name=sprintf('c_%05d.mat',imno);cd uj_coordsnew/;coords=importdata(c_name);cd ..;
figure,imagesc(sob);hold on; plot(coords(:,1),coords(:,2),'w*');

im1=im(1:1300,2051:3550,1);[grim,angim]=imgradient(im1);coords(:,3)=diag(angim(coords(:,2),coords(:,1)));
% coords_cut=coords;%[coords(12:42,:)];%;coords(43:45,:)%coords;%
coords_cut=sgolayfilt(coords(1:end,:),3,11);
% coords_cut=coords_cut(17:end,:);
diff=1;
figure,imagesc(sob);
% hold on;plot(coords_cut(1:diff:end,1),coords_cut(1:diff:end,2),'b*');

splinecoords=[0 0];
for i=1:diff:length(coords_cut)-diff
    xdiff=abs(coords_cut(i,1)-coords_cut(i+1,1));
    ydiff=abs(coords_cut(i,2)-coords_cut(i+1,2));
    if(xdiff<ydiff)
        x1=coords_cut(i,1);y1=coords_cut(i,2);dy1=tand(coords_cut(i,3));
        x2=coords_cut(i+1,1);y2=coords_cut(i+1,2);dy2=tand(coords_cut(i+1,3));
        A=[1/(x1*x1*x1) x1/(x1*x1*x1) x1*x1/(x1*x1*x1) x1*x1*x1/(x1*x1*x1);1/(x2*x2*x2) x2/(x2*x2*x2) x2*x2/(x2*x2*x2) x2*x2*x2/(x2*x2*x2);0/(3*x1*x1) 1/(3*x1*x1) 2*x1/(3*x1*x1) 3*x1*x1/(3*x1*x1);0/(3*x2*x2) 1/(3*x2*x2) 2*x2/(3*x2*x2) 3*x2*x2/(3*x2*x2)];
        B=[y1/(x1*x1*x1);y2/(x2*x2*x2);dy1/(3*x1*x1);dy2/(3*x2*x2)];
%         A=[1/(x1*x1) x1/(x1*x1) x1*x1/(x1*x1) x1*x1*x1/(x1*x1);1/(x2*x2) x2/(x2*x2) x2*x2/(x2*x2) x2*x2*x2/(x2*x2);0/(3*x1*x1) 1/(3*x1*x1) 2*x1/(3*x1*x1) 3*x1*x1/(3*x1*x1);0/(3*x2*x2) 1/(3*x2*x2) 2*x2/(3*x2*x2) 3*x2*x2/(3*x2*x2)];
%         B=[y1/(x1*x1);y2/(x2*x2);dy1/(3*x1*x1);dy2/(3*x2*x2)];
        sol=linsolve(A,B);
        a=sol(1);b=sol(2);c=sol(3);d=sol(4);
        if(x1<x2)
            newx=(x1:0.1:x2)';
        else
            newx=(x2:0.1:x1)';
            newx=flipud(newx);
        end
        newy=a+b.*newx+c.*newx.*newx+d.*newx.*newx.*newx;
    else
        x1=coords_cut(i,1);y1=coords_cut(i,2);dy1=tand(coords_cut(i,3));
        x2=coords_cut(i+1,1);y2=coords_cut(i+1,2);dy2=tand(coords_cut(i+1,3));
        A=[1/(y1*y1*y1) y1/(y1*y1*y1) y1*y1/(y1*y1*y1) y1*y1*y1/(y1*y1*y1);1/(y2*y2*y2) y2/(y2*y2*y2) y2*y2/(y2*y2*y2) y2*y2*y2/(y2*y2*y2);0/(3*y1*y1) 1/(3*y1*y1) 2*y1/(3*y1*y1) 3*y1*y1/(3*y1*y1);0/(3*y2*y2) 1/(3*y2*y2) 2*y2/(3*y2*y2) 3*y2*y2/(3*y2*y2)];
        B=[x1/(y1*y1*y1);x2/(y2*y2*y2);dy1/(3*y1*y1);dy2/(3*y2*y2)];
%          A=[1/(y1*y1) y1/(y1*y1) y1*y1/(y1*y1) y1*y1*y1/(y1*y1);1/(y2*y2) y2/(y2*y2) y2*y2/(y2*y2) y2*y2*y2/(y2*y2);0/(3*y1*y1) 1/(3*y1*y1) 2*y1/(3*y1*y1) 3*y1*y1/(3*y1*y1);0/(3*y2*y2) 1/(3*y2*y2) 2*y2/(3*y2*y2) 3*y2*y2/(3*y2*y2)];
%         B=[x1/(y1*y1);x2/(y2*y2);dy1/(3*y1*y1);dy2/(3*y2*y2)];
        sol=linsolve(A,B);
        a=sol(1);b=sol(2);c=sol(3);d=sol(4);
        if(y1<y2)
            newy=(y1:0.1:y2)';
        else
            newy=(y2:0.1:y1)';
            newy=flipud(newy);
        end
        newx=a+b.*newy+c.*newy.*newy+d.*newy.*newy.*newy;
    end
    clc;
    splinecoords=[splinecoords;newx newy];
end
splinecoords=splinecoords(2:end,:);
% splinecoords=unique(splinecoords,'rows','stable');

% hold on;fnplt(cscvn(splinecoords'),'w',1);
hold on;plot(splinecoords(:,1),splinecoords(:,2),'w-');hold off;
clc;
%figure,plot(1:length(coords),coords(:,3)); title('Plot of Angles');
% figure,imagesc(angim);colormap(jet)

%% Test spline between two points

clc;clear;close all;
imno=565;
imname=sprintf('s4_%05d.png',imno);cd d_images2dcoarsesegment_s4; im=imread(imname); cd ..;
sobname=sprintf('sobel_%05d.png',imno);cd ua_newalgo_sobel;sob=imread(sobname);cd ..;
coords=importdata('testcoords_00565.mat');
testcoords_00565=[703 988;665 1005;589 1114;561 1130];%690 995;572 1127];
coords=testcoords_00565(1:end,1:2);


im1=im(1:1300,2051:3550,1);[grim,angim]=imgradient(im1);coords(:,3)=diag(angim(coords(:,2),coords(:,1)));
% coords=sgolayfilt(coords(1:end,:),2,17);
xy=coords';spcv=cscvn(xy);points=fnplt(spcv,'w',2);
% hold on;plot(points(1,:),points(2,:),'w');
figure,imagesc(sob);
hold on;plot(coords(:,1),coords(:,2),'b*');
hold on;plot(points(1,:),points(2,:),'r-','LineWidth',2);
% figure,imagesc(angim);colormap(jet);
% close all;


coords_cut=coords;%
%coords_cut=sgolayfilt(coords(1:end,:),3,11);
% coords_cut=coords_cut(17:end,:);
%figure,imagesc(sob);hold on;plot(coords_cut(:,1),coords_cut(:,2),'b*');
diff=1;
splinecoords=[0 0];
for i=1:length(coords_cut)-1
    xdiff=abs(coords_cut(i,1)-coords_cut(i+1,1));
    ydiff=abs(coords_cut(i,2)-coords_cut(i+1,2));
    if(xdiff<ydiff)
        x1=coords_cut(i,1);y1=coords_cut(i,2);dy1=tand(coords_cut(i,3)-90);
        x2=coords_cut(i+1,1);y2=coords_cut(i+1,2);dy2=tand(coords_cut(i+1,3)+90);
%         A=[1 x1 x1*x1 x1*x1*x1;1 x2 x2*x2 x2*x2*x2;0 1 2*x1 3*x1*x1;0 1 2*x2 3*x2*x2];
%         B=[y1;y2;dy1;dy2];
        A=[1/(x1*x1*x1) x1/(x1*x1*x1) x1*x1/(x1*x1*x1) x1*x1*x1/(x1*x1*x1);1/(x2*x2*x2) x2/(x2*x2*x2) x2*x2/(x2*x2*x2) x2*x2*x2/(x2*x2*x2);0/(3*x1*x1) 1/(3*x1*x1) 2*x1/(3*x1*x1) 3*x1*x1/(3*x1*x1);0/(3*x2*x2) 1/(3*x2*x2) 2*x2/(3*x2*x2) 3*x2*x2/(3*x2*x2)];
        B=[y1/(x1*x1*x1);y2/(x2*x2*x2);dy1/(3*x1*x1);dy2/(3*x2*x2)];
        sol=linsolve(A,B);
        a=sol(1);b=sol(2);c=sol(3);d=sol(4);   
        if(x1<x2)
            newx=(x1:x2)';
        else
            newx=(x2:x1)';
            newx=flipud(newx);
        end
        newy=a+b.*newx+c.*newx.*newx+d.*newx.*newx.*newx;
    else
        x1=coords_cut(i,1);y1=coords_cut(i,2);dy1=tand(coords_cut(i,3));
        x2=coords_cut(i+1,1);y2=coords_cut(i+1,2);dy2=tand(coords_cut(i+1,3));
%         A=[1 y1 y1*y1 y1*y1*y1;1 y2 y2*y2 y2*y2*y2;0 1 2*y1 3*y1*y1;0 1 2*y2 3*y2*y2];
%         B=[x1;x2;dy1;dy2];
        A=[1/(y1*y1*y1) y1/(y1*y1*y1) y1*y1/(y1*y1*y1) y1*y1*y1/(y1*y1*y1);1/(y2*y2*y2) y2/(y2*y2*y2) y2*y2/(y2*y2*y2) y2*y2*y2/(y2*y2*y2);0/(3*y1*y1) 1/(3*y1*y1) 2*y1/(3*y1*y1) 3*y1*y1/(3*y1*y1);0/(3*y2*y2) 1/(3*y2*y2) 2*y2/(3*y2*y2) 3*y2*y2/(3*y2*y2)];
        B=[x1/(y1*y1*y1);x2/(y2*y2*y2);dy1/(3*y1*y1);dy2/(3*y2*y2)];
        sol=linsolve(A,B);
        a=sol(1);b=sol(2);c=sol(3);d=sol(4);
    
        if(y1<y2)
            newy=(y1:y2)';
        else
            newy=(y2:y1)';
            newy=flipud(newy);
        end
        newx=a+b.*newy+c.*newy.*newy+d.*newy.*newy.*newy;
    end
    
    splinecoords=[splinecoords;newx newy];
end
%%
splinecoords=unique(round(splinecoords(2:end,:)),'rows','stable');

% hold on;fnplt(cscvn(splinecoords'),'w',1);    
hold on;plot(splinecoords(:,1),splinecoords(:,2),'w-','LineWidth',2);clc;

%% Convert input5/input6 to continuous spline (To remove any knots or loops)

clc;clear;close all;
imno=560;
imname=sprintf('s4_%05d.png',imno);cd d_images2dcoarsesegment_s4; im=imread(imname); cd ..;
sobname=sprintf('sobel_%05d.png',imno);cd ua_newalgo_sobel;sob=imread(sobname);cd ..;
coords=importdata('input5.mat');
coords=coords(1:end,1:2);


im1=im(1:1300,2051:3550,1);[grim,angim]=imgradient(im1);coords(:,3)=diag(angim(coords(:,2),coords(:,1)));
coords2=sgolayfilt(coords(1:end,:),3,17);
xy=coords';spcv=cscvn(xy);points=fnplt(spcv,'w',1);
xy2=coords2';spcv2=cscvn(xy2);points2=fnplt(spcv2,'w',1);
% hold on;plot(points(1,:),points(2,:),'w');
figure,imagesc(sob);
hold on;plot(coords(:,1),coords(:,2),'b*');
hold on;plot(points(1,:),points(2,:),'g-','LineWidth',1.2);
hold on;plot(points2(1,:),points2(2,:),'w-','LineWidth',2);
poi=points2';
poi=poi(:,1:2);
input6_filt_splinefit=unique(poi,'rows','stable');
% save('input6_filt_splinefit.mat','input6_filt_splinefit');
