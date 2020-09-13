%% Making V1
% 
% clc;clear;close all;
% for i=560:660
%     in=sprintf('bin_%05d.png',i);cd /hpc_atog/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/s_Binary_normthresh_C;im=imread(in);cd ..;
%     out=sprintf('Scut_%05d.png',i);imout=im(1:1300,2051:3550);
%     cd /hpc_atog/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/v1_cutsection_from_S;imwrite(imout,out);cd ..;
% end
% 
%% Making V1/scale16
% 
clc;clear;close all;
for i=560:660
    in=sprintf('s4close_%05d.png',i);cd /hpc_atog/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/v1_cutsection_from_S/scale4_closed;im=imread(in);cd ..;
    out=sprintf('s16close_%05d.png',i);imout=imresize(im,0.25);
    cd /hpc_atog/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/v1_cutsection_from_S/scale16/s16_closed;imwrite(imout,out);cd ..;cd ..;
end
%% 
clc;clear;close all;
cd /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/;
imno=560;c=importdata('inputall.mat');c=unique(round(c./4),'rows','stable');
imname1=sprintf('Scut_%05d.png',imno);imname2=sprintf('Scut_%05d.png',imno+1);
orname1=sprintf('cut_%05d.png',imno);orname2=sprintf('cut_%05d.png',imno+1);
cd v1_cutsection_from_S/scale16;im1=imread(imname1);im2=imread(imname2);cd ..;cd ..;

% figure,imagesc(im1);colormap(jet);hold on;plot(c(:,1),c(:,2),'w*');
% figure,imagesc(im2);colormap(jet);hold on;plot(c(:,1),c(:,2),'w*');
c=sgolayfilt(c(1:end,:),3,17);
xy=c';spcv=cscvn(xy);points=fnplt(spcv,'w',2);
% hold on;xy2=coords2';spcv2=cscvn(xy2);points2=fnplt(spcv2,'w',2);
% figure,imagesc(im1);colormap(jet);hold on;plot(points(1,:),points(2,:),'w','LineWidth',1.6);
% figure,imagesc(im2);colormap(jet);hold on;plot(points(1,:),points(2,:),'w*');
cd uu1_normalised_from_d_originals/;or1=imread(orname1);or2=imread(orname2);cd ..;
or1=imresize(or1,0.25);or2=imresize(or2,0.25);
[gr1,ang1]=imgradient(or1(:,:,1));[gr2,ang2]=imgradient(or2(:,:,1));
% figure,imagesc(gr1);colormap(jet);hold on;plot(points(1,:),points(2,:),'w*');
% figure,imagesc(ang1);colormap(jet);hold on;plot(points(1,:),points(2,:),'w*');

figure,imagesc(or1);colormap(jet);hold on;plot(points(1,:),points(2,:),'w*');

% cd /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/v1_cutsection_from_S/scale16/coords_s16_closed;save('c_00560.mat','points');

%% 
clc;clear;close all;
cd /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/;
i=560;c=importdata('inputall.mat');
imname1=sprintf('Scut_%05d.png',i);imname2=sprintf('Scut_%05d.png',i+1);
orname1=sprintf('cut_%05d.png',i);orname2=sprintf('cut_%05d.png',i+1);
cd v1_cutsection_from_S;im1=imread(imname1);im2=imread(imname2);cd ..;
cd uu1_normalised_from_d_originals/;or1=imread(orname1);or2=imread(orname2);cd ..;
[gr1,ang1]=imgradient(or1(:,:,1));[gr2,ang2]=imgradient(or2(:,:,1));
gr1(im1==255)=0;
gr2(im2==255)=0;
figure,imagesc(gr1);colormap(jet);hold on; plot(c(:,1),c(:,2),'w*');
figure,imagesc(ang1);colormap(jet);hold on; plot(c(:,1),c(:,2),'w*');
figure,imagesc(im1);colormap(jet);hold on; plot(c(:,1),c(:,2),'w*');
figure,imagesc(im2);colormap(jet);hold on; plot(c(:,1),c(:,2),'w*');
%%
c=[nextx' nexty'];
c=sgolayfilt(c(1:end,:),3,13);

xy=c';spcv=cscvn(xy);points=fnplt(spcv,'w',2);
% hold on;xy2=coords2';spcv2=cscvn(xy2);points2=fnplt(spcv2,'w',2);
figure,imagesc(im1);colormap(jet);hold on;plot(points(1,:),points(2,:),'w','LineWidth',1.6);
%% normal and offnormal markings
c=unique(round(c),'rows','stable');
c(:,3)=diag(ang1(c(:,2),c(:,1)));
c(:,4)=c(:,3)-90;
for t=1:length(c)
for d=1:6
        [x_normal(d,t),y_normal(d,t)]=LocationAfterMoveAtAngledegree(c(t,1),c(t,2),c(t,4),d);% 987,882%1185,615  % 1031,74 % 993,179
end
end
for t=1:length(c)
for d=1:6
        [x_oppnormal(d,t),y_oppnormal(d,t)]=LocationAfterMoveAtAngledegree(c(t,1),c(t,2),c(t,4)+180,d);% 987,882%1185,615  % 1031,74 % 993,179
end
end
x_normalcol=x_normal(:);y_normalcol=y_normal(:);
x_oppnormalcol=x_oppnormal(:);y_oppnormalcol=y_oppnormal(:);

% cd v1_cutsection_from_S;dim1=double(imread(imname1));dim2=double(imread(imname2));cd ..;
% sub=dim1-dim2;sub1=sub(:,:,1);
figure,imagesc(im2);colormap(jet);hold on;plot(x_normalcol,y_normalcol,'w*',x_oppnormalcol,y_oppnormalcol,'k*');

%% Normal
xpt=645;ypt=1006;
for d=1:50
        [test_normalx(d),test_normaly(d)]=LocationAfterMoveAtAngledegree(xpt,ypt,ang1(ypt,xpt)-90,d);% 987,882%1185,615  % 1031,74 % 993,179
end
test=[xpt ypt;test_normalx' test_normaly'];
test=unique(round(test),'rows','stable');
pts=diag(im2(test(:,2),test(:,1)));
l=length(pts);
figure,plot(1:l,pts);
%% off normal
xpt=645;ypt=1006;
for d=1:30
        [test_offnormalx(d),test_offnormaly(d)]=LocationAfterMoveAtAngledegree(xpt,ypt,ang1(ypt,xpt)+90,d);% 987,882%1185,615  % 1031,74 % 993,179
end
test=[xpt ypt;test_offnormalx' test_offnormaly'];
test=unique(round(test),'rows','stable');
pts=diag(im2(test(:,2),test(:,1)));
l=length(pts);
figure,plot(1:l,pts);

%%
m=movavg(double(pts),10);
[pk,lc] = findpeaks(m,1:l);
figure,plot(1:l,m);
hold on;plot(lc,pk,'k*');
findpeaks(pts,1:l,'Annotate','extents')
[tr,trlc] = findpeaks(-pts,1:l);
hold on;plot(trlc,-tr,'r*');title('for points moving in normal direction');


%% offnormal-current point-normal : scale 4
k=0;
c=unique(round(c),'rows','stable');
for i=1:length(c)
    
    xpt=c(i,1);ypt=c(i,2);

    
    xpt=107;ypt=210;
for d=1:20
        [test_normalx(d),test_normaly(d)]=LocationAfterMoveAtAngledegree(xpt,ypt,ang1(ypt,xpt)-90,d);% 987,882%1185,615  % 1031,74 % 993,179
end
test=[test_normalx' test_normaly'];
testnormal=unique(round(test),'rows','stable');
ptsnormal=diag(im2(testnormal(:,2),testnormal(:,1)));

for d=1:20
        [test_offnormalx(d),test_offnormaly(d)]=LocationAfterMoveAtAngledegree(xpt,ypt,ang1(ypt,xpt)+90,d);% 987,882%1185,615  % 1031,74 % 993,179
end
testoffnormal=[flipud(test_offnormalx') flipud(test_offnormaly')];
testoffnormal=unique(round(testoffnormal),'rows','stable');
ptsoffnormal=diag(im2(testoffnormal(:,2),testoffnormal(:,1)));
ptsoffnormal(ptsoffnormal==255)=150;ptsnormal(ptsnormal==255)=150;
l=length(ptsnormal)+length(ptsoffnormal)+1;
pts=[ptsoffnormal;255;ptsnormal];
figure,plot(1:l,pts);
[pk,loc]=findpeaks(double(pts));
% hold on;plot(loc,pk,'k*');
trloc=find(pts==0);
tr=pts(trloc);
% hold on;plot(trloc,tr,'m*');
testx=[testoffnormal(:,1);xpt;testnormal(:,1)];
testy=[testoffnormal(:,2);ypt;testnormal(:,2)];
if(pk(end)==255)
    k=k+1;
    nextx(k)=testx(trloc(end));
    nexty(k)=testy(trloc(end));
end
disp(i);
end

figure,imagesc(im2);colormap(jet);hold on; plot(nextx,nexty,'w*');

%% offnormal-current point-normal : scale 16

xpt=252;ypt=40;
for d=1:20
        [test_normalx(d),test_normaly(d)]=LocationAfterMoveAtAngledegree(xpt,ypt,ang1(ypt,xpt)-90,d);% 987,882%1185,615  % 1031,74 % 993,179
end
test=[test_normalx' test_normaly'];
testnormal=unique(round(test),'rows','stable');
ptsnormal=diag(im2(testnormal(:,2),testnormal(:,1)));

for d=1:20
        [test_offnormalx(d),test_offnormaly(d)]=LocationAfterMoveAtAngledegree(xpt,ypt,ang1(ypt,xpt)+90,d);% 987,882%1185,615  % 1031,74 % 993,179
end
testoffnormal=[flipud(test_offnormalx') flipud(test_offnormaly')];
testoffnormal=unique(round(testoffnormal),'rows','stable');
ptsoffnormal=diag(im2(testoffnormal(:,2),testoffnormal(:,1)));

% ptsoffnormal(ptsoffnormal==255)=150;ptsnormal(ptsnormal==255)=150;
l=length(ptsnormal)+length(ptsoffnormal)+1;
pts=[ptsoffnormal;0;ptsnormal];
figure,plot(1:l,pts);