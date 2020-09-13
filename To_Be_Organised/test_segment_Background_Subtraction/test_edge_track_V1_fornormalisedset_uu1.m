%% Making of uu3_coords

clear;clc;close all;
inputall=importdata('inputall.mat');
for imno=560:659
    close all;
targetno=imno+1;anglethresh=1;
imname=sprintf('cut_%05d.png',imno);cd uu1_normalised_from_d_originals; im=imread(imname); cd ..;
targetname=sprintf('cut_%05d.png',targetno);cd uu1_normalised_from_d_originals; target=imread(targetname); cd ..;
sobname=sprintf('sobel_%05d.png',targetno);cd uu2_sobel_of_uu1;sobtarget=imread(sobname);cd ..;
sob1=sobtarget(:,:,1);
%sob2=sobtarget(:,:,2);sob3=sobtarget(:,:,3);
im1=im(:,:,1);[grim,angim]=imgradient(im1);implot=diag(angim(inputall(:,2),inputall(:,1)));
target1=target(:,:,1);target2=target(:,:,2);target3=target(:,:,3);
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
    [BasedonActual{i}]=inORout(inputallx(i),inputally(i),inputall(i,4),sobtarget,3,2 ); %inORout based on actual angle
    [BasedonPredicted{i}]=inORout(inputallx(i),inputally(i),inputall(i,7),sobtarget,3,2 ); %inORout based on Pred_angle
    Dir_sum(i)=BasedonActual{1,i}{2,1}+BasedonPredicted{1,i}{2,1};
end

xout=inputallx(Dir_sum==2);xin=inputallx(Dir_sum==0);xon=inputallx(Dir_sum==1);
yout=inputally(Dir_sum==2);yin=inputally(Dir_sum==0);yon=inputally(Dir_sum==1);
% figure,imagesc(sobtarget);title(['Image no   ' num2str(targetno)]);hold on;plot(xout,yout,'w*');hold on;plot(xin,yin,'r*');hold on;plot(xon,yon,'g*');

xnew=inputallx;ynew=inputally;
for i=1:length(inputall)
    if(Dir_sum(i)~=1)%&&(inputall(i,5)<=inputall(i,8))
      coords_in=BasedonActual{1,i}{1,1};
      [ xnew(i),ynew(i) ] = BestEdgeNeighbor2(coords_in,sobtarget,targetangu,grtarget); 
    end
%     if(Dir_sum(i)~=1)&&(inputall(i,5)>inputall(i,8))
%         coords_in=BasedonPredicted{1,i}{1,1};
%       [ xnew(i),ynew(i) ] = BestEdgeNeighbor2(coords_in,sobtarget,targetangu,grtarget);  
%     end
    if(Dir_sum(i)==1)
      coords_in=BasedonActual{1,i}{1,1};
      [ xnew(i),ynew(i) ] = BestEdgeNeighbor2(coords_in,sobtarget,targetangu,grtarget); 
    end
end
finaltest=diag(sob1(ynew,xnew));
angletable(:,1)=xnew;angletable(:,2)=ynew;
xnew(finaltest<20)=[];ynew(finaltest<20)=[];
clear inputall;
inputall1=[xnew ynew];
inputall=unique(inputall1,'rows','stable');
c_name=sprintf('c_%05d.mat',targetno);
cd uu8a_testcoords4;
save(c_name,'inputall');
cd ..;

figure,imagesc(sobtarget);title(['New points- Image no   ' ,num2str(targetno), 'Redline(old) White line(new)']);hold on;plot(xnew,ynew,'w*');hold on;plot(inputallx,inputally,'r*');
clearvars -except inputall imno;

end

%% Splines
clc;clear;close all;
for targetno=561:660
close all;
sobname=sprintf('sobel_%05d.png',targetno);cd uu2_sobel_of_uu1;sobtarget=imread(sobname);cd ..;
c_name=sprintf('c_%05d.mat',targetno);cd uu9g_testcoords8;coords=importdata(c_name);cd ..;
% cd ul_coords_for_inputall_trial1_mat;coords2=importdata(c_name);cd ..;
coords=sgolayfilt(coords(1:end,:),2,7);
h=figure;imagesc(sobtarget);title(['Image no   ' ,num2str(targetno)]);
% hold on;plot(coords(:,1),coords(:,2),'w*');

hold on;xy=coords';spcv=cscvn(xy);points=fnplt(spcv,'w',2);
% hold on;xy2=coords2';spcv2=cscvn(xy2);points2=fnplt(spcv2,'w',2);
hold on;plot(points(1,:),points(2,:),'w','LineWidth',1.6);
% hold on;plot(points2(1,:),points2(2,:),'w','LineWidth',1.6);

cd uu9h_spline_figures_for_uu9g/;saveas(h,sprintf('FIG_1_%d.tif',targetno));cd ..;

end

%% To plot coord points on various images
clc;close all;clear;
imno=560;imcoords=importdata('inputall.mat');
targetno=561;cd uu9c_testcoords6/;cname=sprintf('c_%05d.mat',targetno);tarcoords=importdata(cname);cd ..;
cd uu1_normalised_from_d_originals;imname=sprintf('cut_%05d.png',imno);tarname=sprintf('cut_%05d.png',targetno);im=imread(imname);tar=imread(tarname);cd ..;
cd uu2_sobel_of_uu1;imsobname=sprintf('sobel_%05d.png',imno);tarsobname=sprintf('sobel_%05d.png',targetno);imsob=imread(imsobname);tarsob=imread(tarsobname);cd ..;
imsob1=imsob(:,:,1);
tarsob1=tarsob(:,:,1);tarsob2=tarsob(:,:,2);tarsob3=tarsob(:,:,3);
[targr1,tarang1]=imgradient(tar(:,:,1));
[imgr1,imang1]=imgradient(im(:,:,1));
% figure,imagesc(tarang1);hold on;plot(tarcoords(:,1),tarcoords(:,2),'w*');colormap(jet);title('gradient angles');
% figure,imagesc(targr1);hold on;plot(tarcoords(:,1),tarcoords(:,2),'w*');colormap(jet);title('gradient magnitude');
% figure,imagesc(tarsob);hold on;plot(tarcoords(:,1),tarcoords(:,2),'w*');title('sobel 3D');
% figure,imagesc(tar);hold on;plot(tarcoords(:,1),tarcoords(:,2),'w*');title('original normalised');
% 
% figure,imagesc(imang1);hold on;plot(imcoords(:,1),imcoords(:,2),'r*');colormap(jet);title(['gradient angles    ' ,num2str(imno)]);
% figure,imagesc(imgr1);hold on;plot(imcoords(:,1),imcoords(:,2),'r*');colormap(jet);title(['gradient magnitude   ' ,num2str(imno)]);
% figure,imagesc(tarsob);hold on;plot(imcoords(:,1),imcoords(:,2),'r*');title(['sobel 3D   ' ,num2str(imno)]);
% figure,imagesc(im);hold on;plot(imcoords(:,1),imcoords(:,2),'r*');title(['original normalised   ' ,num2str(imno)]);


immatrix(:,1)=imcoords(:,1);
immatrix(:,2)=imcoords(:,2);
immatrix(:,3)=diag(imsob1(immatrix(:,2),immatrix(:,1)));
figure,plot(1:length(imcoords),immatrix(:,3));




% figure,imagesc(tarang1);hold on;plot(tarcoords(:,1),tarcoords(:,2),'w*',imcoords(:,1),imcoords(:,2),'r*');colormap(jet);title('gradient angles');
% figure,imagesc(targr1);hold on;plot(tarcoords(:,1),tarcoords(:,2),'w*',imcoords(:,1),imcoords(:,2),'r*');colormap(jet);title('gradient magnitude');
figure,imagesc(tarsob);hold on;plot(tarcoords(:,1),tarcoords(:,2),'w*',imcoords(:,1),imcoords(:,2),'r*');title('red(old) and white(new)');
% figure,imagesc(tar);hold on;plot(tarcoords(:,1),tarcoords(:,2),'w*',imcoords(:,1),imcoords(:,2),'r*');title('original normalised');

coords=sgolayfilt(tarcoords(1:end,:),2,7);
h=figure;imagesc(tarsob);title(['Image no   ' ,num2str(targetno)]);
% hold on;plot(coords(:,1),coords(:,2),'w*');

xy=coords';spcv=cscvn(xy);points=fnplt(spcv,'w',2);
hold on;plot(points(1,:),points(2,:),'w*');

%% Trying to incorporate angle

clear;clc;
inputall=importdata('inputall.mat');
 imno=560;
    close all;
targetno=imno+1;anglethresh=150;
imname=sprintf('cut_%05d.png',imno);cd uu1_normalised_from_d_originals; im=imread(imname); cd ..;
targetname=sprintf('cut_%05d.png',targetno);cd uu1_normalised_from_d_originals; target=imread(targetname); cd ..;
sobname=sprintf('sobel_%05d.png',targetno);cd uu2_sobel_of_uu1;sobtarget=imread(sobname);cd ..;
sob1=sobtarget(:,:,1);sob2=sobtarget(:,:,2);sob3=sobtarget(:,:,3);
im1=im(:,:,1);[grim,angim]=imgradient(im1);implot=diag(angim(inputall(:,2),inputall(:,1)));
target1=target(:,:,1);target2=target(:,:,2);target3=target(:,:,3);
[grtarget,ang1]=imgradient(target1);[~,ang2]=imgradient(target2);[~,ang3]=imgradient(target3);
angtarget(:,:,1)=ang1;angtarget(:,:,2)=ang2;angtarget(:,:,3)=ang3;
targetangu=uint8(angtarget);
targetplot=diag(ang1(inputall(:,2),inputall(:,1)));
inputall(:,3)=implot;inputall(:,4)=targetplot;
inputall(:,5)=abs(inputall(:,3)-inputall(:,4));
% Lia=zeros(length(inputall),1);
% Lia(inputall(:,5)<anglethresh)=1;
inputallx=inputall(:,1);inputally=inputall(:,2);
% x1=inputallx(Lia==1);x0=inputallx(Lia==0);
% y1=inputally(Lia==1);y0=inputally(Lia==0);
% figure,imagesc(sobtarget);title(['Image no   ' num2str(targetno)]);hold on;plot(inputallx,inputally,'w-');hold on;plot(x0,y0,'r*');

for i=1:length(inputall)
    inputall(i,6)=PredictedAngle(grim,inputallx(i),inputally(i),inputall(i,3),2,5);
    inputall(i,7)=PredictedAngle(grtarget,inputallx(i),inputally(i),inputall(i,4),2,5);
end

inputall(:,8)=abs(inputall(:,6)-inputall(:,7));
Dir_sum=zeros(length(inputall),1);
for i=1:length(inputall)
    [BasedonActual{i}]=inORout(inputallx(i),inputally(i),inputall(i,3),sobtarget,3,2 ); %inORout based on actual angle
    [BasedonPredicted{i}]=inORout(inputallx(i),inputally(i),inputall(i,6),sobtarget,3,2 ); %inORout based on Pred_angle
    Dir_sum(i)=BasedonActual{1,i}{2,1}+BasedonPredicted{1,i}{2,1};
end

xout=inputallx(Dir_sum==2);xin=inputallx(Dir_sum==0);xon=inputallx(Dir_sum==1);
yout=inputally(Dir_sum==2);yin=inputally(Dir_sum==0);yon=inputally(Dir_sum==1);
% figure,imagesc(sobtarget);title(['Image no   ' num2str(targetno)]);hold on;plot(xout,yout,'w*');hold on;plot(xin,yin,'r*');hold on;plot(xon,yon,'g*');

xnew=inputallx;ynew=inputally;
for i=1:length(inputall)
    if(Dir_sum(i)~=1)%&&(inputall(i,5)<=inputall(i,8))
      coords_in=BasedonActual{1,i}{1,1};
      [ xnew(i),ynew(i) ] = BestEdgeNeighbor2(coords_in,sobtarget,targetangu,grtarget); 
    end
%     if(Dir_sum(i)~=1)&&(inputall(i,5)>inputall(i,8))
%         coords_in=BasedonPredicted{1,i}{1,1};
%       [ xnew(i),ynew(i) ] = BestEdgeNeighbor2(coords_in,sobtarget,targetangu,grtarget);  
%     end
    if(Dir_sum(i)==1)
      coords_in=BasedonActual{1,i}{1,1};
      [ xnew(i),ynew(i) ] = BestEdgeNeighbor2(coords_in,sobtarget,targetangu,grtarget); 
    end
end
finaltest=diag(sob1(ynew,xnew));
angletable(:,1)=inputall(:,1);angletable(:,2)=inputall(:,2);angletable(:,3)=xnew;angletable(:,4)=ynew;
angletable(:,5)=inputall(:,3);angletable(:,6)=diag(ang1(ynew,xnew));angletable(:,7)=abs(angletable(:,6)-angletable(:,5));
wrongsx=angletable(:,3);
wrongsy=angletable(:,4);
wrongsx=wrongsx((angletable(:,7)>anglethresh) & (finaltest<170));
wrongsy=wrongsy((angletable(:,7)>anglethresh) & (finaltest<170));
wrongs=[wrongsx wrongsy];
for n=1:length(wrongsx)
    [edgex(n),edgey(n)]=func_Edgepoints(wrongsx(n),wrongsy(n),angim,ang1,grtarget);
end
xnew(finaltest<20)=[];ynew(finaltest<20)=[];
clear inputall;
inputall=[xnew ynew];
inputall=unique(inputall,'rows','stable');
Lia1=ismember(inputall,wrongs,'rows');
inputall1=inputall(:,1);inputall2=inputall(:,2);
inputall1(Lia1==1)=[];inputall2(Lia1==1)=[];
clear inputall;inputall=[inputall1 inputall2];
% coords=sgolayfilt(inputall(1:end,:),2,7);
coords=inputall;
% h=figure;imagesc(tarsob);title(['Image no   ' ,num2str(targetno)]);
% hold on;plot(coords(:,1),coords(:,2),'w*');

xy=coords';spcv=cscvn(xy);points=fnplt(spcv,'w',2);
clear inputall;inputall=unique(round(points'),'rows','stable');


% c_name=sprintf('c_%05d.mat',targetno);
% cd uu9g_testcoords8;
% save(c_name,'inputall');
% cd ..;

figure,imagesc(sobtarget);title(['New points- Image no   ' ,num2str(targetno), 'green(old) magenta(new)']);hold on; plot(wrongsx,wrongsy,'g*',edgex,edgey,'m*');
% hold on;plot(inputall1,inputall2,'w*');hold on;plot(inputallx,inputally,'r*');
% clearvars -except inputall imno;
% end


%% correlation based image subsection search

clc;clear;close all; diffy=27;diffx=27;
im1no=560;im2no=561;
im1name=sprintf('cut_%05d.png',im1no);im2name=sprintf('cut_%05d.png',im2no);
cd uu1_normalised_from_d_originals; im1=imread(im1name); im2=imread(im2name);cd ..;
im1gr=imgradient(im1(:,:,1));im2gr=imgradient(im2(:,:,1));
figure,imagesc(im1gr);title(num2str(im1no));figure,imagesc(im2gr);title(num2str(im2no));
template=im1gr(841:845,408:412);figure,imagesc(template);title(['template from  ',num2str(im1no)]);
imsub=im2gr(841-diffy:843+diffy,408-diffx:410+diffx);figure,imagesc(imsub);title(['Image subsection of  ', num2str(im2no)]);
c = normxcorr2(template,imsub);
 c(c<0.4)=0;
figure, surf(c), colormap(jet);shading flat;

[ypeak, xpeak] = find(c==max(c(:)));
yoffSet = ypeak-size(template,1);
xoffSet = xpeak-size(template,2);
hFig = figure;
hAx  = axes;
imagesc(imsub,'Parent', hAx);
imrect(hAx, [xoffSet, yoffSet, size(template,2), size(template,1)]);title('Rectangle shows the mathcing region');

%% Direction to move : Based on interslice redplanes
clc;clear;close all;
cd uu1_normalised_from_d_originals; 
im1=double(imread('cut_00560.png'));
im2=double(imread('cut_00561.png'));cd ..;
sub=im1-im2;
figure,imagesc(sub(:,:,1))
figure,surf(sub(:,:,1));colormap(jet);shading flat;
sub1=sub(:,:,1);
sub1u=(sub1+255)./2;
figure,imagesc(sub1u);colormap(jet);
figure,imagesc(im2(:,:,1))
%% Direction to move : Based on interslice gradients
clc;clear;close all;
cd uu1_normalised_from_d_originals; 
im1=imread('cut_00560.png');
im2=imread('cut_00561.png');cd ..;
im1g=imgradient(im1(:,:,1));im2g=imgradient(im2(:,:,1));
sub=im1g-im2g;
figure,imagesc(sub);colormap(jet)
%% After rescaling 0-255
sub(sub>10)=255;
sub(sub<10 & sub>-10)=120;
sub(sub<-10)=0;
% minsub=min(min(sub));maxsub=max(max(sub));
% subu=255*((sub-minsub)./(maxsub-minsub));
figure,imagesc(sub);colormap(jet)


%% Making of uu3_coords - single section - for testing the interslice redplane difference based method

clear;clc;close all;
inputall=importdata('inputall.mat');
imno=560;
    close all;
targetno=imno+1;anglethresh=1;
imname=sprintf('cut_%05d.png',imno);cd uu1_normalised_from_d_originals; im=imread(imname); cd ..;
targetname=sprintf('cut_%05d.png',targetno);cd uu1_normalised_from_d_originals; target=imread(targetname); cd ..;
sobname=sprintf('sobel_%05d.png',targetno);cd uu2_sobel_of_uu1;sobtarget=imread(sobname);cd ..;
sob1=sobtarget(:,:,1);
%sob2=sobtarget(:,:,2);sob3=sobtarget(:,:,3);
im1=im(:,:,1);[grim,angim]=imgradient(im1);implot=diag(angim(inputall(:,2),inputall(:,1)));
target1=target(:,:,1);target2=target(:,:,2);target3=target(:,:,3);
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
    [BasedonActual{i}]=inORout(inputallx(i),inputally(i),inputall(i,4),sobtarget,3,2 ); %inORout based on actual angle
    [BasedonPredicted{i}]=inORout(inputallx(i),inputally(i),inputall(i,7),sobtarget,3,2 ); %inORout based on Pred_angle
    Dir_sum(i)=BasedonActual{1,i}{2,1}+BasedonPredicted{1,i}{2,1};
end

xout=inputallx(Dir_sum==2);xin=inputallx(Dir_sum==0);xon=inputallx(Dir_sum==1);
yout=inputally(Dir_sum==2);yin=inputally(Dir_sum==0);yon=inputally(Dir_sum==1);
% figure,imagesc(sobtarget);title(['Image no   ' num2str(targetno)]);hold on;plot(xout,yout,'w*');hold on;plot(xin,yin,'r*');hold on;plot(xon,yon,'g*');

xnew=inputallx;ynew=inputally;
for i=1:length(inputall)
    if(Dir_sum(i)~=1)%&&(inputall(i,5)<=inputall(i,8))
      coords_in=BasedonActual{1,i}{1,1};
      [ xnew(i),ynew(i) ] = BestEdgeNeighbor2(coords_in,sobtarget,targetangu,grtarget); 
    end
%     if(Dir_sum(i)~=1)&&(inputall(i,5)>inputall(i,8))
%         coords_in=BasedonPredicted{1,i}{1,1};
%       [ xnew(i),ynew(i) ] = BestEdgeNeighbor2(coords_in,sobtarget,targetangu,grtarget);  
%     end
    if(Dir_sum(i)==1)
      coords_in=BasedonActual{1,i}{1,1};
      [ xnew(i),ynew(i) ] = BestEdgeNeighbor2(coords_in,sobtarget,targetangu,grtarget); 
    end
end


figure,imagesc(sobtarget);title(['New points- Image no   ' ,num2str(targetno), 'Redline(old) White line(new)']);hold on;plot(xnew,ynew,'w*');hold on;plot(inputallx,inputally,'r*');
figure,imagesc(im1);title(['New points- Image no   ' ,num2str(imno), 'Redline(old) White line(new)']);hold on;plot(xnew,ynew,'w*');hold on;plot(inputallx,inputally,'r*');
figure,imagesc(angim);hold on;plot(inputallx,inputally,'r*');
%%  With points making mistake

% clc;clear;close all;
cd uu1_normalised_from_d_originals; im560=double(imread('cut_00560.png'));im561=double(imread('cut_00561.png'));cd ..;
[~,angim]=imgradient(im560(:,:,1));
t1=[402 839;406 839;407 840;410 842;413 841];
t1(:,3)=diag(angim(t1(:,2),t1(:,1)));
t1(:,4)=t1(:,3)-90;
for t=1:length(t1)
for d=1:30
        [x_normal(d,t),y_normal(d,t)]=LocationAfterMoveAtAngledegree(t1(t,1),t1(t,2),t1(t,4),d);% 987,882%1185,615  % 1031,74 % 993,179
end
end
for t=1:length(t1)
for d=1:30
        [x_oppnormal(d,t),y_oppnormal(d,t)]=LocationAfterMoveAtAngledegree(t1(t,1),t1(t,2),t1(t,4)+180,d);% 987,882%1185,615  % 1031,74 % 993,179
end
end
t11=unique([flipud(x_oppnormal(:,1)) flipud(y_oppnormal(:,1));t1(1,1) t1(1,2);x_normal(:,1) y_normal(:,1)],'rows','stable');
t12=unique([flipud(x_oppnormal(:,2)) flipud(y_oppnormal(:,2));t1(2,1) t1(2,2);x_normal(:,2) y_normal(:,2)],'rows','stable');
t13=unique([flipud(x_oppnormal(:,3)) flipud(y_oppnormal(:,3));t1(3,1) t1(3,2);x_normal(:,3) y_normal(:,3)],'rows','stable');
t14=unique([flipud(x_oppnormal(:,4)) flipud(y_oppnormal(:,4));t1(4,1) t1(4,2);x_normal(:,4) y_normal(:,4)],'rows','stable');
t15=unique([flipud(x_oppnormal(:,5)) flipud(y_oppnormal(:,5));t1(5,1) t1(5,2);x_normal(:,5) y_normal(:,5)],'rows','stable');


sub=im560-im561;
sub1=sub(:,:,1);figure,imagesc(sub1);
hold on;plot(t11(:,1),t11(:,2),'w*')
hold on;plot(t12(:,1),t12(:,2),'r*')
hold on;plot(t13(:,1),t13(:,2),'m*')
hold on;plot(t14(:,1),t14(:,2),'b*')
hold on;plot(t15(:,1),t15(:,2),'k*')
t11=round(t11);t12=round(t12);t13=round(t13);t14=round(t14);t15=round(t15);
p1=diag(sub1(t11(:,2),t11(:,1)));p2=diag(sub1(t12(:,2),t12(:,1)));p3=diag(sub1(t13(:,2),t13(:,1)));p4=diag(sub1(t14(:,2),t14(:,1)));p5=diag(sub1(t15(:,2),t15(:,1)));
figure,plot(1:length(t11),p1);figure,plot(1:length(t12),p2);figure,plot(1:length(t13),p3);figure,plot(1:length(t14),p4);figure,plot(1:length(t15),p5);

%% sgolay
ins=p6;
outs=sgolayfilt(ins,2,7);
figure,plot(1:length(outs),outs);
%% moving average
k=3;inp=p9;
B = 1/k*ones(k,1);
out = filter(B,1,inp);
figure,plot(1:length(out),out,'b*-');

%% With points making no mistake

% clc;clear;close all;
cd uu1_normalised_from_d_originals; im560=double(imread('cut_00560.png'));im561=double(imread('cut_00561.png'));cd ..;
[~,angim]=imgradient(im560(:,:,1));
[~,ang561]=imgradient(im561(:,:,1));
t1=[732 453;815 477;955 305;995 173;993 190];
t1(:,3)=diag(angim(t1(:,2),t1(:,1)));
t1(:,4)=t1(:,3)-90;
for t=1:length(t1)
for d=1:30
        [x_normal(d,t),y_normal(d,t)]=LocationAfterMoveAtAngledegree(t1(t,1),t1(t,2),t1(t,4),d);% 987,882%1185,615  % 1031,74 % 993,179
end
end
for t=1:length(t1)
for d=1:30
        [x_oppnormal(d,t),y_oppnormal(d,t)]=LocationAfterMoveAtAngledegree(t1(t,1),t1(t,2),t1(t,4)+180,d);% 987,882%1185,615  % 1031,74 % 993,179
end
end
t11=unique([flipud(x_oppnormal(:,1)) flipud(y_oppnormal(:,1));t1(1,1) t1(1,2);x_normal(:,1) y_normal(:,1)],'rows','stable');
t12=unique([flipud(x_oppnormal(:,2)) flipud(y_oppnormal(:,2));t1(2,1) t1(2,2);x_normal(:,2) y_normal(:,2)],'rows','stable');
t13=unique([flipud(x_oppnormal(:,3)) flipud(y_oppnormal(:,3));t1(3,1) t1(3,2);x_normal(:,3) y_normal(:,3)],'rows','stable');
t14=unique([flipud(x_oppnormal(:,4)) flipud(y_oppnormal(:,4));t1(4,1) t1(4,2);x_normal(:,4) y_normal(:,4)],'rows','stable');
t15=unique([flipud(x_oppnormal(:,5)) flipud(y_oppnormal(:,5));t1(5,1) t1(5,2);x_normal(:,5) y_normal(:,5)],'rows','stable');

sub=im560-im561;
sub1=sub(:,:,1);
figure,imagesc(sub1);
hold on;plot(t11(:,1),t11(:,2),'w*')
hold on;plot(t12(:,1),t12(:,2),'w*')
hold on;plot(t13(:,1),t13(:,2),'w*')
hold on;plot(t14(:,1),t14(:,2),'w*')
hold on;plot(t15(:,1),t15(:,2),'w*')
t11=round(t11);t12=round(t12);t13=round(t13);t14=round(t14);t15=round(t15);
p1=diag(sub1(t11(:,2),t11(:,1)));p2=diag(sub1(t12(:,2),t12(:,1)));p3=diag(sub1(t13(:,2),t13(:,1)));p4=diag(sub1(t14(:,2),t14(:,1)));p5=diag(sub1(t15(:,2),t15(:,1)));
figure,plot(1:length(t11),p1);figure,plot(1:length(t12),p2);figure,plot(1:length(t13),p3);figure,plot(1:length(t14),p4);figure,plot(1:length(t15),p5);

figure,imagesc(ang561);hold on;plot(xnew,ynew,'w*');hold on;plot(inputallx,inputally,'r*');title('561')

%% Making of uu3_coords - towards generalising the code for all points(for single section 560 to 561) - for testing the interslice redplane difference based method

clc;clear;close all;
cd uu1_normalised_from_d_originals; im560=double(imread('cut_00560.png'));im561=double(imread('cut_00561.png'));cd ..;
[~,ang560]=imgradient(im560(:,:,1));
[~,ang561]=imgradient(im561(:,:,1));
t1=importdata('inputall.mat');
t1(:,3)=diag(ang560(t1(:,2),t1(:,1)));
t1(:,4)=t1(:,3)-90;
for t=1:length(t1)
for d=1:15
        [x_normal(d,t),y_normal(d,t)]=LocationAfterMoveAtAngledegree(t1(t,1),t1(t,2),t1(t,4),d);% 987,882%1185,615  % 1031,74 % 993,179
end
end
for t=1:length(t1)
for d=1:15
        [x_oppnormal(d,t),y_oppnormal(d,t)]=LocationAfterMoveAtAngledegree(t1(t,1),t1(t,2),t1(t,4)+180,d);% 987,882%1185,615  % 1031,74 % 993,179
end
end

sub=im560-im561;
sub1=sub(:,:,1);
figure,imagesc(sub1);


% 

 t11=unique([flipud(x_oppnormal(:,1)) flipud(y_oppnormal(:,1));t1(1,1) t1(1,2);x_normal(:,1) y_normal(:,1)],'rows','stable');
t12=unique([flipud(x_oppnormal(:,200)) flipud(y_oppnormal(:,200));t1(200,1) t1(200,2);x_normal(:,200) y_normal(:,200)],'rows','stable');
t13=unique([flipud(x_oppnormal(:,300)) flipud(y_oppnormal(:,300));t1(300,1) t1(300,2);x_normal(:,300) y_normal(:,300)],'rows','stable');
t14=unique([flipud(x_oppnormal(:,400)) flipud(y_oppnormal(:,400));t1(400,1) t1(400,2);x_normal(:,400) y_normal(:,400)],'rows','stable');
t15=unique([flipud(x_oppnormal(:,521)) flipud(y_oppnormal(:,521));t1(521,1) t1(521,2);x_normal(:,521) y_normal(:,521)],'rows','stable');

t16=unique([flipud(x_oppnormal(:,600)) flipud(y_oppnormal(:,600));t1(600,1) t1(600,2);x_normal(:,600) y_normal(:,600)],'rows','stable');
t17=unique([flipud(x_oppnormal(:,602)) flipud(y_oppnormal(:,602));t1(602,1) t1(602,2);x_normal(:,602) y_normal(:,602)],'rows','stable');
t18=unique([flipud(x_oppnormal(:,603)) flipud(y_oppnormal(:,603));t1(603,1) t1(603,2);x_normal(:,603) y_normal(:,603)],'rows','stable');
t19=unique([flipud(x_oppnormal(:,604)) flipud(y_oppnormal(:,604));t1(604,1) t1(604,2);x_normal(:,604) y_normal(:,604)],'rows','stable');
% hold on;plot(t19(:,1),t19(:,2),'w*');
% t19=round(t19);
% % index=
% p9=diag(sub1(t19(:,2),t19(:,1)));figure,plot(1:length(t19),p9);
% 
% thresh=30;
t110=unique([flipud(x_oppnormal(:,605)) flipud(y_oppnormal(:,605));t1(605,1) t1(605,2);x_normal(:,605) y_normal(:,605)],'rows','stable');

t1a1=unique([flipud(x_oppnormal(:,800)) flipud(y_oppnormal(:,800));t1(800,1) t1(800,2);x_normal(:,800) y_normal(:,800)],'rows','stable');
t1a2=unique([flipud(x_oppnormal(:,802)) flipud(y_oppnormal(:,802));t1(802,1) t1(802,2);x_normal(:,802) y_normal(:,802)],'rows','stable');
t1a3=unique([flipud(x_oppnormal(:,880)) flipud(y_oppnormal(:,880));t1(880,1) t1(880,2);x_normal(:,880) y_normal(:,880)],'rows','stable');
t1a4=unique([flipud(x_oppnormal(:,885)) flipud(y_oppnormal(:,885));t1(885,1) t1(885,2);x_normal(:,885) y_normal(:,885)],'rows','stable');
t1a5=unique([flipud(x_oppnormal(:,920)) flipud(y_oppnormal(:,920));t1(920,1) t1(920,2);x_normal(:,920) y_normal(:,920)],'rows','stable');

% 
hold on;plot(t11(:,1),t11(:,2),'b*')
hold on;plot(t12(:,1),t12(:,2),'b*')
hold on;plot(t13(:,1),t13(:,2),'b*')
hold on;plot(t14(:,1),t14(:,2),'b*')
hold on;plot(t15(:,1),t15(:,2),'b*')
% 
hold on;plot(t16(:,1),t16(:,2),'m*')
hold on;plot(t17(:,1),t17(:,2),'m*')
hold on;plot(t18(:,1),t18(:,2),'m*')
hold on;plot(t19(:,1),t19(:,2),'m*');
hold on;plot(t110(:,1),t110(:,2),'m*')


hold on;plot(t1a1(:,1),t1a1(:,2),'r*')
hold on;plot(t1a2(:,1),t1a2(:,2),'r*')
hold on;plot(t1a3(:,1),t1a3(:,2),'k*')
hold on;plot(t1a4(:,1),t1a4(:,2),'k*')
hold on;plot(t1a5(:,1),t1a5(:,2),'k*')
% 
t11=round(t11);t12=round(t12);t13=round(t13);t14=round(t14);t15=round(t15);

t16=round(t16);t17=round(t17);t18=round(t18);t19=round(t19);t110=round(t110);

t1a1=round(t1a1);t1a2=round(t1a2);t1a3=round(t1a3);t1a4=round(t1a4);t1a5=round(t1a5);

p1=diag(sub1(t11(:,2),t11(:,1)));p2=diag(sub1(t12(:,2),t12(:,1)));p3=diag(sub1(t13(:,2),t13(:,1)));p4=diag(sub1(t14(:,2),t14(:,1)));p5=diag(sub1(t15(:,2),t15(:,1)));
k=3;sg1=5;sg2=13;
m1=movavg(p1,k);m2=movavg(p2,k);m3=movavg(p3,k);m4=movavg(p4,k);m5=movavg(p5,k);
% s1=sgolayfilt(p1,sg1,sg2);s2=sgolayfilt(p2,sg1,sg2);s3=sgolayfilt(p3,sg1,sg2);s4=sgolayfilt(p4,sg1,sg2);s5=sgolayfilt(p5,sg1,sg2);
figure,plot(1:length(t11),m1,'b*-',1:length(t12),m2,'b+-',1:length(t13),m3,'b^-',1:length(t14),m4,'bo-',1:length(t15),m5,'b.-');title('Individual plots of points -1 to 5');
% figure,plot(1:length(t11),s1,'b*-',1:length(t12),s2,'m*-',1:length(t13),s3,'g*-',1:length(t14),s4,'r*-',1:length(t15),s5,'k*-');
% figure,plot(1:length(t12),p2);figure,plot(1:length(t13),p3);figure,plot(1:length(t14),p4);figure,plot(1:length(t15),p5);

p6=diag(sub1(t16(:,2),t16(:,1)));p7=diag(sub1(t17(:,2),t17(:,1)));p8=diag(sub1(t18(:,2),t18(:,1)));p9=diag(sub1(t19(:,2),t19(:,1)));p10=diag(sub1(t110(:,2),t110(:,1)));
m6=movavg(p6,k);m7=movavg(p7,k);m8=movavg(p8,k);m9=movavg(p9,k);m10=movavg(p10,k);
figure,plot(1:length(t16),m6,'m*-',1:length(t17),m7,'m+-',1:length(t18),m8,'m^-',1:length(t19),m9,'mo-',1:length(t110),m10,'m.-');title('Individual plots of points of interest-6 to 10');
% figure,plot(1:length(t16),p6);figure,plot(1:length(t17),p7);figure,plot(1:length(t18),p8);figure,plot(1:length(t19),p9,'b+-');title('t19');figure,plot(1:length(t110),p10);

pa1=diag(sub1(t1a1(:,2),t1a1(:,1)));pa2=diag(sub1(t1a2(:,2),t1a2(:,1)));pa3=diag(sub1(t1a3(:,2),t1a3(:,1)));pa4=diag(sub1(t1a4(:,2),t1a4(:,1)));pa5=diag(sub1(t1a5(:,2),t1a5(:,1)));
ma1=movavg(pa1,k);ma2=movavg(pa2,k);ma3=movavg(pa3,k);ma4=movavg(pa4,k);ma5=movavg(pa5,k);
figure,plot(1:length(t1a1),ma1,'r*-',1:length(t1a2),ma2,'r*-');title('Individual plots of points -11 to 12');
figure,plot(1:length(t1a3),ma3,'k*-',1:length(t1a4),ma4,'k+-',1:length(t1a5),ma5,'ko-');title('Individual plots of points -13 to 15');
% figure,plot(1:length(t1a1),pa1);figure,plot(1:length(t1a2),pa2);figure,plot(1:length(t1a3),pa3);figure,plot(1:length(t1a4),pa4);figure,plot(1:length(t1a5),pa5);
m1to5=(m1+m2+m3+m4+m5)/5;
m6to10=(m6+m7+m8+m9+m10)/5;
m13to15=(ma3+ma4+ma5)/3;
m11to12=(ma1+ma2)/2;
figure,plot(1:length(t11),m1to5,'b*-',1:length(t16),m6to10,'m*-',1:length(t11),m13to15,'k*-',1:length(t11),m11to12,'r*-');title('Averaged plots');

figure,plot(1:length(t11),m5,'b*-',1:length(t16),m10,'m*-',1:length(t11),ma5,'k*-',1:length(t11),ma1,'r*-');title('Individual plots');

% figure,imagesc(ang561);hold on;plot(xnew,ynew,'w*');hold on;plot(inputallx,inputally,'r*');title('561')
%%
clc;clear;close all;
cd uu1_normalised_from_d_originals; im560=double(imread('cut_00560.png'));im561=double(imread('cut_00561.png'));cd ..;
[~,ang560]=imgradient(im560(:,:,1));
[~,ang561]=imgradient(im561(:,:,1));
t1=importdata('inputall.mat');
t1(:,3)=diag(ang560(t1(:,2),t1(:,1)));
t1(:,4)=t1(:,3)-90;
for t=1:length(t1)
for d=1:4
        [x_normal(d,t),y_normal(d,t)]=LocationAfterMoveAtAngledegree(t1(t,1),t1(t,2),t1(t,4),d);% 987,882%1185,615  % 1031,74 % 993,179
end
end
for t=1:length(t1)
for d=1:4
        [x_oppnormal(d,t),y_oppnormal(d,t)]=LocationAfterMoveAtAngledegree(t1(t,1),t1(t,2),t1(t,4)+180,d);% 987,882%1185,615  % 1031,74 % 993,179
end
end
x_normalcol=x_normal(:);y_normalcol=y_normal(:);
x_oppnormalcol=x_oppnormal(:);y_oppnormalcol=y_oppnormal(:);

sub=im560-im561;
sub1=sub(:,:,1);
figure,imagesc(sub1);hold on;plot(x_normalcol,y_normalcol,'w*',x_oppnormalcol,y_oppnormalcol,'r*');
x_normal=round(x_normal);y_normal=round(y_normal);x_oppnormal=round(x_oppnormal);y_oppnormal=round(y_oppnormal);
for i=1:length(t1)
max_normal=max(diag(sub1(y_normal(:,i),x_normal(:,i))));
max_oppnormal=max(diag(sub1(y_oppnormal(:,i),x_oppnormal(:,i))));
thresh=10;
if((max_normal<thresh)&&(max_oppnormal<thresh))
    t1(i,5)=2;
elseif(max_normal>max_oppnormal)
    t1(i,5)=1;
else
    t1(i,5)=0;
end
end
t1x=t1(:,1);t1y=t1(:,2);
inx=t1x(t1(:,5)==1);iny=t1y(t1(:,5)==1);
outx=t1x(t1(:,5)==0);outy=t1y(t1(:,5)==0);
onx=t1x(t1(:,5)==2);ony=t1y(t1(:,5)==2);
figure,imagesc(sub1);hold on;plot(inx,iny,'g*',outx,outy,'b*',onx,ony,'r*');
legend('Moving in Normal direction', 'Moving away from normal','Not Moving');

%% normal displacement

for i=1:length(inx)
xpt=inx(i);ypt=iny(i);
for d=1:20
        [test_normalx(d),test_normaly(d)]=LocationAfterMoveAtAngledegree(xpt,ypt,ang560(ypt,xpt)-90,d);% 987,882%1185,615  % 1031,74 % 993,179
end
test=[xpt ypt;test_normalx' test_normaly'];
test=unique(round(test),'rows','stable');
pts=diag(sub1(test(:,2),test(:,1)));
l=length(pts);
[~,trlc] = findpeaks(-pts,1:l);
d1=find(pts==max(pts),1,'first');
% dist(i)=trlc(find(trlc>find(pts==max(pts),1,'first'),1,'first'));
dist(i)=find(pts==max(pts),1,'first');
% if d1>15
%     dist(i)=5;
% else
%     dist(i)=trlc(find(trlc>find(pts==max(pts),1,'first'));
% end

end
dist=dist';

figure,plot(1:length(dist),dist);
coords=[inx,iny];


%% off normal displacement
for i=1:length(outx)
xpt=outx(i);ypt=outy(i);
for d=1:25
        [test_offnormalx(d),test_offnormaly(d)]=LocationAfterMoveAtAngledegree(xpt,ypt,ang560(ypt,xpt)+90,d);% 987,882%1185,615  % 1031,74 % 993,179
end
test=[xpt ypt;test_offnormalx' test_offnormaly'];
test=unique(round(test),'rows','stable');
pts=diag(sub1(test(:,2),test(:,1)));
l=length(pts);
[~,trlc] = findpeaks(-pts,1:l);
d1=find(pts==max(pts),1,'first');
% distoff(i)=trlc(find(trlc>find(pts==max(pts),1,'first'),1,'first'));
distoff(i)=find(pts==max(pts),1,'first');
% if d1>15
%     distoff(i)=5;
% else
%     distoff(i)=trlc(find(trlc>d1,1,'first'));
% end

end
distoff=distoff';
distoff(distoff>10)=3;

figure,plot(1:length(distoff),distoff);
coords=[outx,outy];

%% Normal

xpt=536;ypt=503;
for d=1:30
        [test_normalx(d),test_normaly(d)]=LocationAfterMoveAtAngledegree(xpt,ypt,ang560(ypt,xpt)-90,d);% 987,882%1185,615  % 1031,74 % 993,179
end
test=[xpt ypt;test_normalx' test_normaly'];
test=unique(round(test),'rows','stable');
pts=diag(sub1(test(:,2),test(:,1)));
l=length(pts);
figure,plot(1:l,pts);
[pk,lc] = findpeaks(pts,1:l);
hold on;plot(lc,pk,'k*');
findpeaks(pts,1:l,'Annotate','extents')
[tr,trlc] = findpeaks(-pts,1:l);
hold on;plot(trlc,-tr,'r*');title('for points moving in normal direction');

%% off normal
xpt=1023;ypt=847;
for d=1:30
        [test_offnormalx(d),test_offnormaly(d)]=LocationAfterMoveAtAngledegree(xpt,ypt,ang560(ypt,xpt)+90,d);% 987,882%1185,615  % 1031,74 % 993,179
end
test=[xpt ypt;test_offnormalx' test_offnormaly'];
test=unique(round(test),'rows','stable');
pts=diag(sub1(test(:,2),test(:,1)));
l=length(pts);
figure,plot(1:l,pts);
[pk,lc] = findpeaks(pts,1:l);
hold on;plot(lc,pk,'k*')
findpeaks(pts,1:l,'Annotate','extents');
[tr,trlc] = findpeaks(-pts,1:l);
hold on;plot(trlc,-tr,'r*');title('for points moving away from normal');

%% Using images from s folder : curvature as moving avg of angles

clc;clear;close all;
cd /hpc_atog/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/s_Binary_normthresh_C;i=imread('bin_00560.png');cd ..;
im1=i(1:1300,2051:3550);[ig,ang560]=imgradient(im1);
t1=importdata('inputall.mat');
p1=diag(ang560(t1(:,2),t1(:,1)));
m1=movavg(p1,51);plot(1:1284,m1,'b');figure,plot(1:1284,p1,'r');
figure,imagesc(im1);hold on;plot(t1(:,1),t1(:,2),'g*')


%%  Using images from s folder :  To do interslice gradient based operations
clc;clear;close all;
cd /hpc_atog/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/s_Binary_normthresh_C; im560=double(imread('bin_00560.png'));im561=double(imread('bin_00561.png'));cd ..;
im560=im560(1:1300,2051:3550);im561=im561(1:1300,2051:3550);
[~,ang560]=imgradient(im560(:,:));
[~,ang561]=imgradient(im561(:,:));
t1=importdata('inputall.mat');
t1(:,3)=diag(ang560(t1(:,2),t1(:,1)));
t1(:,4)=t1(:,3)-90;
for t=1:length(t1)
for d=1:6
        [x_normal(d,t),y_normal(d,t)]=LocationAfterMoveAtAngledegree(t1(t,1),t1(t,2),t1(t,4),d);% 987,882%1185,615  % 1031,74 % 993,179
end
end
for t=1:length(t1)
for d=1:6
        [x_oppnormal(d,t),y_oppnormal(d,t)]=LocationAfterMoveAtAngledegree(t1(t,1),t1(t,2),t1(t,4)+180,d);% 987,882%1185,615  % 1031,74 % 993,179
end
end
x_normalcol=x_normal(:);y_normalcol=y_normal(:);
x_oppnormalcol=x_oppnormal(:);y_oppnormalcol=y_oppnormal(:);

sub=im560-im561;
sub1=sub(:,:,1);
figure,imagesc(sub1);hold on;plot(x_normalcol,y_normalcol,'w*',x_oppnormalcol,y_oppnormalcol,'r*');
x_normal=round(x_normal);y_normal=round(y_normal);x_oppnormal=round(x_oppnormal);y_oppnormal=round(y_oppnormal);
for i=1:length(t1)
max_normal=max(diag(sub1(y_normal(:,i),x_normal(:,i))));
max_oppnormal=max(diag(sub1(y_oppnormal(:,i),x_oppnormal(:,i))));
thresh=10;
if((max_normal<thresh)&&(max_oppnormal<thresh))
    t1(i,5)=2;
elseif(max_normal>max_oppnormal)
    t1(i,5)=1;
else
    t1(i,5)=0;
end
end
t1x=t1(:,1);t1y=t1(:,2);
inx=t1x(t1(:,5)==1);iny=t1y(t1(:,5)==1);
outx=t1x(t1(:,5)==0);outy=t1y(t1(:,5)==0);
onx=t1x(t1(:,5)==2);ony=t1y(t1(:,5)==2);
figure,imagesc(sub1);hold on;plot(inx,iny,'g*',outx,outy,'b*',onx,ony,'r*');
legend('Moving in Normal direction', 'Moving away from normal','Not Moving');
