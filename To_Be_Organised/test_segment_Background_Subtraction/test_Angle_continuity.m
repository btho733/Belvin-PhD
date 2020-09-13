clc;clear; close all;
inputall=importdata('inputall.mat');
inputall=inputall(1:280,:);
for imno=560:660
targetno=imno+1;
imname=sprintf('s4_%05d.png',imno);cd d_images2dcoarsesegment_s4; im=imread(imname); cd ..;
targetname=sprintf('s4_%05d.png',targetno);cd d_images2dcoarsesegment_s4; target=imread(targetname); cd ..;
sobname=sprintf('sobel_%05d.png',targetno);sobimname=sprintf('sobel_%05d.png',imno);cd ua_newalgo_sobel;sobtarget=imread(sobname);sobim=imread(sobimname);cd ..;
target1=target(1:1300,2051:3550,1);%target2=target(1:1300,2051:3550,2);target3=target(1:1300,2051:3550,3);
[targr1,tarang1]=imgradient(target1);%[~,tarang2]=imgradient(target2);[~,tarang3]=imgradient(target3);
sob1=sobtarget(:,:,1);%sob2=sobtarget(:,:,2);sob3=sobtarget(:,:,3);
im1=im(1:1300,2051:3550,1);[imgr1,imang1]=imgradient(im1);inputall(:,3)=diag(imang1(inputall(:,2),inputall(:,1)));
for i=1:length(inputall)
    inputall(i,4)=func_findavgangle(inputall(i,1),inputall(i,2),inputall(i,3),imang1,1,5);
end
inputall(:,5)=abs(inputall(:,3)-inputall(:,4));
for i=1:length(inputall)
if(inputall(i,5)<70)
[ xnew(i),ynew(i) ] = func_BestAngleBasedNeighbor(inputall(i,1),inputall(i,2),inputall(i,4),tarang1,5,5);
else
    xnew(i)=1;ynew(i)=1;
end
end
% figure,imagesc(sobtarget);colormap(jet);hold on;plot(xnew,ynew,'w*',inputall(:,1),inputall(:,2),'r*');

% figure,plot(1:length(inputall),inputall(:,5))
% figure,plot(1:length(inputall),sgolayfilt(inputall(:,5),3,17))
finaltest=diag(sob1(ynew,xnew));
xnew(finaltest<20)=[];ynew(finaltest<20)=[];
h=figure;imagesc(sobtarget);title(['Image no   ' ,num2str(targetno)]);
% hold on;plot(coords(:,1),coords(:,2),'w*');
newpoints=[xnew' ynew'];
coords=unique(round(newpoints),'rows','stable');
coords1=sgolayfilt(coords(1:end,:),2,7);
hold on;xy=coords';spcv=cscvn(xy);points=fnplt(spcv,'w',1);
hold on;plot(points(1,:),points(2,:),'w','LineWidth',1.6);
cd ur_figures/;saveas(h,sprintf('FIG_1_%d.tif',targetno));cd ..;
% 
clear inputall;
% coords=unique(round(coords1),'rows','stable');

cd ur0_coords_inputall_trial4_mat;c_name=sprintf('c_%05d.mat',targetno);save(c_name,'coords');cd ..;
inputall=unique(round(points'),'rows','stable');
clear xnew; clear ynew;
close all;
end

% points1=points';figure,imagesc(tarang1);colormap(jet);hold on;plot(inputall(:,1),inputall(:,2),'w*',xnew,ynew,'r*',coords1(:,1),coords(:,2),'b*',points1(:,1),points1(:,2),'m*');title(['Angle-red- Image no   ' ,num2str(targetno)]);
%%
% 
% angtarget(:,:,1)=tarang1;angtarget(:,:,2)=tarang2;angtarget(:,:,3)=tarang3;
% targetangu=uint8(angtarget);
% figure,imagesc(targetangu);hold on;plot(inputall(:,1),inputall(:,2),'w*');title(['Angle(Unsigned)-all- Image no   ' ,num2str(targetno)]);
% figure,imagesc(imang1);colormap(jet);hold on;plot(inputall(:,1),inputall(:,2),'w*');title(['Angle-red- Image no   ' ,num2str(imno)]);
% figure,imagesc(tarang1);colormap(jet);hold on;plot(inputall(:,1),inputall(:,2),'w*');title(['Angle-red- Image no   ' ,num2str(targetno)]);
% figure,imagesc(sobim);colormap(jet);hold on;plot(inputall(:,1),inputall(:,2),'w*');title(['Sobel- Image no   ' ,num2str(imno)]);