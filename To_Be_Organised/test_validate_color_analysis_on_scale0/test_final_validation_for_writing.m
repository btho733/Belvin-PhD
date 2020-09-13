clc;clear;close all;
cd D:\hpc_ABI_20Mar20\ABI\pacedSheep01\test_validate_color_analysis_on_scale0\;
uab=importdata('uab_00580_0scale.mat');
%scatter plot of a-b(Unique values only)
x=uab(:,3);y=uab(:,4);z=uab(:,5);a=uab(:,1);b=uab(:,2);
s=15*ones(numel(a),1);
co1= x/255;
co2= y/255;
co3= z/255;
co=[co1,co2,co3];
figure,scatter(a,b,s,co,'fill'); title('a-b of L-a-b');
%% Mark a polygonal Section
 [xpts,ypts]=getpts;
 xy=[xpts,ypts];
 %Mask1: Get the mask corr. to polygonal section
 mask1 = poly2mask(xpts,ypts,240,240); 
 % figure,imagesc(mask1);set(gca,'Ydir','normal');
[bsel,asel]=find(mask1);
%  save('yellow.mat','xy');
avg_bsel=mean(bsel);avg_asel=mean(asel);

%%
clear;close all;clc;
uab=importdata('uab_00580_0scale.mat');
r=uab(:,3);g=uab(:,4);b=uab(:,5);x=uab(:,1);y=uab(:,2);
xmax=max(x);xmin=min(x);
ymax=max(y);ymin=min(y);
im=255.*ones(240,240,3);
for i=1:length(r)
im(y(i),x(i),1)=r(i);
im(y(i),x(i),2)=g(i);
im(y(i),x(i),3)=b(i);
disp(i);
end
%%
clear;clc;close all;
im=imread('Image_00580_manualcleared_colorbalanced.png');
im1=im(:,:,1);rplane=im1(:);
im2=im(:,:,2);gplane=im2(:);
im3=im(:,:,3);bplane=im3(:);
cform = makecform('srgb2lab');
lab_he = applycform(im,cform);
[p,q,dim]=size(im);
lab_a=lab_he(:,:,2);
lab_b=lab_he(:,:,3);
ab=[lab_a(:) lab_b(:)];
uab=importdata('uab_00580.mat');


%scatter plot of a-b(Unique values only)
x=uab(:,3);y=uab(:,4);z=uab(:,5);a=uab(:,1);b=uab(:,2);
s=15*ones(numel(a),1);
co1= x/255;
co2= y/255;
co3= z/255;
co=[co1,co2,co3];
figure,scatter(a,b,s,co,'fill'); title('a-b of L-a-b');
% Make a polygonal selection from scatter plot and Get the a-b values
% contained within that selection

% Mark a polygonal Section
 [xpts,ypts]=getpts;
  %xy=[xpts,ypts];save('marker_00580_0scale_purple1.mat','xy');
% xy=importdata('marker_00580_0scale_purple1.mat');
% xpts=xy(:,1);ypts=xy(:,2);
 %Mask1: Get the mask corr. to polygonal section
 mask1 = poly2mask(xpts,ypts,240,240); 
 %figure,imagesc(mask1);axis xy;title('Selected polygonal mask')
 %Mask2 : Get the binary equivalent of a-b scatter plot
 mask2=zeros(240,240);
 for i=1:length(a)
 mask2(b(i),a(i))=1;
 end
 %figure,imagesc(mask2);axis xy;title('binary equivalent of a-b scatter plot');
% filter out the intersection of Mask1 and Mask2
filt=zeros(240,240);
filt((mask2==1) & (mask1==1))=1;figure,imagesc(filt);axis xy;title('binary a-b plot filtered as per the selected polygonal mask criteria');
ind=find(filt==1);s=[240,240];[filtb,filta]=ind2sub(s,ind);
filtab=[filta,filtb];
Lia=ismember(ab,filtab,'rows');
rplane(Lia==0)=255;
gplane(Lia==0)=255;
bplane(Lia==0)=255;
finalplane(:,:,1)=reshape(rplane,[p,q]);
finalplane(:,:,2)=reshape(gplane,[p,q]);
finalplane(:,:,3)=reshape(bplane,[p,q]);
finalplane=uint8(finalplane);
figure,imagesc(finalplane);