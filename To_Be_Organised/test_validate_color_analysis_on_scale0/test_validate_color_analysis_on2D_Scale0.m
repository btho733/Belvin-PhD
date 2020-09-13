% Method 1  (slower)  :  Run section 1 followed by section2
% Method 2  (Faster)  :  Run section 2 only (After uncommenting its first line - combinep loading)

%% Mark a polygon on a-b map and remap the selected region into image -EFFICIENT version
clear;clc;close all;
im=imread('Image_00580_manualcleared_colorbalanced.png');% red=0;green=255;blue=0;
% RAW values
%-----------
% Convert to L-a-b and Get combinep table with raw values of R,G,B,a,b in
% vector form ( Note: ind2sub is used to do this formation in an efficient way)
cform = makecform('srgb2lab');
lab_he = applycform(im,cform);
[p,q,dim]=size(im);
IND=1:p*q;
s=[p,q];
[r,c] = ind2sub(s,IND); %Creates the row and column vectors for co-ordinate
p1 = impixel(im,c,r);    % p1 gives RGB values for all pixels
p2=impixel(lab_he,c,r);   % p2 gives LAB values for all pixels
ab=p2(:,2:3);combinep=[p1 ab];
x=p1(:,1);y=p1(:,2);z=p1(:,3);a=ab(:,1);b=ab(:,2);
s=15*ones(numel(a),1);
co1= x/255;
co2= y/255;
co3= z/255;
co=[co1,co2,co3];
figure,scatter(a,b,s,co,'fill'); title('a-b of L-a-b');
%%
 combinep=importdata('combinep_00580_fat4_try1.mat');ab=combinep(:,4:5);
p=9960;q=16560;
uab=importdata('uab_00580_0scale.mat');


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
rplane=combinep(:,1);rplane(Lia==0)=0;
gplane=combinep(:,2);gplane(Lia==0)=0;
bplane=combinep(:,3);bplane(Lia==0)=0;
finalplane(:,:,1)=reshape(rplane,[p,q]);
finalplane(:,:,2)=reshape(gplane,[p,q]);
finalplane(:,:,3)=reshape(bplane,[p,q]);
finalplane=uint8(finalplane);
figure,imagesc(finalplane);
% finalplane1=finalplane(:,:,1);finalplane2=finalplane(:,:,2);finalplane3=finalplane(:,:,3);
% fatplane=finalplane1;
% fatplane((finalplane1>0)|(finalplane2>0)|finalplane3>0)=100;
% fatplane
%% Iterative application of the approach taking the finalplane from section 2 each time.
clear uab;clear combinep;clear ab;clear im;clear lab_he;
im=finalplane;
% RAW values
%-----------
% Convert to L-a-b and Get combinep table with raw values of R,G,B,a,b in
% vector form ( Note: ind2sub is used to do this formation in an efficient way)
cform = makecform('srgb2lab');
lab_he = applycform(im,cform);
[p,q,dim]=size(im);
IND=1:p*q;
s=[p,q];
[r,c] = ind2sub(s,IND); %Creates the row and column vectors for co-ordinate
p1 = impixel(im,c,r);    % p1 gives RGB values for all pixels
p2=impixel(lab_he,c,r);   % p2 gives LAB values for all pixels
ab=p2(:,2:3);combinep=[p1 ab];

%Unique values 
%(to make the scatter plot look good so that rgb colours are precisely shown.)
% [NOTE : Unique values step is slightly inefficient and slower. To
% workaround that, save uab table into a mat file  and run the code in
% section2 above after loading the saved mat file (FASTER OPTION described above(Method 2) is just doing this).

% Make uab table with unique values of a-b combination
uab=unique(ab,'rows','stable');
% To ensure that only the unique a-b combinations go into the scatter plot
for n=1:length(uab)
    cond1=(combinep(:,4)>(uab(n,1)-1))&(combinep(:,4)<(uab(n,1)+1));
    cond2=(combinep(:,5)>(uab(n,2)-1))&(combinep(:,5)<(uab(n,2)+1));
    fil_combinep=combinep(cond1&cond2,:);
    uab(n,3)=mean(fil_combinep(:,1));
    uab(n,4)=mean(fil_combinep(:,2));
    uab(n,5)=mean(fil_combinep(:,3));
    disp(length(uab)-n);
    
end

% save('uab_00580_fat4_try1.mat','uab');
% save('combinep_00580_fat4_try1.mat','combinep','-v7.3');


% save('uab_00516.mat','uab'); %