%%  Instructions to run the program

% Run First 2 sections for L, a*, b* values(Validate normalisation in LAB space)
% Run Last 2 sections for R,G,B values(Validate normalisation results after converting back to RGB space)


%% LAB-Before Normalisation
clc;clear;close all;
cd /hpc/btho733/ABI/pacedSheep01/medsci_poster/stain_normalisation_toolbox/pacedsheepimages/v8/images/;
for i=1:30
imname=sprintf('v8%04d.tif',i-1);
im=imread(imname);
cform = makecform('srgb2lab');
lab_he = applycform(im,cform);
[p,q,dim]=size(im);
IND=1:p*q;
s=[p,q];
[r,c] = ind2sub(s,IND); %Creates the row and column vectors for co-ordinate
p1 = impixel(im,c,r);    % p gives RGB values for all pixels
p2=impixel(lab_he,c,r);
m(i,1)=mean(p2(:,1));sd(i,1)=std(p2(:,1));
m(i,2)=mean(p2(:,2));sd(i,2)=std(p2(:,2));
m(i,3)=mean(p2(:,3));sd(i,3)=std(p2(:,3));
end
figure(1),plot(1:30,m(:,1),'*-r',1:30,m(:,2),'*-g',1:30,m(:,3),'*-b');
figure(2),plot(1:30,sd(:,1),'*-r',1:30,sd(:,2),'*-g',1:30,sd(:,3),'*-b');
legend('L before','a* before','b* before','L after','a* after','b* after');xlabel('Slice');ylabel('Mean');
%% LAB-After normalisation
clc;clear;

cd /hpc/btho733/ABI/pacedSheep01/medsci_poster/stain_normalisation_toolbox/pacedsheepimages/v8/output_RH/Nothreshold/
for i=1:30
imname=sprintf('cut_%05d.tif',i-1);
im=imread(imname);
cform = makecform('srgb2lab');
lab_he = applycform(im,cform);
[p,q,dim]=size(im);
IND=1:p*q;
s=[p,q];
[r,c] = ind2sub(s,IND); %Creates the row and column vectors for co-ordinate
p1 = impixel(im,c,r);    % p gives RGB values for all pixels
p2=impixel(lab_he,c,r);
m(i,1)=mean(p2(:,1));sd(i,1)=std(p2(:,1));
m(i,2)=mean(p2(:,2));sd(i,2)=std(p2(:,2));
m(i,3)=mean(p2(:,3));sd(i,3)=std(p2(:,3));
end

figure(1),hold on;plot(1:30,m(:,1),'r',1:30,m(:,2),'g',1:30,m(:,3),'b');
figure(2),hold on;plot(1:30,sd(:,1),'r',1:30,sd(:,2),'g',1:30,sd(:,3),'b');
legend('L before','a* before','b* before','L after','a* after','b* after');xlabel('Slice');ylabel('Standard Deviation');
% hold on,plot(1:30,m(:,1));
% axis([0 30 10 180])

%% RGB values : Before Normalisation
clc;clear;close all;
cd /hpc/btho733/ABI/pacedSheep01/medsci_poster/stain_normalisation_toolbox/pacedsheepimages/v8/images;
for i=1:30
imname=sprintf('v8%04d.tif',i-1);
im=imread(imname);
cform = makecform('srgb2lab');
lab_he = applycform(im,cform);
[p,q,dim]=size(im);
IND=1:p*q;
s=[p,q];
[r,c] = ind2sub(s,IND); %Creates the row and column vectors for co-ordinate
p1 = impixel(im,c,r);    % p gives RGB values for all pixels
p2=impixel(lab_he,c,r);
m(i,1)=mean(p1(:,1));sd(i,1)=std(p1(:,1));
m(i,2)=mean(p1(:,2));sd(i,2)=std(p1(:,2));
m(i,3)=mean(p1(:,3));sd(i,3)=std(p1(:,3));
end
figure(1),plot(1:30,m(:,1),'*-r',1:30,m(:,2),'*-g',1:30,m(:,3),'*-b');
figure(2),plot(1:30,sd(:,1),'*-r',1:30,sd(:,2),'*-g',1:30,sd(:,3),'*-b');

%% RGB values :After normalisation
clc;clear;

cd /hpc/btho733/ABI/pacedSheep01/medsci_poster/stain_normalisation_toolbox/pacedsheepimages/v8/output_RH/Nothreshold/
for i=1:30
imname=sprintf('cut_%05d.tif',i-1);
im=imread(imname);
cform = makecform('srgb2lab');
lab_he = applycform(im,cform);
[p,q,dim]=size(im);
IND=1:p*q;
s=[p,q];
[r,c] = ind2sub(s,IND); %Creates the row and column vectors for co-ordinate
p1 = impixel(im,c,r);    % p gives RGB values for all pixels
p2=impixel(lab_he,c,r);
m(i,1)=mean(p1(:,1));sd(i,1)=std(p1(:,1));
m(i,2)=mean(p1(:,2));sd(i,2)=std(p1(:,2));
m(i,3)=mean(p1(:,3));sd(i,3)=std(p1(:,3));
end

figure(1),hold on;plot(1:30,m(:,1),'r',1:30,m(:,2),'g',1:30,m(:,3),'b');
figure(2),hold on;plot(1:30,sd(:,1),'r',1:30,sd(:,2),'g',1:30,sd(:,3),'b');
% hold on,plot(1:30,m(:,1));
% axis([0 30 10 180])