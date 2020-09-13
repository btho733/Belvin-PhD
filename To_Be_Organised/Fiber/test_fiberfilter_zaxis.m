%% For converting angle from -180 to 180  to 0 to 180

[p,q]=size(an);
for i= 1:p
    for j=1:q
        
        theta_element=an(i,j);
        
        if theta_element <0
            temp = 180+theta_element;
        end
        if theta_element >0
            temp = theta_element;
        end
        
        theta1(i,j)=temp;
    end
end
figure,imagesc(theta1);colormap(jet)
%%
clear;close all;clc;
% cd /hpc/btho733/ABI/pacedSheep01/medsci_poster/stain_normalisation_toolbox/pacedsheepimages/f2v1/synth_op2/;  %change this for new image
% cd V:\ABI\pacedSheep01\medsci_poster\stain_normalisation_toolbox\pacedsheepimages\v8\output_RH\Nothreshold; 
% cd V:\ABI\JZ\Fiber_2016\Work_2017\CentralSection50um3\cut_for_elevation\bgdCorrect;
% cd /hpc_atog/btho733/ABI/JZ/Fiber_2016/Work_2017/CentralSection50um3/cut_for_elevation/bgdCorrect;
% cd /hpc_atog/btho733/ABI/JZ/Fiber_2016/Work_2017/CentralSection50um3/raw_25um3/bgdCorrect;
% cd /hpc_atog/btho733/ABI/JZ/Fiber_2016/Work_2017/CentralSection50um3/raw_50um3/bgdBlack;
% cd V:\ABI\JZ\Fiber_DTI\filtered_images;
cd V:\ABI\JZ\Fiber_2016\Work_2017\CentralSection50um3\cut_for_elevation\;
len=100;  %change this for new image
for i=1:len
infile=sprintf('cut_%04d.tif',i);%outfile=sprintf('cut_%05d.tif',i-500);                       %change this for new image
im=imread(infile);
% im=im(y1:y2,x1:x2,:);
p1=im(:,:,1);
p2=im(:,:,2);
p3=im(:,:,3);

D1(:,:,i,1)=p1;
D1(:,:,i,2)=p2;
D1(:,:,i,3)=p3;
end

[Ny, Nx, Nz,k] = size(D1);
clear p1;clear p2;clear p3;
%% %% For Y-plane (XZ)
section=100;
clc;close all;
redplane=squeeze(D1(section,:,:,1));
figure,imagesc(redplane);colormap(jet);
im=squeeze(D1(section,:,:,:));
h=figure;
imagesc(im);
%% For X-plane (YZ)
section=80;
clc;close all;
redplane=squeeze(D1(:,section,:,1));
figure,imagesc(redplane);colormap(jet);
im=squeeze(D1(:,section,:,:));
h=figure;
imagesc(im);
%% click only one point at a time
for n=1:100
[col,row]=getpts(h);
col=round(col);row=round(row);
y(n,:)=im(:,col);
figure,plot(1:150,y(n,:),'b-*');ylim([0 255]);
end
%%
clc;
figure;
for t=1:n-1
    plot(1:length(y),y(t,:));ylim([0 255]);hold on;

end

%%
 y1=y(1:3,:);
 r=5;
 m1=movmean(y1(1,:),r,2);
 m2=movmean(y1(2,:),r,2);
 m3=movmean(y1(3,:),r,2);
 figure,plot(1:length(y),m1,1:length(y),m2);legend('column 16','column 17');
figure,plot(1:length(y),m2,1:length(y),m3);legend('column 17','column 18');

%%
diff=m2-m3;
corrected21=double(y1(3,:))+diff;
figure,plot(1:length(y),y1(2,:),1:length(y),corrected21);legend('column 17','corrected 18');
% figure,plot(1:length(y),diff);legend('diff');
figure,plot(1:length(y),y1(2,:),1:length(y),y1(3,:));legend('column 17','column 18');

%%
image1=[y1(2,:);y1(3,:)];
figure,imagesc(image1');colormap(jet);
image2=[y1(2,:);corrected21];
figure,imagesc(image2');colormap(jet);

%% For Y-plane (XZ)
corrected_2=255*ones(180,100);
upto=99;  r=5;
for column=1:upto
    if(column==1)
    im1(:,1)=squeeze(im(:,column,1));
    im1(:,2)=squeeze(im(:,column+1,1));
    else
    im1(:,1)=corrected_2(:,column);
    im1(:,2)=squeeze(im(:,column+1,1));  
    end
  
    m1=movmean(im1(:,1),r,1);
    m2=movmean(im1(:,2),r,1);
    diff=m1-m2;
    corrected_2(:,column+1)=double(im1(:,2))+diff;
end
% figure,imagesc(uint8(corrected_2));colormap(jet);
figure,imagesc(im);colormap(jet);
[gr_red,ang_red]=imgradient(redplane);figure,imagesc(gr_red);colormap(jet);title('red plane gradient');
[gr,ang]=imgradient(corrected_2);figure,imagesc(gr);colormap(jet);title('gradient after correction');
%% %% For X-plane (YZ)
corrected_2=255*ones(150,100);
upto=99;r=10;
for column=1:upto
    if(column==1)
    im1(:,1)=squeeze(im(:,column,1));
    im1(:,2)=squeeze(im(:,column+1,1));
    else
    im1(:,1)=corrected_2(:,column);
    im1(:,2)=squeeze(im(:,column+1,1));  
    end
    
    m1=movmean(im1(:,1),r,1);
    m2=movmean(im1(:,2),r,1);
    diff=m1-m2;
    corrected_2(:,column+1)=double(im1(:,2))+diff;
end
% figure,imagesc(uint8(corrected_2));colormap(jet);
figure,imagesc(corrected_2);colormap(jet);
[gr_red,ang_red]=imgradient(redplane);figure,imagesc(gr_red);colormap(jet);title('red plane gradient');
[gr,ang]=imgradient(corrected_2);figure,imagesc(gr);colormap(jet);title('gradient after correction');
%% Gaussian
img=imgaussian(JI,1,3);
figure,imagesc(img);colormap(jet);
figure,imagesc(squeeze(im(:,:,1)));colormap(jet);title('Red plane');
%% Anisotropic diffusion
cd V:\ABI\pacedSheep01\Anisotropic;
JI = CoherenceFilter(corrected_2,struct('T',1,'sigma',0.15,'lambda_c',0.95,'lambda_h',0.9,'alpha',0.00001,'rho',0.1,'Scheme','I','eigenmode',3));
figure,imagesc(JI);colormap(jet);

%% All 3D Y-planes
yplane=255*ones(1,Nx,Nz);
for section=1:Ny
    im=squeeze(D1(section,:,:,:));
    corrected_yplane=255*ones(Nx,Nz);
    upto=99;r=12;
    for column=1:upto
        if(column==1)
            im1(:,1)=squeeze(im(:,column,1));
            im1(:,2)=squeeze(im(:,column+1,1));
        else
            im1(:,1)=corrected_yplane(:,column);
            im1(:,2)=squeeze(im(:,column+1,1));
        end
        
        m1=movmean(im1(:,1),r,1);
        m2=movmean(im1(:,2),r,1);
        diff=m1-m2;
        corrected_yplane(:,column+1)=double(im1(:,2))+diff;
    end
    yplane(section,:,:)=corrected_yplane;
%     gr_yplane(:,:,section)=imgradient(corrected_yplane);
end
cd V:\ABI\pacedSheep01\Anisotropic\functions;
showcs3(single(yplane));
% figure,showcs3(single(squeeze(D1(:,:,:,1))));

%% All 3D X-planes
xplane=255*ones(Ny,1,Nz);
for section=1:Nx
    im=squeeze(D1(:,section,:,:));
    corrected_xplane=255*ones(Ny,Nz);
    upto=99;r=12;
    for column=1:upto
        if(column==1)
            im1(:,1)=squeeze(im(:,column,1));
            im1(:,2)=squeeze(im(:,column+1,1));
        else
            im1(:,1)=corrected_xplane(:,column);
            im1(:,2)=squeeze(im(:,column+1,1));
        end
        
        m1=movmean(im1(:,1),r,1);
        m2=movmean(im1(:,2),r,1);
        diff=m1-m2;
        corrected_xplane(:,column+1)=double(im1(:,2))+diff;
    end
    xplane(:,section,:)=corrected_xplane;
%     gr_xplane(:,:,section)=imgradient(corrected_xplane);
end
cd V:\ABI\pacedSheep01\Anisotropic\functions;
showcs3(single(xplane));
% showcs3(single(squeeze(D1(:,:,:,1))));
%% Averaging and Comparing results from corrections in yplane and xplane
clc;clear;close all;
cd V:\ABI\JZ\Fiber_DTI;


% px=xplane_corrected(:,:,50);
% py=yplane_corrected(:,:,50);
% figure,imagesc(px);colormap(jet);
% figure,imagesc(py);colormap(jet);
avg=(xplane_corrected+yplane_corrected)./2;
cd V:\ABI\pacedSheep01\Anisotropic\functions;
showcs3(single(avg));
%% Load Averaged xplane_corrected and yplane_corrected
xplane_corrected=importdata('xplane_26Oct17.mat');
avg=importdata('averaged_26Oct17.mat');
yplane_corrected=importdata('yplane_26Oct17.mat');

%% For Y-plane (XZ)
corrected_2=255*ones(180,100);
upto=99; 
figure;
% figure;title('Corrected planes at different window sizes');
for num=1:8
    r=num*2;
for column=1:upto
    if(column==1)
    im1(:,1)=squeeze(im(:,column,1));
    im1(:,2)=squeeze(im(:,column+1,1));
    else
    im1(:,1)=corrected_2(:,column);
    im1(:,2)=squeeze(im(:,column+1,1));  
    end
  
    m1=movmean(im1(:,1),r,1);
    m2=movmean(im1(:,2),r,1);
    diff=m1-m2;
    corrected_2(:,column+1)=double(im1(:,2))+diff;
end
corrected_r(:,:,num)=corrected_2;
rnum=sprintf('r = %d',r);
subplot(4,2,num);imagesc(squeeze(corrected_r(:,:,num)));colormap(gray);title(rnum);
end

% subplot(2,1,1);imagesc(squeeze(corrected_r(:,:,1)));colormap(jet);title('r=
% subplot(2,1,2);imagesc(squeeze(corrected_r(:,:,2)));colormap(jet);
%% Loading <<<<   Greyscale (1-plane) version >>>>>
clear;close all;clc;
% cd /hpc/btho733/ABI/pacedSheep01/medsci_poster/stain_normalisation_toolbox/pacedsheepimages/f2v1/synth_op2/;  %change this for new image
% cd V:\ABI\pacedSheep01\medsci_poster\stain_normalisation_toolbox\pacedsheepimages\v8\output_RH\Nothreshold; 
% cd V:\ABI\JZ\Fiber_2016\Work_2017\CentralSection50um3\cut_for_elevation\bgdCorrect;
cd V:\ABI\JZ\Fiber_DTI\filtered_images_255\contrast_enhanced;
len=98;  %change this for new image
for i=1:len
infile=sprintf('cut_%04d.tif',i);%outfile=sprintf('cut_%05d.tif',i-500);                       %change this for new image
im=imread(infile);
% im=im(y1:y2,x1:x2,:);
% p1=im(:,:,1);
% p2=im(:,:,2);
% p3=im(:,:,3);

Dgrey(:,:,i)=im;
% D1(:,:,i,2)=p2;
% D1(:,:,i,3)=p3;
end

[Ny, Nx, Nz] = size(D1);


%% All 3D Y-planes and X-planes : <<<<    Wrong one   >>>>>>>
yplane=255*ones(1,Nx,Nz);xplane=255*ones(Ny,1,Nz);
  for sectiony=1:Ny
      for sectionx=1:Nx
        imy=squeeze(D1(100,:,:,:));imx=squeeze(D1(:,sectionx,:,:));
        corrected_yplane=255*ones(Nx,Nz);corrected_xplane=255*ones(Ny,Nz);
        corrected_zplane=255*ones(Ny,Nx);
        upto=99;rx=10;ry=10;
    for zplane=1:upto
        
        if(zplane==1)
            imx1(:,1)=squeeze(imx(:,zplane,1));
            imx1(:,2)=squeeze(imx(:,zplane+1,1));
            imy1(:,1)=squeeze(imy(:,zplane,1));
            imy1(:,2)=squeeze(imy(:,zplane+1,1));
        else
            imx1(:,1)=corrected_xplane(:,zplane);
            imx1(:,2)=squeeze(imx(:,zplane+1,1));
            imy1(:,1)=corrected_yplane(:,zplane);
            imy1(:,2)=squeeze(im(:,zplane+1,1));
        end
        
        mx1=movmean(imx1(:,1),rx,1);
        mx2=movmean(imx1(:,2),rx,1);
        my1=movmean(imy1(:,1),ry,1);
        my2=movmean(imy1(:,2),ry,1);
        avg1=(mx1+my1)/2;avg2=(mx2+my2)/2;
        diff=avg1-avg2;
        corrected_xplane(:,zplane+1)=double(imx1(:,2))+diff;
        corrected_yplane(:,zplane+1)=double(imy1(:,2))+diff;
    end
    xplane(:,sectionx,:)=corrected_xplane;
%     gr_xplane(:,:,section)=imgradient(corrected_xplane);
      end
  end
cd V:\ABI\pacedSheep01\Anisotropic\functions;
showcs3(single(xplane));

%% Filtering All 3D Y-planes and X-planes : <<<<   RGB Correct version  >>>>>>>
corrected_block=squeeze(D1(:,:,:,1));
corrected_zplane=255*ones(Ny,Nx);upto=Nz-1;
for zplane=1:upto
    rx=10;ry=10;
    if(zplane==1)       
        im1=squeeze(D1(:,:,zplane,1));
        im2=squeeze(D1(:,:,zplane+1,1));
    else
        im1=corrected_zplane;
        im2=squeeze(D1(:,:,zplane+1,1));
    end
    my1=movmean(im1,ry,2);
    my2=movmean(im2,ry,2);
    mx1=movmean(im1,rx,1);%mx1=mx_1';
    mx2=movmean(im2,rx,1);%mx2=mx_2';
    avg1=(mx1+my1)/2;avg2=(mx2+my2)/2;
    diff=avg1-avg2;
    corrected_zplane=double(im2)+diff;
    corrected_block(:,:,zplane+1)=corrected_zplane;
end

%% Filtering All 3D Y-planes and X-planes : <<<<   Greyscale Correct version  >>>>>>>
corrected_block=squeeze(D1(:,:,1));
corrected_zplane=255*ones(Ny,Nx);upto=97;
for zplane=1:upto
    rx=10;ry=10;
    if(zplane==1)       
        im1=squeeze(D1(:,:,zplane));
        im2=squeeze(D1(:,:,zplane+1));
    else
        im1=corrected_zplane;
        im2=squeeze(D1(:,:,zplane+1));
    end
    my1=movmean(im1,ry,2);
    my2=movmean(im2,ry,2);
    mx1=movmean(im1,rx,1);%mx1=mx_1';
    mx2=movmean(im2,rx,1);%mx2=mx_2';
    avg1=(mx1+my1)/2;avg2=(mx2+my2)/2;
    diff=avg1-avg2;
    corrected_zplane=double(im2)+diff;
    corrected_block(:,:,zplane+1)=corrected_zplane;
end
%% To produce the high contrast template image (From filtering All 3D Y-planes and X-planes : <<<<  RGB Correct version  >>>>>>>)
corrected_block=squeeze(D1(:,:,:,1));
upto=99;
for zplane=1:upto
    rx=10;ry=10;
    corrected_zplane=255*ones(Ny,Nx);
    if(zplane==1)       
        im1=squeeze(D1(:,:,zplane,1));
        im2=squeeze(D1(:,:,zplane+1,1));
    else
        im1=corrected_zplane;
        im2=squeeze(D1(:,:,zplane+1,1));
    end
    my1=movmean(im1,ry,2);
    my2=movmean(im2,ry,2);
    mx1=movmean(im1,rx,1);%mx1=mx_1';
    mx2=movmean(im2,rx,1);%mx2=mx_2';
    avg1=(mx1+my1)/2;avg2=(mx2+my2)/2;
    diff=avg1-avg2;
    corrected_zplane=double(im2)+diff;
    corrected_block(:,:,zplane+1)=corrected_zplane;
end
%% To produce the high contrast template image (From filtering All 3D Y-planes and X-planes : <<<<  Greyscale Correct version  >>>>>>>)
contrast_block=corrected_block;  % corrected_block the greyscale block resulting from the filtering for interslice jitters
upto=99;
for zplane=1:upto
    rx=10;ry=10;
    contrast_zplane=255*ones(Ny,Nx);
    if(zplane==1)       
        im1=squeeze(D1(:,:,zplane,1));
        im2=squeeze(D1(:,:,zplane+1,1));
    else
        im1=contrast_zplane;
        im2=squeeze(D1(:,:,zplane+1,1));
    end
    my1=movmean(im1,ry,2);
    my2=movmean(im2,ry,2);
    mx1=movmean(im1,rx,1);%mx1=mx_1';
    mx2=movmean(im2,rx,1);%mx2=mx_2';
    avg1=(mx1+my1)/2;avg2=(mx2+my2)/2;
    diff=avg1-avg2;
    contrast_zplane=double(im2)+diff;
    contrast_block(:,:,zplane+1)=contrast_zplane;
end
%% For Saving yplanes and xplanes

for u=1:100
    p1=squeeze(V2(:,:,u));
%     p1_RGB=uint8(ind2rgb(p1,255*summer(256)));
    name=sprintf('z_%04d.tif',u-1);
    cd Zplanes; imwrite(p1,name);cd ..;
end

%% How to see the elevation angles (based on orientation angles obtained in y-plane)
clear;close all;clc;
cd V:\ABI\JZ\Fiber_DTI\Large_block_25um_filtered\New\ST_Results\yplanes;
cd 
% cd V:\ABI\JZ\Fiber_DTI\Large_block_25um_filtered\ST_Results\Zplane\sigma3;
len=560;  %change this for new image
for i=1:len
infile=sprintf('y_%04d.tif',i-1);%outfile=sprintf('cut_%05d.tif',i-500);                       %change this for new image
im=imread(infile);

p1=im(:,:,1);
p2=im(:,:,2);
p3=im(:,:,3);

D1(:,:,i,1)=p1;
D1(:,:,i,2)=p2;
D1(:,:,i,3)=p3;
disp(i);
end

cd V:\ABI\pacedSheep01\Anisotropic\functions;%/hpc/btho733/ABI/pacedSheep01/Anisotropic/functions/;
showcs3(single(D1))  % Here, Zplane is the actual yplane. So go to X -plane and see the view from Actual Z-plane...
%This may not show the angles well. You can't get to the correct figure orientation. So try 2D planes as shown below
figure,imagesc(squeeze(D1(1,:,:,:)));colormap(hsv); %  rotate and adjust the size and scale of figure window
%% cleared version in z-plane
clear;close all;clc;
len=200;  %change this for new image
for i=1:len
infile=sprintf('cs_%04d.tif',i-1);%outfile=sprintf('cut_%05d.tif',i-500);                       %change this for new image
cd V:\ABI\JZ\Fiber_DTI\Large_block_25um_filtered\ST_Results\Zplane\sigma3;im=imread(infile);
orfile=sprintf('cut_%04d.tif',i-1);%outfile=sprintf('cut_%05d.tif',i-500);                       %change this for new image
cd V:\ABI\JZ\Fiber_2016\Work_2017\CentralSection50um3\raw_25um3\;or=imread(orfile);% im=im(y1:y2,x1:x2,:);
p1=im(:,:,1);
p2=im(:,:,2);
p3=im(:,:,3);
o1=or(:,:,1);
o2=or(:,:,2);
o3=or(:,:,3);
p1((o1==255)&(o2==255)&(o3==255))=0;
p2((o1==255)&(o2==255)&(o3==255))=0;
p3((o1==255)&(o2==255)&(o3==255))=0;
D1(:,:,i,1)=p1;
D1(:,:,i,2)=p2;
D1(:,:,i,3)=p3;
disp(i);
end

cd V:\ABI\pacedSheep01\Anisotropic\functions;%/hpc/btho733/ABI/pacedSheep01/Anisotropic/functions/;
showcs3(single(D1));
%% cleared version in y-plane
clear;close all;clc;
cd V:\ABI\JZ\Fiber_DTI\Large_block_25um_filtered;
D=importdata('inverted_ytoz.mat');
len=200;  %change this for new image
for i=1:len
im=squeeze(D(:,:,i,:));
orfile=sprintf('cut_%04d.tif',i-1);%outfile=sprintf('cut_%05d.tif',i-500);                       %change this for new image
cd V:\ABI\JZ\Fiber_2016\Work_2017\CentralSection50um3\raw_25um3\;or=imread(orfile);% im=im(y1:y2,x1:x2,:);
p1=im(:,:,1);
p2=im(:,:,2);
p3=im(:,:,3);
o1=or(:,:,1);
o2=or(:,:,2);
o3=or(:,:,3);
p1((o1==255)&(o2==255)&(o3==255))=0;
p2((o1==255)&(o2==255)&(o3==255))=0;
p3((o1==255)&(o2==255)&(o3==255))=0;
D1(:,:,i,1)=p1;
D1(:,:,i,2)=p2;
D1(:,:,i,3)=p3;
disp(i);
end

cd V:\ABI\pacedSheep01\Anisotropic\functions;%/hpc/btho733/ABI/pacedSheep01/Anisotropic/functions/;
showcs3(single(D1)) 

%%  Turning around Y and z
% RUN the following section first 
% <How to see the elevation angles (based on orientation angles obtained in y-plane)
for i=1:200
    plane=squeeze(D1(i,:,:,:));
    plane1=squeeze(plane(:,:,1));
    plane2=squeeze(plane(:,:,2));
    plane3=squeeze(plane(:,:,3));
    p1=plane1';p2=plane2';p3=plane3';
    rev(:,end:-1:1,i,1)=p1;rev(:,end:-1:1,i,2)=p2;rev(:,end:-1:1,i,3)=p3;
end
save('inverted_ytoz.mat','rev');