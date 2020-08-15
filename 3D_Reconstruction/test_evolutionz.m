clc;clear;close all;
cd /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/uu1_normalised_from_d_originals;
i=560;
imname=sprintf('cut_%05d.png',i);
im=imresize(imread(imname),0.25);
[mag,angle]=imgradient(im(:,:,1));
cd /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/v1_cutsection_from_S/scale16;
evname=sprintf('Scut_%05d.png',i);
ev=imread(evname);
figure,imagesc(ev);
bin=ev;bin(ev<255)=0;bin(ev==255)=1;
figure,imagesc(bin);

mag(ev==255)=0;angle(ev==255)=500;
figure,imagesc(angle);colormap(jet);
cd /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction;
c=importdata('inputall.mat');
c=c./4;
figure,imagesc(mag);colormap(jet);hold on;plot(c(:,1),c(:,2),'r.');

%% saving scale16_nn_interp (nearest neighbor interpolation)

cd /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/v1_cutsection_from_S;
for imno=560:660
    imname=sprintf('Scut_%05d.png',imno);
    I = imread(imname);
    J = imresize(I, 0.25, 'nearest');
    cd /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/v1_cutsection_from_S/scale16_nn_interp/;
    imwrite(J,imname);cd ..;
end

%% Calling sobel_color on set of files (Output : /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/v2_parametric_snake/a1_sobel_from_uu1)
clear;clc;close all;
for imageno=560:660
    inname=sprintf('cut_%05d.png',imageno);outname=sprintf('sobel_%05d.png',imageno);
    cd /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/uu1_normalised_from_d_originals;i=imread(inname);cd ..;
    A=sobel_color(i);
    cd /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/v2_parametric_snake/a1_sobel_from_uu1/;imwrite(uint8(A),outname);cd ..;
end

%% Remove the blue outer line(Output : /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/v2_parametric_snake/a2_thresh_a1)

clear;clc;close all;

threshr=30;threshb=150;
for imageno=560:660
imagename=sprintf('sobel_%05d.png',imageno);outname=sprintf('thresh_%05d.png',imageno);
cd /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/v2_parametric_snake/a1_sobel_from_uu1;i=imread(imagename);cd ..;
i1=i(:,:,1);i2=i(:,:,2);i3=i(:,:,3);
i1(i1<threshr | i3>threshb)=0;i2(i1<threshr | i3>threshb)=0;i3(i1<threshr | i3>threshb)=0;
after(:,:,1)=i1;after(:,:,2)=i2;after(:,:,3)=i3;
cd /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/v2_parametric_snake/a2_thresh_a1;imwrite(after,outname);cd ..;
end

%% Binarising (Output :  /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/v2_parametric_snake/a3_binarisedfrom_a2)

clear;clc;close all;

for imageno=560:660
inname=sprintf('thresh_%05d.png',imageno);outname=sprintf('bin_%05d.png',imageno);
cd /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/v2_parametric_snake/a2_thresh_a1;in=imread(inname);cd ..;
in1=in(:,:,1);in2=in(:,:,2);in3=in(:,:,3);
[y,x,k]=size(in);
bin=uint8(zeros(y,x));
bin((in1==0)&(in2==0)&(in3==0))=255;
cd /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/v2_parametric_snake/a3_binarisedfrom_a2;imwrite(bin,outname);cd ..;
end

%% Gradient overlay
clc;clear;close all;
cd /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/uu1_normalised_from_d_originals;
i=560;
imname=sprintf('cut_%05d.png',i);
im=imread(imname);
[mag,angle]=imgradient(imresize(im(:,:,1),0.25, 'nearest'));
cd /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/v2_parametric_snake/a5_scale16_nn_interp_a4;
evname=sprintf('s16_%05d.png',i);
ev=imread(evname);
figure,imagesc(ev);

mag(ev==255)=0;
angle(ev==255)=500;
figure,imagesc(angle);colormap(jet);
cd /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction;
c=importdata('inputall.mat');
points=c./4;
hold on;plot(points(:,1),points(:,2),'w.');
figure,imagesc(mag);colormap(jet);
hold on;plot(points(:,1),points(:,2),'r.');

cd /hpc_atog/btho733/ABI/matlab_central/curvature_ANd_Normals;
vertices=unique(points,'rows','stable'); N=LineNormals2D(vertices);
xx=[vertices(:,1)-5*N(:,1) vertices(:,1)-4*N(:,1) vertices(:,1)-3*N(:,1) vertices(:,1)-2*N(:,1) vertices(:,1)-1*N(:,1) vertices(:,1) vertices(:,1)+1*N(:,1) vertices(:,1)+2*N(:,1) vertices(:,1)+3*N(:,1) vertices(:,1)+4*N(:,1) vertices(:,1)+5*N(:,1)]';
yy=[vertices(:,2)-5*N(:,2) vertices(:,2)-4*N(:,2) vertices(:,2)-3*N(:,2) vertices(:,2)-2*N(:,2) vertices(:,2)-1*N(:,2) vertices(:,2) vertices(:,2)+1*N(:,2) vertices(:,2)+2*N(:,2) vertices(:,2)+3*N(:,2) vertices(:,2)+4*N(:,2) vertices(:,2)+5*N(:,2)]';
hold on; plot(xx,yy);


%% FROM HERE STARTS PROCESSING WHOLE IMAGES
%#####################################################

%% Remove the blue outer line(Output : test__segment_Background_Subtraction/v4_parametric_snake_whole/a101_thresh_fromg)

clear;clc;close all;

threshr=30;threshb=150;
for imageno=201:680
imagename=sprintf('g_%05d.png',imageno);outname=sprintf('thresh_%05d.png',imageno);
cd /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/v4_parametric_snake_whole/a100_after_kuw_sobel_fromg;i=imread(imagename);cd ..;
i1=i(:,:,1);i2=i(:,:,2);i3=i(:,:,3);
i1(i1<threshr | i3>threshb)=0;i2(i1<threshr | i3>threshb)=0;i3(i1<threshr | i3>threshb)=0;
after(:,:,1)=i1;after(:,:,2)=i2;after(:,:,3)=i3;
cd /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/v4_parametric_snake_whole/a101_thresh_fromg;imwrite(after,outname);cd ..;
end

%% Binarising (Output :  /test__segment_Background_Subtraction/v4_parametric_snake_whole/a102_binarised_fromg)

clear;clc;close all;

for imageno=201:680
inname=sprintf('thresh_%05d.png',imageno);outname=sprintf('bin_%05d.png',imageno);
cd /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/v4_parametric_snake_whole/a101_thresh_fromg;in=imread(inname);cd ..;
in1=in(:,:,1);in2=in(:,:,2);in3=in(:,:,3);
[y,x,k]=size(in);
bin=uint8(zeros(y,x));
bin((in1==0)&(in2==0)&(in3==0))=255;
cd /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/v4_parametric_snake_whole/a102_binarised_fromg;imwrite(bin,outname);cd ..;
end

%% Remove the blue outer line(Output : test__segment_Background_Subtraction/v4_parametric_snake_whole/a201_thresh_fromd)

clear;clc;close all;

threshr=30;threshb=150;
for imageno=201:680
imagename=sprintf('b200_s4_%05d.png',imageno);outname=sprintf('thresh_s4_%05d.png',imageno);
%cd V:\ABI\pacedSheep01\test__segment_Background_Subtraction\v4_parametric_snake_whole\a200_after_kuw_sobel_fromd;%
cd /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/v4_parametric_snake_whole/b200_after_kuw_sobel_fromd/s4;
i=imread(imagename);cd ..;
i1=i(:,:,1);i2=i(:,:,2);i3=i(:,:,3);
i1(i1<threshr | i3>threshb)=0;i2(i1<threshr | i3>threshb)=0;i3(i1<threshr | i3>threshb)=0;
after(:,:,1)=i1;after(:,:,2)=i2;after(:,:,3)=i3;
 %cd V:\ABI\pacedSheep01\test__segment_Background_Subtraction\v4_parametric_snake_whole\a201_thresh_fromd;
cd /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/v4_parametric_snake_whole/b201_thresh_fromd/s4;
imwrite(after,outname);cd ..;
end

%% Binarising (Output :  /test__segment_Background_Subtraction/v4_parametric_snake_whole/a202_binarised_fromd)

clear;clc;close all;

for imageno=201:680
inname=sprintf('thresh_s4_%05d.png',imageno);outname=sprintf('bin_%05d.png',imageno);
cd /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/v4_parametric_snake_whole/b201_thresh_fromd/s4;in=imread(inname);cd ..;
in1=in(:,:,1);in2=in(:,:,2);in3=in(:,:,3);
[y,x,k]=size(in);
bin=uint8(zeros(y,x));
bin((in1==0)&(in2==0)&(in3==0))=255;
cd /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/v4_parametric_snake_whole/b202_binarised_fromd/s4;imwrite(bin,outname);cd ..;
end
%% Greyscale instead of bInarising (Output :  /test__segment_Background_Subtraction/v4_parametric_snake_whole/a203_binarised_fromd)

clear;clc;close all;

for imageno=201:680
inname=sprintf('thresh_%05d.png',imageno);outname=sprintf('grey_%05d.png',imageno);origname=sprintf('s4_%05d.png',imageno);
cd V:\ABI\pacedSheep01\test__segment_Background_Subtraction\v4_parametric_snake_whole\a201_thresh_fromd;in=imread(inname);cd ..;
cd V:\ABI\pacedSheep01\test__segment_Background_Subtraction\d_images2dcoarsesegment_s4;orig=imread(origname);cd ..;
origr=imresize(orig,0.25);
in1=in(:,:,1);in2=in(:,:,2);in3=in(:,:,3);
[y,x,k]=size(in);
bin=origr(1:622,1:1035,1);%% This is using the redplane. Instead of this, rgb2gray can be tried
bin((in1==0)&(in2==0)&(in3==0))=255;
cd V:\ABI\pacedSheep01\test__segment_Background_Subtraction\v4_parametric_snake_whole\a203_greyscale_fromd;imwrite(bin,outname);cd ..;
end
%% Greyscale instead of bInarising : for b series(Output :  /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/v4_parametric_snake_whole/b203_greyscale_fromd/s4)

clear;clc;close all;

for imageno=201:680
inname=sprintf('s4_%05d.png',imageno);outname=sprintf('grey_%05d.png',imageno);binname=sprintf('bin_%05d.png',imageno);
cd /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/d_images2dcoarsesegment_s4;in=imread(inname);cd ..;
cd /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/v4_parametric_snake_whole/b202_binarised_fromd/s4;bin=imread(binname);cd ..;
out=rgb2gray(in);
out(bin==255)=255;
cd /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/v4_parametric_snake_whole/b203_greyscale_fromd/s4;imwrite(out,outname);cd ..;
end

%% Overlaying using Amira mesh [output : a207_overlaid_using_Amira_labels]

clc;close all;clear;
for i=1:61
cd /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/v4_parametric_snake_whole/a207_overlaid_using_Amira_labels;
infile=sprintf('a207_overlaid_%05d.png',i);inp=imread(infile);
cd /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/v4_parametric_snake_whole/a206_labels_from_Amira_mesh_endo;
labelfile=sprintf('a206_endo_%05d.png',i);label=imread(labelfile);
labelr=imresize(label,0.5);
out=inp;out(labelr==1)=255;
cd /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/v4_parametric_snake_whole/a207_overlaid_using_Amira_labels_epi_and_endo;
outfile=sprintf('a207_overlaid_%05d.png',i);imwrite(out,outfile);
end
%% Making a209 For ITK
clc;close all;clear;
for i=1:61
cd /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/v4_parametric_snake_whole/a210_closed_froma207;
infile=sprintf('a210_%05d.tif',i);inp=imread(infile);
out=inp;out(inp<255)=0;
cd /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/v4_parametric_snake_whole/a211_For_ITK_from_a210;
outfile=sprintf('a211_%05d.tif',i);imwrite(out,outfile);
end

%% saving scale32_nn_interp (nearest neighbor interpolation) [output :a106 or a107]
clc;clear;close all;

for imno=201:680
    cd /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/v4_parametric_snake_whole/a104_median_filtered_0001;
    imname=sprintf('a104_%05d.png',imno);
    outname=sprintf('a106_%05d.png',imno);
    I = imread(imname);
    J = imresize(I, 0.5, 'nearest');
    cd /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/v4_parametric_snake_whole/a106_scale32_resizedfrom_a104;
    imwrite(J,outname);cd ..;
end

%%  VTK write (tip : execute in hpc3 for faster performance)  [output: /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/uu1_normalised_from_d_originals/a106.vtk]
clear;clc;
cd /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/v4_parametric_snake_whole/a106_scale32_resizedfrom_a104;
for i=1:480
    in=sprintf('a106_%05d.png',i+200);
    im=imread(in);
    A(:,:,i)=im;
end
cd /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/uu1_normalised_from_d_originals;
vtkwrite('a106.vtk', 'structured_points', 'scale32', A);

%% Resizing using interp3

% png to mat
clear;clc;close all;
cd /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/v4_parametric_snake_whole/b203_greyscale_fromd/s4;
for i=1:480
infile=sprintf('grey_%05d.png',i+200);%outfile=sprintf('cut_%05d.tif',i-500);
im=imread(infile);
S(:,:,i)=im;
disp(i);
end
% cd  /hpc/btho733/ABI/pacedSheep01;figure,h = vol3d('cdata',S); view(3);axis equal;view(54,18);title('stain normalised');box on;alphamap('rampup',3);
S=double(S);
% [Nx_input, Ny_input, Nz_input] = size(mat);
    Nx_input = 4140;
    Ny_input = 2490;
    Nz_input = 480;
    % extract the desired number of pixels
    Nx_output = 414;
    Ny_output = 249;
    Nz_output = 48;        

    % update command line status
    disp(['  input grid size: ' num2str(Ny_input) ' by ' num2str(Nx_input) ' by ' num2str(Nz_input) ' elements']);
    disp(['  output grid size: ' num2str(Ny_output) ' by ' num2str(Nx_output) ' by ' num2str(Nz_output) ' elements']); 

    % create normalised plaid grids of current discretisation
    [x_mat, y_mat, z_mat] = ndgrid((0:Ny_input-1)/(Ny_input-1), (0:Nx_input-1)/(Nx_input-1), (0:Nz_input-1)/(Nz_input-1));       

    % create plaid grids of desired discretisation
    [x_mat_interp, y_mat_interp, z_mat_interp] = ndgrid((0:Ny_output-1)/(Ny_output-1), (0:Nx_output-1)/(Nx_output-1), (0:Nz_output-1)/(Nz_output-1));

    % compute interpolation; for a matrix indexed as [M, N, P], the
    % axis variables must be given in the order N, M, P
    mat_rs = interp3(y_mat, x_mat, z_mat, S, y_mat_interp, x_mat_interp, z_mat_interp, 'cubic');        
    um=uint8(mat_rs);
    
    cd /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/v4_parametric_snake_whole/b204_resize_cubic_interp3;
    for i=1:48
    slice_i=um(:,:,i);    
    outname=sprintf('s40_250um_%05d.png',i);
    imwrite(slice_i,outname);
    end
    
%     cd /hpc/btho733/ABI/pacedSheep01;figure,h = vol3d('cdata',um); view(3);axis equal;view(54,18);box on;alphamap('rampup',3);
    
%%   From 14 Feb 2017
% 1. Resize from v4/a205a
clc;clear;close all;
for i=1:61
    imname=sprintf('s16_%05d.png',i); outname=sprintf('s32_%05d.png',i);
cd /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/v4_parametric_snake_whole/a205a_1in8_fromd_for_reference;im=imread(imname);imr=imresize(im,0.5);
cd /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/v4_parametric_snake_whole/a212_s32_originals_fora207;imwrite(imr,outname);
end
%%
% 2. Generate the 3 images
clc;clear;close all;
i=49;
imname=sprintf('s32_%05d.png',i);
cd /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/v4_parametric_snake_whole/a212_s32_originals_fora207;im=imread(imname);
[mag,angle]=imgradient(im(:,:,1));
cd /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/v4_parametric_snake_whole/a207_overlaid_using_Amira_labels;
evname=sprintf('a207_overlaid_%05d.png',i);ev=imread(evname);figure,imagesc(ev);points=importdata('input_a207_46.mat');
hold on;plot(points(:,1),points(:,2),'r.');
mag(ev==255)=0;
angle(ev==255)=500;
figure,imagesc(angle);colormap(jet);
    hold on;plot(points(:,1),points(:,2),'w.');
figure,imagesc(mag);colormap(jet);
hold on;plot(points(:,1),points(:,2),'r.');
spcv=cscvn(points');
p=fnplt(spcv,'w',2);
figure,imagesc(im);hold on;plot(p(1,:),p(2,:),'w.','LineWidth',1.6);
% 
% cd /hpc_atog/btho733/ABI/matlab_central/curvature_ANd_Normals;
% vertices=unique(p','rows','stable'); N=LineNormals2D(vertices);
% xx=[vertices(:,1)-5*N(:,1) vertices(:,1)-4*N(:,1) vertices(:,1)-3*N(:,1) vertices(:,1)-2*N(:,1) vertices(:,1)-1*N(:,1) vertices(:,1) vertices(:,1)+1*N(:,1) vertices(:,1)+2*N(:,1) vertices(:,1)+3*N(:,1) vertices(:,1)+4*N(:,1) vertices(:,1)+5*N(:,1)]';
% yy=[vertices(:,2)-5*N(:,2) vertices(:,2)-4*N(:,2) vertices(:,2)-3*N(:,2) vertices(:,2)-2*N(:,2) vertices(:,2)-1*N(:,2) vertices(:,2) vertices(:,2)+1*N(:,2) vertices(:,2)+2*N(:,2) vertices(:,2)+3*N(:,2) vertices(:,2)+4*N(:,2) vertices(:,2)+5*N(:,2)]';
% hold on; plot(xx,yy);

%% Overlaying a207 on a213 
clc;clear;
for i=1:61
imname=sprintf('s32_%05d.png',i);
cd /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/v4_parametric_snake_whole/a212_s32_originals_fora207;im=imread(imname);
im1=im(:,:,1);im2=im(:,:,2);im3=im(:,:,3);
cd /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/v4_parametric_snake_whole/a209_For_ITK_Binarisedfrom_a208;
evname=sprintf('a209_%05d.tif',i);ev=imread(evname);
im1(ev==255)=0;im2(ev==255)=0;im3(ev==255)=0;
out(:,:,1)=im1;out(:,:,2)=im2;out(:,:,3)=im3;
cd /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/v4_parametric_snake_whole/a214_s32_overlaid_using_a209;
imwrite(out,imname);
end
%% Save points

[x,y]=getpts;
input_a207_46=[x y];
save('input_a207_46.mat','input_a207_46');
%% TRYING TO REMOVE AMIRA FROM PIPELINE VIA Delayed BGD SUBTRACTION

%% Remove the blue outer line(Output : test__segment_Background_Subtraction/v4_parametric_snake_whole/a201_thresh_fromd)

clear;clc;close all;

threshr=30;threshb=150;
outname=sprintf('a03a_thresh_on_a02.png');

cd /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/v5_TrytoRemove_Amira_from_pipeline;
i=imread('a02_s16_sobelkuw17ats4_BG_00710.png');
i1=i(:,:,1);i2=i(:,:,2);i3=i(:,:,3);
i1(i1<threshr | i3>threshb)=0;i2(i1<threshr | i3>threshb)=0;i3(i1<threshr | i3>threshb)=0;
after(:,:,1)=i1;after(:,:,2)=i2;after(:,:,3)=i3;
imwrite(after,outname);
%% Binarising (Output :  /test__segment_Background_Subtraction/v4_parametric_snake_whole/a202_binarised_fromd)

clear;clc;close all;
cd /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/v5_TrytoRemove_Amira_from_pipeline;in=imread('a02a_s16_sobelkuw17ats4_BG_00710.png');
in1=in(:,:,1);in2=in(:,:,2);in3=in(:,:,3);
[y,x,k]=size(in);
bin=uint8(zeros(y,x));
bin((in1==0)&(in2==0)&(in3==0))=255;
imwrite(bin,'a02b_threshBinarised_from_a02a.png');

%% Resize to s32
clear;clc;
cd /hpc/btho733/ABI/pacedSheep01/Backgrounds/;
tic;
parfor i=1:137
imname=sprintf('BG_%05d.png',10*i-10);
im=imread(imname);
imr=imresize(im,0.03125);
cd s32;imwrite(imr(1:311,:,:),imname);cd ..;
end
toc;
%% Background subtraction (a05-a04)
clear;clc;close all;
cd /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/v5_TrytoRemove_Amira_from_pipeline;in=imread('a05_r8_a205_s32_00056.png');
bgd=imread('a04c_open_a04a.png');
in(bgd<255)=255;
imwrite(in,'a06_bgdsubtracted_00056.png');

%%  Generating overlaid figures to show my requirement
clc;clear;
cd /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/z1_Results_for_writing;
im=imread('r1_s4_00641.png');
imr=imresize(im,0.125);
im=imr(1:311,:,:);
r8=imread('r8_a207_final_00056.png');
im1=im(:,:,1);im2=im(:,:,2);im3=im(:,:,3);
im1(r8==255)=255;im2(r8==255)=255;im3(r8==255)=255;
out(:,:,1)=im1;out(:,:,2)=im2;out(:,:,3)=im3;
cd /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/v5_TrytoRemove_Amira_from_pipeline;
imwrite(out,'a07_from_r8_a207.png');
