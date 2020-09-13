%% Make mask outputs (output: f_output_masks)

clear;clc;close all;cd /hpc_atog/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/e_coords2segment_s4/
[coords_All]=uigetfile('.mat','Select the co-ordinates','multiselect','on');cd ..;
FileStr=char(coords_All);
p=4140;q=2490;
for imageno=201:567
    cd d_images2dcoarsesegment_s4;imname=sprintf('s4_%05d.png',imageno);im=imread(imname);cd ..;
    cno=round(imageno/10)*10;
    sel_files=FileStr((str2num(FileStr(:,3:7))==cno),:);
    endo_files=sel_files((sel_files(:,8)=='i'),:);n_endo=size(endo_files,1);
    epi_files=sel_files((sel_files(:,8)=='o'),:);n_epi=size(epi_files,1);
    mask=zeros(q,p);
    for obj=1:n_epi
        cd e_coords2segment_s4;cname=sprintf('c_%05do_%02d.mat',cno,obj);c=importdata(cname);cd ..;
        mask_epi = poly2mask(c(:,1),c(:,2),q,p);
        mask(mask_epi==1)=1;clear mask_epi;
    end
    
    for obj=1:n_endo
        cd e_coords2segment_s4;cname=sprintf('c_%05di_%02d.mat',cno,obj);c=importdata(cname);cd ..;
        mask_endo= poly2mask(c(:,1),c(:,2),q,p);
        mask(mask_endo==1)=0;clear mask_endo;
    end
    outname=sprintf('mask_%05d.png',imageno);
    mask(mask==1)=255;cd f_output_masks;imwrite(mask,outname);cd ..;
end

%% Overlay (Output : g_coarse_segmented_C)
clear;clc;close all;cd /hpc_atog/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/
for Image_no=201:567
infile=sprintf('s4_%05d.png',Image_no);
cd d_images2dcoarsesegment_s4;original=imread(infile);cd ..;
originalr=original(:,:,1);originalg=original(:,:,2);originalb=original(:,:,3);
maskname=sprintf('mask_%05d.png',Image_no);
cd f_output_masks;mask=imread(maskname);cd ..;
originalr(mask==0)=255;originalg(mask==0)=255;originalb(mask==0)=255;
segmented(:,:,1)=originalr;segmented(:,:,2)=originalg;segmented(:,:,3)=originalb;
outfile=sprintf('seg_%05d.png',Image_no);disp(['Image No  ',num2str(Image_no),'  Generated in g_segmented folder']);
cd g_coarse_segmented_C;imwrite(segmented,outfile);cd ..;
end

%%
clear;clc;
cd /hpc/btho733/ABI/pacedSheep01/PacedSheep01_RGB_new_sub_corrected;i=imread('Image_00610.png');cd ../test__segment_Background_Subtraction;
im=imresize(i,0.25);[A]=sobel_color(im);
figure,imagesc(uint8(A))


%% Calling sobel_color for one file (with input unsegmented original)
clear;clc;
cd d_images2dcoarsesegment_s4/;i=imread('s4_00610.png');cd ..;
[A]=sobel_color(i);
figure,imagesc(uint8(A))

%% Calling sobel_color for one file (with input C)
clear;clc;
cd g_coarse_segmented_C/;i=imread('seg_00610.png');cd ..;
A=sobel_color(i);
figure,imagesc(uint8(A))
%% Calling sobel_color for one file (with input D)
clear;clc;
cd n_Difference_D/;i=imread('Diff_00583.png');cd ..;
A=sobel_color(i);
figure,imagesc(uint8(A))
%% Calling sobel_color for one file (with input R)
clear;clc;
cd t_1-R;i=imread('cut_00437.tif');cd ..;
[A]=sobel_color(i);
figure,imagesc(uint8(A))
% figure,imagesc(2*uint8(Ar))
%% Threshold remove blue outline for one image
threshr=30;threshb=120;
i=uint8(A);
i1=i(:,:,1);i2=i(:,:,2);i3=i(:,:,3);
i1(i1<threshr | i3>threshb)=0;i2(i1<threshr | i3>threshb)=0;i3(i1<threshr | i3>threshb)=0;
after(:,:,1)=i1;after(:,:,2)=i2;after(:,:,3)=i3;
figure,imagesc(uint8(after))

%% Calling sobel_color on set of files (Output : l_25um_fullset_sobel_R)
clear;clc;
for imageno=201:680
    inname=sprintf('AllbutR_%05d.png',imageno);outname=sprintf('sobel_%05d.png',imageno);
    cd /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/t0_1-R;i=imread(inname);cd ..;
    A=sobel_color(i);
    cd t1_sobel_of_t0;imwrite(uint8(A),outname);cd ..;
end
%% Remove the blue outer line(Output : m_25um_fullset_thresh_R)

clear;clc;cd /hpc_atog/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction;

threshr=30;threshb=150;
for imageno=201:680
imagename=sprintf('sobel_%05d.png',imageno);outname=sprintf('thresh_%05d.png',imageno);
cd q_25um_fullset_sobel_C;i=imread(imagename);cd ..;
i1=i(:,:,1);i2=i(:,:,2);i3=i(:,:,3);
i1(i1<threshr | i3>threshb)=0;i2(i1<threshr | i3>threshb)=0;i3(i1<threshr | i3>threshb)=0;
after(:,:,1)=i1;after(:,:,2)=i2;after(:,:,3)=i3;
cd r_25um_fullset_thresh_C;imwrite(after,outname);cd ..;
%figure,imagesc(after)
end

%% Making n_Difference images D= C-R; (Output : n_Difference_D)

clear;clc;
cd /hpc_atog/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/;
for imageno=568:680;
Cname=sprintf('seg_%05d.png',imageno);Rname=sprintf('Seg_%05d.png',imageno);outname=sprintf('Diff_%05d.png',imageno);
cd g_coarse_segmented_C;C=imread(Cname);cd ..;
C1=C(:,:,1);C2=C(:,:,2);C3=C(:,:,3);
cd k_25um_fullset_images_R;R=imread(Rname);cd ..;
R1=R(:,:,1);R2=R(:,:,2);R3=R(:,:,3);
D1=C1;D2=C2;D3=C3;
D1((C1==R1)&(C2==R2)&(C3==R3))=255;
D2((C1==R1)&(C2==R2)&(C3==R3))=255;
D3((C1==R1)&(C2==R2)&(C3==R3))=255;
D(:,:,1)=D1;D(:,:,2)=D2;D(:,:,3)=D3;
cd n_Difference_D;imwrite(D,outname);cd ..;
end



%% Testing binarisation

clear;clc;
cd /hpc_atog/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction/;
imageno=610;
inname=sprintf('noisywax_selected_in_white_from_m_thresh00610.png');
Rname=sprintf('Seg_%05d.png',imageno);outname=sprintf('waxcleared_%05d.png',imageno);
in=imread(inname);
cd k_25um_fullset_images_R/;R=imread(Rname);cd ..;
R1=R(:,:,1);R2=R(:,:,2);R3=R(:,:,3);
in1=in(:,:,1);in2=in(:,:,2);in3=in(:,:,3);
o1=R1;o2=R2;o3=R3;
o1((in1==0)&(in2==0)&(in3==0))=255;
o2((in1==0)&(in2==0)&(in3==0))=255;
o3((in1==0)&(in2==0)&(in3==0))=255;
o1((in1==255)&(in2==255)&(in3==255))=255;
o2((in1==255)&(in2==255)&(in3==255))=255;
o3((in1==255)&(in2==255)&(in3==255))=255;

o1((o1<255)&(o2<255)&(o3<255))=0;
o2((o1<255)&(o2<255)&(o3<255))=0;
o3((o1<255)&(o2<255)&(o3<255))=0;
o(:,:,1)=o1;o(:,:,2)=o2;o(:,:,3)=o3;
figure,imagesc(o);

%% Normalising m folder (Output : o_normalised_thresh_R)  - copy contents from output_RH

clc;clear;close all;
cd /hpc/btho733/ABI/pacedSheep01/medsci_poster/stain_normalisation_toolbox/pacedsheepimages/m_25umfullset_thresh_R_fromtest_segment_BGsub_and_color_thresh;
%cd /hpc/btho733/ABI/pacedSheep01/medsci_poster/stain_normalisation_toolbox/pacedsheepimages/f3v4;
TargetImage = imread('thresh_00583.png');

for i=568:680    
Sourcename=sprintf('thresh_%05d.png',i);outfile=sprintf('cut_%05d.png',i);
SourceImage = imread(Sourcename);
cd /hpc/btho733/ABI/pacedSheep01/medsci_poster/stain_normalisation_toolbox/StainNormalisation;
% [ NormHS ] = Norm( SourceImage, TargetImage, 'RGBHist');
[ NormRH ] = Norm( SourceImage, TargetImage, 'Reinhard');
% [ NormMM ] = Norm(SourceImage, TargetImage, 'Macenko', 255, 0.15, 1);
cd /hpc/btho733/ABI/pacedSheep01/medsci_poster/stain_normalisation_toolbox/pacedsheepimages/m_25umfullset_thresh_R_fromtest_segment_BGsub_and_color_thresh;
% cd output_HS;imwrite(NormHS,outfile);cd ..;
cd output_RH;imwrite(NormRH,outfile);cd ..;
% cd output_MM;imwrite(NormMM,outfile);cd ..;
end


%% Binarising (Output :  p_Binary_normthresh_R)

clear;clc;close all;
cd /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction;
for imageno=201:680
inname=sprintf('thresh_%05d.png',imageno);outname=sprintf('bin_%05d.png',imageno);
cd r_25um_fullset_thresh_C;in=imread(inname);cd ..;
in1=in(:,:,1);in2=in(:,:,2);in3=in(:,:,3);
[y,x,k]=size(in);
bin=uint8(zeros(y,x));
bin((in1==0)&(in2==0)&(in3==0))=255;
cd s_Binary_normthresh_C;imwrite(bin,outname);cd ..;
end


%% Getting 1-R
clear;clc;
cd /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction;
for imageno=201:680
inname=sprintf('Seg_%05d.png',imageno);outname=sprintf('AllbutR_%05d.png',imageno);
cd k_25um_fullset_images_R;in=imread(inname);cd ..;
in1=in(:,:,1);in2=in(:,:,2);in3=in(:,:,3);
cd d_images2dcoarsesegment_s4;imname=sprintf('s4_%05d.png',imageno);im=imread(imname);cd ..;
im1=im(:,:,1);im2=im(:,:,2);im3=im(:,:,3);
im1((in1<255)&(in2<255)&(in3<255))=255;
im2((in1<255)&(in2<255)&(in3<255))=255;
im3((in1<255)&(in2<255)&(in3<255))=255;
out(:,:,1)=im1;out(:,:,2)=im2;out(:,:,3)=im3;
cd t_1-R;imwrite(out,outname);cd ..;
end


%% processing 1-R
clear;clc;close all;
cd t_1-R;i=imread('AllbutR_00555.png');cd ..;
A=sobel_color(i);
figure,imagesc(uint8(A));
threshr=30;threshb=120;threshg=80;
i=uint8(A);
i1=i(:,:,1);i2=i(:,:,2);i3=i(:,:,3);
i1(i1<threshr |i2>threshg | i3>threshb)=0;i2(i1<threshr |i2>threshg | i3>threshb)=0;i3(i1<threshr |i2>threshg | i3>threshb)=0;
after(:,:,1)=i1;after(:,:,2)=i2;after(:,:,3)=i3;
figure,imagesc(uint8(10*after));title('After sobel and thresh');
in=after;
in1=in(:,:,1);in2=in(:,:,2);in3=in(:,:,3);
[y,x,k]=size(in);
bin=uint8(255*ones(y,x));
bin((in1>0)&(in2>0)&(in3>0))=0;
figure,imagesc(bin);
imwrite(bin,'c_AllbutR_00555_binary.png');

%%

