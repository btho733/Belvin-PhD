%% Calling sobel_color for one file
clear;clc;
cd k_25um_fullset_images_R/;i=imread('Seg_00610.png');cd ..;
A=sobel_color(i);
figure,imagesc(uint8(5*A))

%% Calling sobel_color on set of files


%% Making n_Difference images D= C-R;

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


%%

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

%% Normalising m folder

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


%% Binarising to get p_Binary_normthresh_R

clear;clc;close all;
cd /hpc/btho733/ABI/pacedSheep01/test__segment_Background_Subtraction;
for imageno=568:680
inname=sprintf('thresh_%05d.png',imageno);outname=sprintf('bin_%05d.png',imageno);
cd m_25um_fullset_thresh_R;in=imread(inname);cd ..;
in1=in(:,:,1);in2=in(:,:,2);in3=in(:,:,3);
[y,x,k]=size(in);
bin=uint8(zeros(y,x));
bin((in1==0)&(in2==0)&(in3==0))=255;
cd p_Binary_normthresh_R;imwrite(bin,outname);cd ..;
nd