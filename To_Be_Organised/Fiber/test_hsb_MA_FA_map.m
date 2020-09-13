cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/mid_section_from_rgboutpng_251to650/results_s1m3rho3k15;
load('FA.mat')
load('MA.mat')
h=(MA+1)./2; % converting from (-1,1) to (0,1)
s=ones(900,1200,400);
b=FAv;
hsb1=single(zeros(900,1200,400,3));

for i=1:400
    hsb1(:,:,i,1)=h(:,:,i);hsb1(:,:,i,2)=s(:,:,i);hsb1(:,:,i,3)=b(:,:,i);rgb1(:,:,i,:)=hsv2rgb(squeeze(hsb1(:,:,i,:)));
end
cd /hpc/btho733/ABI/pacedSheep01/Anisotropic/functions/;
showcs3(rgb1(:,:,end:-1:1,:))
%%
clear;
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI/Whole_Atria/mid_section_from_rgboutpng_251to650/results_s1m3rho10k50;
load('FA.mat')
load('MA.mat')
h=(MA+1)./2; % converting from (-1,1) to (0,1)
s=ones(800,1000,400);
b=FAv;
hsb1=single(zeros(800,1000,400,3));

for i=1:400
    hsb1(:,:,i,1)=h(:,:,i);hsb1(:,:,i,2)=s(:,:,i);hsb1(:,:,i,3)=b(:,:,i);rgb1(:,:,i,:)=hsv2rgb(squeeze(hsb1(:,:,i,:)));
end
cd /hpc/btho733/ABI/pacedSheep01/Anisotropic/functions/;
showcs3(rgb1(:,:,end:-1:1,:))