%% Cross_shaped Version : Filtering All 3D Y-planes and X-planes : <<<<   RGB Correct version  >>>>>>>
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
    diff=avg1-avg2;l
    corrected_zplane=double(im2)+diff;
    corrected_block(:,:,zplane+1)=corrected_zplane;
end

%% Square Version : New(using movmean2D) Filtering for All 3D Y-planes and X-planes : <<<<   RGB Correct version  >>>>>>>
cd /hpc_atog/btho733/ABI/JZ/Fiber_DTI;
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
%     my1=movmean(im1,ry,2);
%     my2=movmean(im2,ry,2);
%     mx1=movmean(im1,rx,1);%mx1=mx_1';
%     mx2=movmean(im2,rx,1);%mx2=mx_2';
%     avg1=(mx1+my1)/2;avg2=(mx2+my2)/2;
    avg1=movmean2D(im1,rx,ry);
    avg2=movmean2D(im2,rx,ry);
    diff=avg1-avg2;
    corrected_zplane=im2+diff;
    corrected_block(:,:,zplane+1)=corrected_zplane;
end

%% Movmean2D
%Moving Average 2-D filter
%X contains values to be filtered, HL is  
%horizontal while VL is vertical limit.
%HL adds columns and VL adds rows to padded matrix
function [y]=movmean2D(x,hl,vl) 
if nargin<2
    hl=1;
    vl=1;
end    
if (vl<0)||(hl<0)
    error('limits must be positive')
end

x=im2double(x);
[row,col,space]=size(x);    %Checking size of input
y=0*x;     %output of same size and type as input

for s=1:space   %For computation across all slices
    x_pad=zeros(row+2*vl,col+2*hl); %Padded Matrix (hl increase col)
    x_pad(1+vl:row+vl,1+hl:col+hl)=x(:,:,s);  
        for i=1+vl:row+vl     %Going through rows
            for j=1+hl:col+hl     %Going through values one by one
                y(i-vl,j-hl,s)=sum(x_pad(i,j-hl:j+hl))+sum(x_pad(i-vl:i+vl,j));
                 %Summing Values around current entry according 2 vl and hl
                y(i-vl,j-hl,s)=y(i-vl,j-hl,s)-x_pad(i,j);%Current Values gets added
                %twice. Once in row and once in Col
                
            end
        end
    
end
y=y/(2*hl+2*vl+1);
y=im2uint8(y);
end