function [ sobel3D,angles3D ] = IsEdge( x,y,sobel3Dinput,angle3Dinput)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

r=sobel3Dinput(y,x,1);
g=sobel3Dinput(y,x,2);
b=sobel3Dinput(y,x,3);
if(r>160)&&(g>100)&&(b<100)
    sobel3D=1;
else sobel3D=0;
end
angler=angle3Dinput(y,x,1);
angleg=angle3Dinput(y,x,2);
angleb=angle3Dinput(y,x,3);    
if(angler==0)&&(angleg==0)&&(angleb==0)
    angles3D=1;
else angles3D=0;
end
% angleset1=angleset(:,1);
% angleset2=angleset(:,2);
% angleset3=angleset(:,3);
% diff1=abs(angleset1-angleset2);
% diff2=abs(angleset2-angleset3);
% if(sum(diff1,diff2)<30)
%     anglecontinuity=1;
% else anglecontinuity=0;
% end

end

