function [ avg_angle ] = func_findavgangle(x,y,pivotangle,gr_angles,kernelsize,anglestep)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
sectorangle=180;
startangle=pivotangle-sectorangle;
end_angle=pivotangle+sectorangle;

iter=0;
for m=startangle:anglestep:end_angle
    iter=iter+1;
    for d=1:kernelsize
        [x4eachangle(d,iter),y4eachangle(d,iter)]=LocationAfterMoveAtAngle(x,y,m,d);% 987,882%1185,615  % 1031,74 % 993,179
    end
    
end
x_column=x4eachangle(:);
y_column=y4eachangle(:);
uniqueneighbors=unique([x_column y_column],'rows','stable');
angles=diag(gr_angles(uniqueneighbors(:,2),uniqueneighbors(:,1)));
avg_angle=sum(angles)/length(angles);
end

