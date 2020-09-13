function [cel] = inORout(x,y,pivotangle,sob,kernelsize,anglestep )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
startangle=pivotangle;
end_angle1=pivotangle+180;
end_angle2=pivotangle-180;
sob1=sob(:,:,1);

iter=0;
for i=startangle:anglestep:end_angle1
    iter=iter+1;
    for d=1:kernelsize
        [x4eachangle(d,iter),y4eachangle(d,iter)]=LocationAfterMoveAtAngle(x,y,i,d);% 987,882%1185,615  % 1031,74 % 993,179
    end
    
end
x_column=x4eachangle(:);
y_column=y4eachangle(:);
uniqueneighbors1=unique([x_column y_column],'rows');
% clear x4eachangle;clear y4eachangle;clear x_column;clear y_column;clear d;clear i;



iter=0;
for i=startangle:-anglestep:end_angle2
    iter=iter+1;
    for d=1:kernelsize
        [x4eachangle2(d,iter),y4eachangle2(d,iter)]=LocationAfterMoveAtAngle(x,y,i,d);% 987,882%1185,615  % 1031,74 % 993,179
    end
    
end
x_column=x4eachangle2(:);
y_column=y4eachangle2(:);
uniqueneighbors2=unique([x_column y_column],'rows');
out=diag(sob1(uniqueneighbors1(:,2),uniqueneighbors1(:,1)));
in=diag(sob1(uniqueneighbors2(:,2),uniqueneighbors2(:,1)));
sumout=sum(out);sumin=sum(in);
if sumout>sumin
    direction=1;
    coords=uniqueneighbors1;
else
    direction=0;
    coords=uniqueneighbors2;
end
cel={coords;direction};
end

