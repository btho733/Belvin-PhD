function [ xnew,ynew ] = func_BestAngleBasedNeighbor(x,y,pivotangle1,gr_angles1,kernelsize1,anglestep1)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
sectorangle1=180;
startangle1=pivotangle1-sectorangle1;
end_angle1=pivotangle1+sectorangle1;

iter1=0;
for j=startangle1:anglestep1:end_angle1
    iter1=iter1+1;
    for d1=1:kernelsize1
        [x4eachangle1(d1,iter1),y4eachangle1(d1,iter1)]=LocationAfterMoveAtAngle(x,y,j,d1);% 987,882%1185,615  % 1031,74 % 993,179
    end
    
end
x_column1=x4eachangle1(:);
y_column1=y4eachangle1(:);
uniqueneighbors1=unique([x_column1 y_column1],'rows','stable');
uniqueneighbors1(:,3)=diag(gr_angles1(uniqueneighbors1(:,2),uniqueneighbors1(:,1)));
for k=1:length(uniqueneighbors1)
    avg_angles(k,1)=func_findavgangle(uniqueneighbors1(k,1),uniqueneighbors1(k,2),uniqueneighbors1(:,3),gr_angles1,1,5);
end
diff=abs(avg_angles-pivotangle1);
% diff=diff';
mindiff=min(diff);
allx=uniqueneighbors1(:,1);ally=uniqueneighbors1(:,2);
setofminx=allx(diff==mindiff);
setofminy=ally(diff==mindiff);
xnew=setofminx(1);ynew=setofminy(1);
end


