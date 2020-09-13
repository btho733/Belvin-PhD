function [ All_neighborcontrolpts,stop ] = NearestControlPoint(x,y,pivotangle,controlpts,kernelsize,anglestep)
%You have a pivot angle - the pred_angle from previous point
%
%   Detailed explanation goes here
startangle=pivotangle-90;
end_angle=pivotangle+90;
disp('Inside NearestControlpointFunction');
iter=0;
for i=startangle:anglestep:end_angle
    iter=iter+1;
    for d=1:kernelsize
        [x4eachangle(d,iter),y4eachangle(d,iter)]=LocationAfterMoveAtAngle(x,y,i,d);% 987,882%1185,615  % 1031,74 % 993,179
    end
    
end
x_column=x4eachangle(:);
y_column=y4eachangle(:);
uniqueneighbors=unique([x_column y_column],'rows');
deleterow=ismember(uniqueneighbors,[x y],'rows');
uniqueneighbors(deleterow==1)=[];
Lia2=ismember(uniqueneighbors,controlpts,'rows');
match_controlpts=sum(Lia2);
if(match_controlpts<1)
    stop=1;
    All_neighborcontrolpts=[x y];
    disp('No matching control points ahead');
else
    x_uniqueneighbors=uniqueneighbors(:,1);
    y_uniqueneighbors=uniqueneighbors(:,2);
    x_required=x_uniqueneighbors(Lia2==1);
    y_required=y_uniqueneighbors(Lia2==1);
    All_neighborcontrolpts=[x_required y_required];
    stop=0;
end

end

