function [edgex,edgey ] = func_Edgepoints(x,y,prev_angle,curr_angle,curr_grad)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
startangle=-35;%diag(prev_angle(y,x));

iter=0;
for i=startangle-180:5:startangle+180
    iter=iter+1;
    for d=1:5
        [x4eachangle(d,iter),y4eachangle(d,iter)]=LocationAfterMoveAtAngle(x,y,i,d);% 987,882%1185,615  % 1031,74 % 993,179
    end
    
end
x_column=x4eachangle(:);
y_column=y4eachangle(:);
uniqueneighbors1=unique([x_column y_column],'rows');
uniq_neibrx=uniqueneighbors1(:,1);uniq_neibry=uniqueneighbors1(:,2);
Edgetable(:,1)=abs(diag(prev_angle(uniq_neibry,uniq_neibrx))-diag(curr_angle(uniq_neibry,uniq_neibrx)));
Edgetable(:,2)=diag(curr_grad(uniq_neibry,uniq_neibrx));
Edgetable(:,3)=(mean(Edgetable(:,2))*Edgetable(:,2))./(Edgetable(:,1)+mean(Edgetable(:,2)));
edgex_all=uniq_neibrx(Edgetable(:,3)==max(Edgetable(:,3)));% & (Edgetable(:,3)~=Inf));
edgey_all=uniq_neibry(Edgetable(:,3)==max(Edgetable(:,3)));% & (Edgetable(:,3)~=Inf));
edgex=edgex_all(1);edgey=edgey_all(1);
end

