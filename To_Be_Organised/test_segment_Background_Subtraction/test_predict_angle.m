x1=1033;y1=77;actual_angle=-135.5;kernelsize=5;anglestep=10;

startangle=actual_angle-120;
end_angle=actual_angle+120;

iter=0;
for i=startangle:anglestep:end_angle
    iter=iter+1;
    for d=1:kernelsize
        [x2(d),y2(d)]=LocationAfterMoveAtAngle(x1,y1,i,d);% 987,882%1185,615  % 1031,74 % 993,179
    end
    x3=x2';
    y3=y2';
    grad(iter)=sum(diag(gr(y3,x3)));
end
n=find(grad==max(grad));
pred_angle=startangle+(n-1)*anglestep;