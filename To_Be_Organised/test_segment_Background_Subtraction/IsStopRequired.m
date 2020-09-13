function [ check ] =IsStopRequired(x1,y1,x3,y3,Lia,gr1,anglecontinuity)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
check=0;
if (gr1(y3,x3)<10)
    check=4;
end
if(abs(y1-y3)>10)||(abs(x1-x3)>10)
    check=3;
end

if(sum(Lia)>0)
    check=2;
end
if(abs(anglecontinuity(2)-anglecontinuity(1))>150)
    check=1;
end

end

