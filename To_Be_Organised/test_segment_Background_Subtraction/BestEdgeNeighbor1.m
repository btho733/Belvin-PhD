function [x3,y3]=BestEdgeNeighbor1(coords,sob,angu,gr1)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
check1=zeros(length(coords),1);
check2=zeros(length(coords),1);
check3=zeros(length(coords),1);

x=coords(:,1);y=coords(:,2);
r=diag(sob(y,x,1));
g=diag(sob(y,x,2));
b=diag(sob(y,x,3));
check1((r>200)&(g>100)&(b<100))=1;
  
angler=diag(angu(y,x,1));
angleg=diag(angu(y,x,2));
angleb=diag(angu(y,x,3));    
check2((angler==0)&(angleg==0)&(angleb==0))=1;

gr=diag(gr1(y,x));
check3(gr==max(gr))=1;


result=2*check1+check2+5*check3;
sel_y=y(result==max(result));
sel_x=x(result==max(result));
x3=sel_x(1);y3=sel_y(1);
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

