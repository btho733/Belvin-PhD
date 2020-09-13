function [ xnew,ynew ] = BestEdgeNeighbor2(coords_in,sob,angu,grtarget)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
check1=zeros(length(coords_in),1);
check2=zeros(length(coords_in),1);
check3=zeros(length(coords_in),1);
%check4=zeros(length(coords),1);
x=coords_in(:,1);y=coords_in(:,2);
r=diag(sob(y,x,1));
g=diag(sob(y,x,2));
b=diag(sob(y,x,3));
check1((r>150)&(g>100)&(b<100))=1;

angler=diag(angu(y,x,1));
angleg=diag(angu(y,x,2));
angleb=diag(angu(y,x,3)); 
check2((angler==0)&(angleg==0)&(angleb==0))=1;

gr=diag(grtarget(y,x));
check3(gr==max(gr))=1;

result=2*check1+check2+5*check3;
sel_y=y(result==max(result));
sel_x=x(result==max(result));
xnew=sel_x(1);ynew=sel_y(1);

end

