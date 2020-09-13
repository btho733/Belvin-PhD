function [ x2,y2 ] = LocationAfterMoveAtAngle(x1,y1,angle,dist )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
x=round(dist*sin((180-angle)*pi/180)); % horizontal distance to move
y=round(dist*cos((180-angle)*pi/180)); % vertical distance to move
x2=x1-x;                         % Next X-co-ordinate
y2=y1+y;                         % Next Y-co-ordinate


end

