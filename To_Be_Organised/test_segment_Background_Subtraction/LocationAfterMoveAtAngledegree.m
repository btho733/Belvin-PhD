function [ x2,y2 ] = LocationAfterMoveAtAngledegree(x1,y1,angle,dist )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
x=dist*sind(180-angle); % horizontal distance to move
y=dist*cosd(180-angle); % vertical distance to move
x2=(x1-x);                         % Next X-co-ordinate
y2=(y1+y);                         % Next Y-co-ordinate


end
