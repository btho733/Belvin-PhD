function [ out ] = movavg( inp,k )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

B = 1/k*ones(k,1);
out = filter(B,1,inp);


end

