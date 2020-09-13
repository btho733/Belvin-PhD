%% Modified Sobel filter
% Generates vertical and horizontal derivatives (ux and uy) for each color plane 
% using two 3x3  convolution kernels 
% Final reconstructs the RGB image from detected edges.
% Separates out regions based on their relative degree of homogeneity.

function [A]=sobel_color(u)

u=double(u);


[M, N, P]=size(u);

ux=zeros(size(u));
uy=zeros(size(u));

for k=1:P
    for i=2:M-1
        for j=2:N-1
            ux(i, j,k)=(u(i+1, j-1,k)-u(i-1, j-1,k)...
                +2*u(i+1, j,k)-2*u(i-1, j,k)...
                +u(i+1, j+1,k)-u(i-1, j+1,k));
            
            uy(i, j,k)=(u(i-1, j+1,k)-u(i-1, j-1,k)...
                +2*u(i, j+1,k)-2*u(i, j-1,k)...
                +u(i+1, j+1,k)-u(i+1, j-1,k));
        end
    end
end
A=sqrt(ux.*ux+uy.*uy);

% The code below is to rescale each plane back to (0,255) range after
% getting ux and uy. Found not quite useful.
% A1=squeeze(A(:,:,1));A2=squeeze(A(:,:,2));A3=squeeze(A(:,:,3));
% A1min=min(min(A1));A1max=max(max(A1));
% A2min=min(min(A2));A2max=max(max(A2));
% A3min=min(min(A3));A3max=max(max(A3));
% 
% A1=255*(A1-A1min)/(A1max-A1min);
% A2=255*(A2-A2min)/(A2max-A2min);
% A3=255*(A3-A3min)/(A3max-A3min);
% Ar(:,:,1)=A1;Ar(:,:,2)=A2;Ar(:,:,3)=A3;

end
