x=[1 2 3 4 5;6 7 8 9 10;11 12 13 14 15;16 17 18 19 20;21 22 23 24 25];
x=im2double(x);
[row,col,space]=size(x);
vl=3;hl=3;
numelem=hl*vl;
for s=1:space   %For computation across all slices
    x_pad=zeros(row+2*vl,col+2*hl); %Padded Matrix (hl increase col)
    x_pad(1+vl:row+vl,1+hl:col+hl)=x(:,:,s);  
        for i=1+vl:row+vl     %Going through rows
            for j=1+hl:col+hl     %Going through values one by one
                y(i-vl,j-hl,s)=sum(sum(x_pad(ceil(i-vl/2):floor(i+vl/2),ceil(j-hl/2):floor(j+hl/2))));%+sum(x_pad(i-vl:i+vl,j));
                 %Summing Values around current entry according 2 vl and hl
                %y(i-vl,j-hl,s)=y(i-vl,j-hl,s)-x_pad(i,j);%Current Values gets added
                %twice. Once in row and once in Col
                check=numelem-1;
                
            end
        end
    
end