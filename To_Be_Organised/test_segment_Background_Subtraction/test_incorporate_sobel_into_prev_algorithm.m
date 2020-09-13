%% Mark and save new object (Instruction : Run this section, Mark the points, Then Run the next section(save))

clear all;close all;clc;

type='i';Im_no=560;obj_no=1;  % CHANGE ONLY THIS LINE 
% type='o' means outside 
% type='i' means inside
% type='e' means extra(like islands)


Im_name=sprintf('sobel_%05d.png',Im_no);
cd ua_newalgo_sobel;imgg=imread(Im_name);cd ..;
figure,imagesc(i);title([' Sobel output of Image no.  ',num2str(Im_no) ,' (Mark co-ordinates and save)']);
%% save
[x,y]=getpts;
c=[x,y];
c_name=sprintf('c_%05d%c_%02d_incorporate_Sobel.mat',Im_no,type,obj_no);
save(c_name,'c');

%% Viewing A, ux and uy
 
for i=560:660
    imname=sprintf('s4_%05d.png',i);outname=sprintf('sobel_%05d.png',i);
cd d_images2dcoarsesegment_s4;im=imread(imname); cd ..;
im1=im(1:1300,2051:3550,:);
[A]=sobel_color(im1);
cd ua_newalgo_sobel;imwrite(uint8(A),outname);cd ..;

end

% A1=squeeze(A(:,:,1));A2=squeeze(A(:,:,2));A3=squeeze(A(:,:,3));
% ux1=squeeze(ux(:,:,1));ux2=squeeze(ux(:,:,2));ux3=squeeze(ux(:,:,3));
% uy1=squeeze(uy(:,:,1));uy2=squeeze(uy(:,:,2));uy3=squeeze(uy(:,:,3));
%  figure,imagesc(uint8(A)),title('sobel output');
% figure,imagesc(uint8(ux1)),title('ux-red channel');
% figure,imagesc(uint8(uy1)),title('uy-red channel');
% figure,imagesc(uint8(ux2)),title('ux-green channel');
% figure,imagesc(uint8(uy2)),title('uy-green channel');
% figure,imagesc(uint8(ux3)),title('ux-blue channel');
% figure,imagesc(uint8(uy3)),title('uy-blue channel');

%%
clear;clc;close all;
Im_name=sprintf('sobel_%05d.png',560);
cd ua_newalgo_sobel;imgg=imread(Im_name);cd ..;
    %coord_name=sprintf('c_%05do_%02d.mat',c_no,obj_no);
    c_name=importdata('c_00560i_01_incorporate_Sobel.mat');
    c=round(c_name);% round the co-ordinates
    l=length(c);
    final_c=c(1,:); % first row of co-ordinates
    figure,imagesc(imgg);hold on;plot(c(:,1),c(:,2),'r*');title('Red stars indicate initially selected points (24 points)'); colormap(jet);
% Do interpolation between two adjacent points in the co-ordinate set, for all the 276 points. Result is 15349  points
    % #####################################################################################################################
    for i=1:l-1
        xdiff=abs(c(i,1)-c(i+1,1)); % hor. difference between adjacent points 
        ydiff=abs(c(i,2)-c(i+1,2)); % ver. difference between adjacent points
        if(xdiff==0)
        xx1=ones(1,ydiff+1);
        xx = c(i,1)*xx1;
            if((c(i,2)-c(i+1,2))<0)
             yy = [c(i,2):c(i+1,2)];
            elseif((c(i,2)-c(i+1,2))>0)
             yy = [c(i,2):-1:c(i+1,2)];
            end
        elseif(ydiff==0)
        yy1=ones(1,xdiff+1);
        yy = c(i,2)*yy1;        
            if((c(i,1)-c(i+1,1))<0)
             xx = [c(i,1):c(i+1,1)];
            elseif((c(i,1)-c(i+1,1))>0)
             xx = [c(i,1):-1:c(i+1,1)];
            end
        elseif(xdiff>ydiff)   % do vertical interpolation (predict y values for given x values)
        x = [c(i,1),c(i+1,1)];
        y = [c(i,2),c(i+1,2)];
        xx =linspace(x(1),x(2));
        yy = spline(x,y,xx);
        else                 % do horizontal interpolation (predict x values for given y values)
        x = [c(i,1),c(i+1,1)];
        y = [c(i,2),c(i+1,2)];
        yy =linspace(y(1),y(2));
        xx = spline(y,x,yy);
        end
        xxt=xx';yyt=yy';
        xxr=round(xxt);yyr=round(yyt);
        xtra_c=[xxr,yyr]; % xtra_c contains the extra co-ordinates derived by interpolation
        unique_xtra_c=unique(xtra_c,'rows','stable'); % Remove duplicate entries (resulting from rounding of xtra_c in previous step) and stay stable (keep up the co-ordinates order)
        final_c=[final_c;unique_xtra_c(2:end,:)]; % Append the unique_xtra_c to final_c [ final_c columns 1 and 2 : finalised co-ordinates (32579 points) to draw the approximate boundary
        clear unique_xtra_c;clear xx;clear yy;
    end
    
    final_c=unique(final_c,'rows','stable'); % Remove duplicate entries, if any

    figure,imagesc(imgg);hold on;plot(c(:,1),c(:,2),'r*-',final_c(:,1),final_c(:,2),'g-');title('Green line indicate approx. outer boundary plotted with the interpolated points(a few thousand points)');colormap(jet);

% Make the binary mask using the approximate boundary, take its gradient, and use the gradient directions (to decide the direction to move)
% #########################################################################################################################################

bw1=poly2mask(final_c(1:end,1),final_c(1:end,2),2490,4140);

[bw1g,bw1dr]=imgradient(bw1); % gradient of binary mask
bw1g(bw1g~=0)=1; % set all non-zero gradient magnitude values to 1(Binarising)
bw1dr(bw1g==0)=350;  % set background to 350

range=100; % Limiting the maximum range to move (100 pixels)
% Setting the flags(column 4 of final_c), based on Angles of approximate boundary(column 3). Flags will predict the direction of movement
% #######################################################################################################################################
cr=60;
final_c(:,3)=diag(bw1dr(final_c(:,2),final_c(:,1)));
for j=1:length(final_c)
    oldx=final_c(j,1);
    oldy=final_c(j,2);
    angle=final_c(j,3);
    for r=1:range
        inter(r,1)=round(r*sin((90-angle)*pi/180)); % horizontal distance to move
        inter(r,2)=round(r*cos((90-angle)*pi/180)); % vertical distance to move
        inter(r,3)=oldx-inter(r,1);                         % Next X-co-ordinate
        inter(r,4)=oldy+inter(r,2);                         % Next Y-co-ordinate
        
       if((inter(r,4)<1)||(inter(r,3)<1))
           break;
       end
        inter(r,5)=imgg(inter(r,4),inter(r,3));
        
    end
    set=find(inter(:,5)>cr);
    if(~isempty(set))
        add(j)=min(set);
        final_c(j,4)=inter(add(j),3);
        final_c(j,5)=inter(add(j),4);
    else
        add(j)=0;
        final_c(j,4)=oldx;
        final_c(j,5)=oldy;
    end
        final_c(j,6)= (final_c(j,1)==final_c(j,4))&(final_c(j,2)==final_c(j,5));% condition to Remove all points which have not changed at all
end

figure, imagesc(imgg);hold on;plot(final_c(1:end,4),final_c(1:end,5),'g*'); title('Co-ordinates-Before Filtering- All points ')
add=add';
div=80;
rs=reshape(add(1:div*(floor(length(add)/div))),[div,floor(length(add)/div)]);
meann=mean(rs);
stdd=std(rs);
I = bsxfun(@lt, meann + 2*stdd, rs) | bsxfun(@gt, meann - 2*stdd, rs);
rs_end=add(div*(floor(length(add)/div))+1:end);
meann_end=mean(rs_end);stdd_end=std(rs_end);
I_end = bsxfun(@lt, meann_end + 2*stdd_end, rs_end) | bsxfun(@gt, meann_end - 2*stdd_end, rs_end);
I_col=[I(:);I_end];  % condition to Remove all outlier points 
%Removing all points which have not changed at all or those belonging to
%outliers
%#################################################
adjusted_c=[final_c(:,4:6),I_col];
cond=(adjusted_c(:,3)==1)|(adjusted_c(:,4)==1);
adjusted_c(cond,:)=[];


figure, imagesc(imgg);hold on;plot(adjusted_c(1:end,1),adjusted_c(1:end,2),'g*'); title('Adjusted co-ordinates- All points- ')% plotting every 60th point from the adjusted  co-ordinates (adjusted_c)
    
 % Polnomial fit

l=length(adjusted_c);
step=11;
innerstep=1;
deg=2;
storex=0;storey=0;storex1=0;storey1=0;
for i=0:floor(l/step)-1
    % adjusted_c will not be an exact multiple of step always. So the polyfit
    % should be done in two steps 
    
    % Step 1 - for the co-ordinates till the highest possible perfect multiple 
    x=[adjusted_c((i*step+1):innerstep:(i+1)*step,1)];
    y=[adjusted_c((i*step+1):innerstep:(i+1)*step,2)];
    xdiff=abs(x(1)-x(end)); % hor. difference between points separated by step
    ydiff=abs(y(1)-y(end)); % ver. difference between points separated by step
    if(xdiff>ydiff)
        p=polyfit(x,y,deg);
        x1=linspace(x(1),x(end),1000);
        y1 = polyval(p,x1);
    else
     p=polyfit(y,x,deg);
    y1=linspace(y(1),y(end),1000);
    x1 = polyval(p,y1);
    end

    storex=[storex;x];storey=[storey;y];
    storex1=[storex1;x1'];storey1=[storey1;y1'];
end
    %   Step 2- for the remaining co-ordinates 
if((l/step)-(floor(l/step))~=0)    
x=[adjusted_c((step*floor(l/step)+1):end,1)];
y=[adjusted_c((step*floor(l/step)+1):end,2)];
xdiff=abs(x(1)-x(end)); % hor. difference between points separated by step
ydiff=abs(y(1)-y(end)); % ver. difference between points separated by step
if(xdiff>ydiff)
     p=polyfit(x,y,deg);
    x1=linspace(x(1),x(end),1000);
    y1 = polyval(p,x1);
else    
    p=polyfit(y,x,deg);
    y1=linspace(y(1),y(end),1000);
    x1 = polyval(p,y1);
end    
storex=[storex;x];storey=[storey;y];
storex1=[storex1;x1'];storey1=[storey1;y1'];
end

storex=storex(2:end);storey=storey(2:end);
storex1=storex1(2:end);storey1=storey1(2:end);
figure,imagesc(imgg);hold on;plot(storex1,storey1,'g-');title('The segmented boundary(green line) overlaid on sobel');
clc;bw=poly2mask(storex1,storey1,1300,1500);
figure,imagesc(bw);title('Final binary mask');
imgg1=imgg(:,:,1);imgg2=imgg(:,:,2);imgg3=imgg(:,:,3);
imgg_sum=imgg1+imgg2+imgg3;
imgg1(bw==1 | imgg_sum<60)=0;imgg2(bw==1| imgg_sum<60)=0;imgg3(bw==1| imgg_sum<60)=0;
out(:,:,1)=imgg1;out(:,:,2)=imgg2;out(:,:,3)=imgg3;
figure,imagesc(out)
bw(imgg1==0 & imgg2==0 & imgg3==0)=255;
figure,imagesc(bw);
bw2=bwareaopen(bw,500);
figure,imagesc(bw2);

%%
se = strel('disk',5);
bw3=imclose(bw2,se);

cd d_images2dcoarsesegment_s4;im=imread('s4_00560.png'); cd ..;
im1=im(1:1300,2051:3550,1);im2=im(1:1300,2051:3550,2);im3=im(1:1300,2051:3550,3);
im1(bw3==1)=255;im2(bw3==1)=255;im3(bw3==1)=255;
out2(:,:,1)=im1;out2(:,:,2)=im2;out2(:,:,3)=im3;
figure,imagesc(out2)    

%%
clear;clc;close all;
Im_name=sprintf('sobel_%05d.png',560);thresh=100;
cd ua_newalgo_sobel;imgg=imread(Im_name);cd ..;
imgg1=imgg(:,:,1);imgg2=imgg(:,:,2);imgg3=imgg(:,:,3);
imgg_sum=imgg1+imgg2+imgg3;
imgg1(imgg_sum<thresh)=255;imgg2(imgg_sum<thresh)=255;imgg3(imgg_sum<thresh)=255;
out(:,:,1)=imgg1;out(:,:,2)=imgg2;out(:,:,3)=imgg3;
figure,imagesc(out);
bw=zeros(1330,1500);
bw(imgg1==255 & imgg2==255 & imgg3==255)=1;
figure,imagesc(bw);

cd d_images2dcoarsesegment_s4;im=imread('s4_00560.png'); cd ..;
im1=im(1:1300,2051:3550,1);im2=im(1:1300,2051:3550,2);im3=im(1:1300,2051:3550,3);
im1(bw==1)=255;im2(bw==1)=255;im3(bw==1)=255;
out2(:,:,1)=im1;out2(:,:,2)=im2;out2(:,:,3)=im3;
figure,imagesc(out2)    


%% edge tracking
clc;clear;
%cd ua_newalgo_sobel; im=imread('sobel_00560.png'); cd ..;
cd d_images2dcoarsesegment_s4; im=imread('s4_00560.png'); cd ..;
im1=im(1:1300,2051:3550,1);
[gr,ang1]=imgradient(im1);
x11=564;%1277;%1197;%1033;%995;%;%508;%550;%448;%331;%309;%233;%330;%435;%;%571;%1157;%1293;%1015;
y11=409;%391;%157;%77;%175;%403;%409;%490;%814;%869;%887;%889;%993;%1085;%1096;%1114;%677;%422;%145;
k=0;
for i=1:1000
x1(i)=x11;y1(i)=y11;   % Now v r @ point(x11,y11)
%allangles(i)=ang1(y11,x11);
actual_angle=ang1(y11,x11);
angle=PredictedAngle(gr,x11,y11,actual_angle,3,20);%ang1(y11,x11);%(atan2(ux(i),uy(i))+pi/2)*(180/pi);    % Angle at (x11,y11)
dist=3;
[x2,y2]=LocationAfterMoveAtAngle(x1(i),y1(i),angle,dist);
ys=[-1 -1 -1;-0 0 0;1 1 1];         % to find neighbours of (x2,y2) : y coordinates
newy2=y2+ys;
xs=[-1 0 1;-1 0 1;-1 0 1];          % to find neighbours of (x2,y2) : x coordinates
newx2=x2+xs;
coords=[newy2(:,1) newx2(:,1);newy2(:,2) newx2(:,2);newy2(:,3) newx2(:,3)]; % Making the matrix to look like the co-ordinate locations
values=diag(gr(coords(:,1),coords(:,2)));    % find corresponding gradient values
newcoords=coords(max(find(values==max(values))),:); % find locations in 3x3 neighbourhood with max gradient. 
                      %There may be more than one  locations. In that case , choose the last in the coords table(This is doubtful. may have to change later ?)
y11=newcoords(1,1); % Set the new co-ordinates to act as initial point for next iteration
x11=newcoords(1,2);
grin(i)=gr(y1(i),x1(i)); % check the gradient at current location (if gr>10)and add currrent location (not next location) to final mask input table
if (grin(i)>10)&&(abs(y11-y1(i))<20)&&(abs(x11-x1(i))<20) % make sure that x and y do not change drastically in one step
    k=k+1;
    input(k,1)=x1(i);
    input(k,2)=y1(i);
else break;
end
end
figure,imagesc(gr);hold on;
plot(input(:,1),input(:,2),'w-');hold on;



%%
clear;clc;close all;
cd d_images2dcoarsesegment_s4; im=imread('s4_00560.png'); cd ..;thresh=6;
im1=im(1:1300,2051:3550,1);im2=im(1:1300,2051:3550,2);im3=im(1:1300,2051:3550,3);
[gr1,ang1]=imgradient(im1);[gr2,ang2]=imgradient(im2);[gr3,ang3]=imgradient(im3);
out(:,:,1)=gr1;out(:,:,2)=gr2;out(:,:,3)=gr3;
ang(:,:,1)=ang1;ang(:,:,2)=ang2;ang(:,:,3)=ang3;
figure,imagesc(uint8(out));
anghalf=ang;
[y,x,z]=size(anghalf);
for k=1:z
for i=1:y
    for j=1:x
        if(anghalf(i,j,k)<0)
            anghalf(i,j,k)=180+ang(i,j,k);
        end
    end
end
end
anghalf1=anghalf(:,:,1);
anghalf2=anghalf(:,:,2);
anghalf3=anghalf(:,:,3);
diff1=anghalf1-anghalf2;
diff2=anghalf2-anghalf3;
diff3=anghalf3-anghalf1;
diff=abs(diff1)+abs(diff2);%+abs(diff3);
bin=zeros(y,x);
bin(diff<thresh)=255;
figure,imagesc(bin);colormap(gray);
ang1=ang(:,:,1);
ang2=ang(:,:,2);
ang3=ang(:,:,3);
bin1=zeros(y,x);bin1((ang1+ang2+ang3)<100)=255;
figure,imagesc(uint8(ang));
figure,imagesc(ang1);
figure,imagesc(uint8(anghalf));
figure,imagesc(medfilt2(bin,[2 2]));colormap(gray);
outun8=uint8(out);
imwrite(outun8,'sobel_00560_for_directionality.png');

%%

cd d_images2dcoarsesegment_s4; im=imread('s4_00560.png'); cd ..;
im1=im(1:1300,2051:3550,1);im2=im(1:1300,2051:3550,2);im3=im(1:1300,2051:3550,3);
[gr1,ang1]=imgradient(im1);[gr2,ang2]=imgradient(im2);[gr3,ang3]=imgradient(im3);
ang(:,:,1)=ang1;ang(:,:,2)=ang2;ang(:,:,3)=ang3;
angu=uint8(ang);
figure,imagesc(angu);
