% edge tracking
clc;clear;
%cd ua_newalgo_sobel; im=imread('sobel_00560.png'); cd ..;
cd d_images2dcoarsesegment_s4; im=imread('s4_00560.png'); cd ..;
im1=im(1:1300,2051:3550,1);
[gr,ang1]=imgradient(im1);
x11=598;%986;%741;%1277;%1197;%1033;%995;%;%508;%550;%448;%331;%309;%233;%330;%435;%;%571;%1157;%1293;%1015;
y11=411;%78;%459;%391;%157;%77;%175;%403;%409;%490;%814;%869;%887;%889;%993;%1085;%1096;%1114;%677;%422;%145;
k=0;
input=[0 0];
for i=1:500
    x1(i)=x11;y1(i)=y11;   % Now v r @ point(x11,y11)
    
    actual_angle=ang1(y11,x11);
    angle=PredictedAngle(gr,x11,y11,actual_angle,3,5);%ang1(y11,x11);%(atan2(ux(i),uy(i))+pi/2)*(180/pi);    % Angle at (x11,y11)
    if(i>1)
        diff=abs(angle-allangles(i-1));
    else diff=0;
    end
    if(diff<90)
        allangles(i)=angle;
    elseif(abs(angle-actual_angle)<20)
        allangles(i)= actual_angle;
        diff=0;
    else allangles(i)= allangles(i-1);  
    end
    dist=3;
    [x2,y2]=LocationAfterMoveAtAngle(x1(i),y1(i),allangles(i),dist);
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
    B=[x1(i) y1(i)];
    Lia = ismember(input,B,'rows');
    
    grin(i)=gr(y1(i),x1(i)); % check the gradient at current location (if gr>10)and add currrent location (not next location) to final mask input table
    if (grin(i)>10)&&(abs(y11-y1(i))<20)&&(abs(x11-x1(i))<20)&&(sum(Lia)==0)&&(diff<90) % make sure that x and y do not change drastically in one step
        k=k+1;
        input(k,1)=x1(i);
        input(k,2)=y1(i);
    else
        break;
    end
end
figure,imagesc(gr);hold on;
plot(input(:,1),input(:,2),'w-');hold on;