load HighResPVsmallblockfibers.mat;
% Plot all the fibers
N = length(fibers)
NofGroups = zeros(N,1) + 20/255.0;
MeasureAngles = zeros(N,1);
steps = 6 
for i=1:steps:N,
    fiber=fibers{i};
    for j=1:1:(length(fiber)-1),
    x1 = fiber(round(j),1);
    y1 = fiber(round(j),2);
    z1 = fiber(round(j),3);
    x2 = fiber(round(j+1),1);
    y2 = fiber(round(j+1),2);
    z2 = fiber(round(j+1),3);
    x = x2 - x1;
    y = y2 - y1;
    z = z2 - z1;
    %theta = atan2(y,x);
    phi = atan2(z,sqrt(x.^2 + y.^2));
    %r = sqrt(x.^2 + y.^2 + z.^2);
    if (abs(phi) >= pi/4),
        MeasureAngles(i) = (abs(phi) - pi/4)/(pi/4)*0.5+0.5 + MeasureAngles(i);
    else
        MeasureAngles(i) = abs(phi)/(pi/4)*0.5 + MeasureAngles(i);
    end;
    end;
    MeasureAngles(i) = MeasureAngles(i)/(length(fiber)-1);
    %if MeasureAngles(i) >=0.75,
    %    NofGroups(i) = 160.0/255.0;
    %elseif MeasureAngles(i) >=0.25, 
    %   NofGroups(i) = 130.0/255.0; 
    %end;
    NofGroups(i) = MeasureAngles(i); 
end
%indexsmall = find(NofGroups==20/255.0); 
%NofGroups(indexsmall(1:20)) = 1/255.0;
%indexsmall = find(NofGroups==160/255.0);
%NofGroups(indexsmall(1:20)) = 1;
NofGroups(1) = 1/255.0;
NofGroups(steps + 1) = 1;
for i=1:steps:N,
    fiber=fibers{i};
    %h=plot3t(fiber(1:2:end,1),fiber(1:2:end,2),fiber(1:2:end,3),0.2,'r');
    %c0 = 255.0*ones(size(fiber(1:4:end,2)))*i/single(N);
    c0 = 255*NofGroups(i)*ones(size(fiber(1:1:end,2)));
    %c0 = scaleG*GroupedFibres(i)*ones(size(fiber(1:1:end,2)));
    h=cline(fiber(1:1:end,1),fiber(1:1:end,2),fiber(1:1:end,3),c0);
    set(h, 'FaceLighting','phong','SpecularColorReflectance', 1, 'SpecularExponent', 50, 'DiffuseStrength', 1);
end
view(3);
camlight;
material shiny
view(60,10);
light;
lightangle(60,10)
axis off;

daspect([1,1,1])
axis on; box on;

