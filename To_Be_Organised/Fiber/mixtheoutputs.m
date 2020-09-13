function [mixedoutput] = mixtheoutputs(kmeanslice,thr_slice)
[row,col]=size(kmeanslice);
for i=1:row
    for j=1:col
        if(thr_slice(i,j)>90 && thr_slice(i,j)<155)
            thr_slice(i,j)=kmeanslice(i,j);
        end
    end
end
mixedoutput=thr_slice;
end

