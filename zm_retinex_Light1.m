function [R,L]=zm_retinex_Light1(S,a,b,c,icount,method,maxi)
if nargin<7
    maxi=max(S(:));
    if nargin<6
        method=1;
    end
    if nargin<5
        icount=4;
    end
end

if numel(S)<25
    L=maxi*ones(size(S));
else
    [h,w]=size(S);%s is input
    Y=[S S(:,end)];Y=[Y ;Y(end,:)];%enlare at right and bottom 
    Sf=blkproc(Y,[2 2],@mean2);% get the mean of 2*2 matrix to reduce the scale,to get the reflection
    %分成多个2*2的块对其平均（两次列平均）
%     blockproc
    %Sf=blkproc(Y,[2 2],@(x) max(x(:)));
    Sf=Sf(1:ceil(h/2),1:ceil(w/2));% get interger greater than now ceil向上取整
    Lf=Sf-zm_retinex_Light1(Sf,a,b,c,icount,method,maxi);
    Y(1:2:h,1:2:w)=Lf;
    Y(2:2:h+1,1:2:w)=Lf;
    Y(1:2:h,2:2:w+1)=Lf;
    Y(2:2:h+1,2:2:w+1)=Lf;
    L=max(Y(1:h,1:w),S);%reconstruct
end

Th=0.01;
Z=[S(:,1) S S(:,end)];Z=[Z(1,:); Z ;Z(end,:)];
Z=conv2(Z,[-1 -1 -1;-1 8 -1;-1 -1 -1],'valid');%Z=max(abs(Z),Th).*sign(Z);
Z=a*S+(b-c)*Z;
for i=1:icount
    if method==0
        Y=[L(:,1) L L(:,end)];Y=[Y(1,:); Y ;Y(end,:)];
        Y=conv2(Y,[1 1 1;1 0 1;1 1 1],'valid')*(1+b);
        L=max(((Y+Z)/(8+a+8*b)+L)/2,S);
    else
        [h,w]=size(L);
        oh=[1 2 1 2];ow=[1 2 2 1];
        for n=1:4
            hh=oh(n):2:h;
            ww=ow(n):2:w;
            Y=[L(:,1) L L(:,end)];Y=[Y(1,:); Y ;Y(end,:)];
            tmp=(Y(hh,ww)+Y(hh,ww+1)+Y(hh,ww+2)+Y(hh+1,ww)+Y(hh+1,ww+2)+Y(hh+2,ww)+Y(hh+2,ww+1)+Y(hh+2,ww+2))*(1+b);
            L(hh,ww)=max((tmp+Z(hh,ww))/(8+a+8*b),S(hh,ww));%get illumination factors
        end
    end
end
R=S-L;
