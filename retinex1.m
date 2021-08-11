function result=retinex1(img,method,a,b,c,icount,step)
%----------------------------retinex低照度图像增强------------------------%
if nargin<6
    a=0.01;b=0.2;c=1; %a=0.008;b=0.3;c=1;
    icount=8;   %icount=4;
    step=2.4;%decide the times  %step=2.2;
%     method = 1;
end
if ~exist('method', 'var') 
    method = 1;
end
I=img+1;
Z=log(I);
Z=Z/max(Z(:));
R=zeros(size(Z));
% R1=zeros(size(Z));
% nn = 3;

switch method
    case 1           %zm算法
        for i=1:3
            [R(:,:,i),L]=zm_retinex_Light1(Z(:,:,i),a,b,c,icount,0);
        end
    case 2        %mccann算法
        for i=1:3
            R(:,:,i)=zm_retinex_mccann992(Z(:,:,i),icount);
        end
    case 3        %Kimmel算法程序1
        for i=1:3
            R(:,:,i)=zm_Kimmel(Z(:,:,i),a,b,icount,4.5);
        end
    case 4        %Kimmel算法程序2
        for i=1:3
            R(:,:,i)=Z(:,:,i)-zm_Devir_retinex(Z(:,:,i),a,b);
        end
end

m=mean2(R);s=std(R(:));
mini=max(m-step*s,min(R(:)));maxi=min(m+step*s,max(R(:)));
range=maxi-mini;
result=(R-mini)/range*0.8;
%result=max(result(:))-result;

