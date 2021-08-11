function  [maxorg,minorg,index]=signalsort3(signals,WH,N,slectindex,r)
%% 直接对simg进行排序
% signals=signals1;  
Nn=size(signals,1);
% [width,height]=size(imgrgb);
width=WH(1);
height=WH(2);
medw=round(width/2);
medh=round(height/2);
if Nn<N
    N=Nn;
end
% if ~exist('r','var')
%     %     r=0.2; %放缩尺度 p为1时为全局评价
% %     r='[1:10 width-9:width],[1:10 height-90:height]';
% end
% maxorg = max(imgrgb(:));
% minorg = min(imgrgb(:));
maxorg = 1;
minorg = 0;
l=r; %局部尺寸
l1=fix(l/2-1);
l2=fix(l/2);
% maxorg = max(imgrgb(:))/2+0.5;
% minorg = min(imgrgb(:))/2;

tstd=zeros(1,N);
% tssim=zeros(1,N);
if exist('r','var')&&r<=1   %numel(r)==1
    %     if r==1
    %         imgrgb1 = imgrgb;
    %     elseif r<1
    %         imgrgb1 = imresize(imgrgb,r);
    %     end
    for j = 1:1:N
        s = signals(j,:);
        smin = min(s);
        smax = max(s);
        s = (s - smin)./(smax-smin)*(maxorg-minorg)+minorg; % 灰度范围变换
%         maxss = max(s(:));
        ssimg = reshape(s,width,height);
%         ns = maxss - s;
%         nnsimg = reshape(ns,width,height);
        %------------尺度放缩评价-------------%
%         nnsimg1=imresize(nnsimg,r);
        ssimg1=imresize(ssimg,r);
        tstd(j) = std2(ssimg1);
%         tstd1 = std2(ssimg1);
%         tstd2 = std2(nnsimg1);
                
%         if tstd1>tstd2
%             tstd(j)=tstd1;
%         else
%             tstd(j)=tstd2;
%         end
        % %------------全局评价-------------%
        %     if ssim(nnsimg,imgrgb) > ssim(ssimg,imgrgb)
        %         ssimg = nnsimg;
        %     end
        % %     tssim(j) = ssim(ssimg,imgrgb);
        %     tssim(j) = psnr(ssimg,imgrgb);
    end
    
else     %if exist('r','var')&&r>1
    %     imgrgb1=imgrgb([1:10 medw-4:medw+5 width-9:width],[1:10 medh-4:medh+5 height-9:height]);
    %     imgrgb1=imgrgb([1:l medw-l1:medw+l2 width-l+1:width],[1:l medh-l1:medh+l2 height-l+1:height]);
    
    for j = 1:1:N
        s = signals(j,:);
        smin = min(s);
        smax = max(s);
        s = (s - smin)./(smax-smin)*(maxorg-minorg)+minorg; % 灰度范围变换
        maxss = max(s(:));
        ssimg = reshape(s,width,height);
%         ns = maxss - s;
%         nnsimg = reshape(ns,width,height);
        %-----------局部评价---------------%
%         nnsimg1=nnsimg([1:l medw-l1:medw+l2 width-l+1:width],[1:l medh-l1:medh+l2 height-l+1:height]);
        ssimg1=ssimg([1:l medw-l1:medw+l2 width-l+1:width],[1:l medh-l1:medh+l2 height-l+1:height]);
        tstd(j) = std2(ssimg1);
%         tstd1 = std2(ssimg1);
%         tstd2 = std2(nnsimg1);
%         if tstd1>tstd2
%             tstd(j)=tstd1;
%         else
%             tstd(j)=tstd2;
%         end
    end
end

if ~exist('slectindex','var')
    slectindex='max'; %% default profile
end

switch slectindex
    case 'max'
        [~,index]=find(tstd==max(tstd));
    case 'sort'
        [~,index] = sort(tstd);
end
% [~,index] = max(tssim);
% [~,index] = sort(tssim);
% [A,index] = sort(tssim);
%     find(tssim==A(100));
%     for k=1:N
%     H = signals(index(101-k),:);

