function  [simg,nsimg]=signalsort(signals,imgrgb,N)

maxorg = max(imgrgb(:));
minorg = min(imgrgb(:));
[width,height]=size(imgrgb(:,:,1));
for j = 1:1: N
    s = signals(j,:);
    smin = min(s);
    smax = max(s);  
    s = (s - smin)/(smax-smin)*(maxorg-minorg)+minorg;   
    maxss = max(s(:));  
    ssimg = reshape(s,width,height);
    ns = maxss - s;
    nnsimg = reshape(ns,width,height);
    if ssim(nnsimg,imgrgb) > ssim(ssimg,imgrgb)
        ssimg = nnsimg;
    end 
    tssim(j) = ssim(ssimg,imgrgb);
%     tssim(j) = psnr(ssimg,imgrgb);
    
end

[~,index] = max(tssim);
%     [A,index] = sort(tssim);
%     find(tssim==A(100));
%     for k=1:N
%     H = signals(index(101-k),:);
H = signals(index,:);
Hmin = min(H);
Hmax = max(H);
H = (H - Hmin)/(Hmax-Hmin)*(maxorg-minorg)+minorg;
maxHH = max(H(:));
simg = reshape(H,width,height);
G = maxHH - H;
nsimg = reshape(G,width,height);

