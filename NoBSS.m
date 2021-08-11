% clc;
%----------------专利最终版-------------%
clear ;
close all;


%%
%% 灰度累加曲线
% [filename, pathname] = uigetfile('*.*','待计算图像1');
% img=im2double(imread([pathname, filename]));
I=imread('D:\Program Files\Polyspace\TU\低照度\iris.png');
colorspace = ' ';
tic
I_orig=im2double(I);
Imean=mean2(I_orig);

if Imean<=0.01
    Imean=mean2(I_orig);
    p1=log(0.4);
    beta=0.02; %0.02
elseif Imean<=0.02
    p1=log(round(Imean*1000)/100+0.3);
    beta=0.02;
else
    p1=log(0.5);%0.53
    beta=0.02;%0.05
end

p=p1/log(Imean); %p=round(10*p1/log(Imean))/10;%  p=0.3;
Img=power(I_orig,p);  %imshow(Img) mean2(Img)
%%
[width,height,m]=size(I_orig);

%----------------------------1-----------------------------------%       
%         p=p1/log(mean2(I_rgb)); 
p=p1/log(sum(median(median(I_orig)))/3);
%         P(j)=p;
I_orig=power(I_orig,p); %I_rgb=log2(1+I_rgb);
%         [I_yuv,~, ~] = rgb_to(I_rgb,colorspace);
I_yuv = rgb_yuv(I_orig,colorspace);  
I_yuv = reshape(I_yuv, [width, height, 3]);

simg=I_yuv(:,:,1);

%%
%增加饱和度
% mean2(round(simg*255)) imshow(simg)
% smean=mean2(simg);
% if smean>0.5
%     simg=imdivide(simg,smean/0.5);  
%     I_yuv=immultiply(I_yuv,3); %r的范围一般为(0,5), 如果为1表示不改变图像数据    
% end

simg12=I_yuv(:,:,2); % max(max(simg12))
simg13=I_yuv(:,:,3);

% Iout=cat(3,simg,simg12,simg13);
% Iout1 = rgb_yuv(Iout,colorspace,true); %转回RGB
% Iout1=reshape(Iout1,[width, height,3]);
% figure,imshow(Iout1)
% 
% F=rgb2hsv(Iout1);
% F(:,:,2)=F(:,:,2)*1.8;
% F(:,:,3)=F(:,:,3)*5/3-1/3;
% Fout=hsv2rgb(F);
% figure,imshow(Fout)

%%
%---------------------------2-----------------------------------%
% figure,imhist(simg)
simg1=adapthisteq(simg,'Distribution','uniform','ClipLimit',0.01); %'uniform'平坦 'rayleigh'钟形 'exponential'曲线
simg1=0.5*simg+0.5*simg1;  %mean2(round(simg1*255))
% simg1=simg;
% figure,
% subplot(121),imshow(simg)
% subplot(122),imshow(simg1)

smean=mean2(simg1);
% p2=log(round(smean*1000)/1000+beta); %0.4 0.5之间
p2=log(smean+beta); 
% p2=log(0.5); 
% if smean>0.4
% %     p2=log(round(smean*100)/100+0.05); %0.4 0.5之间
%     p2=log(round(smean*1000)/1000+beta);
% else
%     p2=log(0.45); %0.4 0.5之间
% end
%%
%---------------------------------------------------------------%
% p3=p2/log(mean2(simg));  
% simg=power(simg,p3);
% % % imshow(simg)

% 高斯滤波
HSIZE= min(width,height);%高斯卷积核尺寸
q=sqrt(2);
SIGMA1=15;%论文里面的c
SIGMA2=80;
SIGMA3=250;
F1 = fspecial('gaussian',HSIZE,SIGMA1/q);
F2 = fspecial('gaussian',HSIZE,SIGMA2/q) ;
F3 = fspecial('gaussian',HSIZE,SIGMA3/q) ;
gaus1= imfilter(simg, F1, 'replicate');
gaus2= imfilter(simg, F2, 'replicate');
gaus3= imfilter(simg, F3, 'replicate');
Iq=(gaus1+gaus2+gaus3)/3;    %多尺度高斯卷积，加权，权重为1/3
% gaus=(gaus*255);
% figure,imshow(gaus,[]);title('gaus光照分量');

% % 引导滤波
% r = 16;
% eps = 0.01^2;
% % Iq = zeros(size(I));
% Iq = guidedfilter(simg,simg, r, eps); %第一个是引导量
% % figure,imshow(Iq)
%% 皮尔模型
%---------------------------3-----------------------------------%
% Imx=max(max(simg)); % mean2(round(simg*255))

Beta1=[0.1605 0.0675 -25.7754];%专利版本
% Beta1=[0.082 0.5949 -21.8727];
for i=1:width
    for j=1:height
        deta=1-simg1(i,j);
%         if deta<=0.05
%             simg1(i,j)=simg1(i,j)-0.05;
%             p1(i,j)=(Imean-gaus(i,j))/Imean; %若光照值高于预设值，则降低原图亮度
        if deta<=0.3
%             simg1(i,j)=simg1(i,j)-0.16; 
%             simg1(i,j)=simg1(i,j)-0.32+0.32./(1+exp(-12*deta.^1));
%             simg1(i,j)=simg1(i,j)-Beta(1)+Beta(2)./(Beta(3)+exp(-Beta(4)*deta.^Beta(5)));
            simg1(i,j)=simg1(i,j)-Beta1(1)./(1+Beta1(2).*exp(-Beta1(3).*deta));
%             Imean(i,j)=0.5-0.5*(Beta(1)-Beta(2)./(Beta(3)+exp(-Beta(4)*deta.^Beta(5))));
%             p1(i,j)=(Imean-gaus(i,j))/Imean;
        end
    end
end
%%
%----------------------------4-----------------------------------%
Iomean=exp(p2); %*ones(width,height);   % mean2(L) mean2(v)  level;  0.5
p3=(Iomean-Iq)./Iomean; %原  %若光照值高于预设值，则降低原图亮度  imshow(Imean)
% gama=(ones(size(p1))*0.5).^p1;  imshow(p1)
% p=p2/log(median(median(simg)))-beta;
p=p2/log(mean2(simg1));
gama=power(p,p3);%根据公式gamma校正处理，论文公式有误 mean2(gama)
simgout=power(simg1,gama);   %mean2(round(simgout*255))
% imshow(gama) imshow(simgout) % imshow(p1)
%%
%-------合成分离去噪图-------%
% imgout=cat(3,Simg1,simg12,simg13);
imgout=cat(3,simgout,simg12,simg13);
imgout1 = rgb_yuv(imgout,colorspace,true); %转回RGB
imgout1=reshape(imgout1,[width, height,3]);

% imgout1(find(imgout1>1))=1;
% imgout1(find(imgout1<0))=0;
imgout1=abs(imgout1);

ymin = min(min(imgout1));
ymax = max(max(imgout1));
% % imgout1 = (imgout1 - ymin)./(ymax-ymin);%*(maxorg-minorg)+minorg; %归一化
imgout1 = 0.95.*(imgout1 - ymin)./(ymax-ymin);%*(maxorg-minorg)+minorg; %归一化
%---------------------------3-----------------------------------%
% p4=p2./log(mean2(imgout1))-0.01; 
% imgout2=power(imgout1,p4);  % mean2(round(imgout2*255))

time=toc

% figure,imshow(imgout1,'border','tight','initialmagnification','fit');
% subplot(221),imshow(I_orig);title('原低照图');% subplot(222),imshow(Img);title('原低照增强图');
% subplot(223),imshow(imgout);title(['yuv合成分离去噪图',method]);% subplot(224),imshow(imgout1);title(['转rgb合成分离去噪图',method]);
img_final=uint8(imgout1*255);

figure,
subplot(221),imshow(I);title('原图');
subplot(222),imshow(Img);title('原低照增强图');
subplot(223),imshow(img_final);title('本算法');
% figure(1),imshowpair(I_orig,imgout1,'montage')
%% 直方图
% %------------------增强图----------------------%
gray = rgb2gray(img_final);
% [count,x]=imhist(gray);
figure(),imhist(gray)
axis tight;
set(gca,'FontSize',18); 
h1=xlabel({'Grayscale'},'FontSize',20,'Rotation',0);%h1=xlabel({'灰度级'},'FontSize',18,'Rotation',0);
h2=ylabel({'Pixel count'},'FontSize',20,'Rotation',90);%h2=ylabel({'像','素','数'},'FontSize',18,'Rotation',0);
set(h1, 'Units', 'Normalized', 'Position', [0.5, -0.12, 0]); % 负值：ylabel 左移，正值： 右移；0：中间
set(h2, 'Units', 'Normalized', 'Position', [-0.05, 0.5, 0]); % 负值：ylabel 左移，正值： 右移；0：中间
set(gca,'LooseInset',get(gca,'TightInset'))
set(gca,'looseInset',[0 0 0 0])

ax = gca;
ax.YAxis.Exponent = 3;

% set(gca,'yTickLabel',num2str(get(gca,'yTick')','%.2f'))
% set(gca,'ytick',get(gca,'yTick')'*2)

% print(gcf, '-djpeg', '-r300', ['D:\学习\小论文\','10.09增强直方图.bmp']);
% %------------------原图----------------------%
gray = rgb2gray(I);
figure(),imhist(gray)
axis tight;
set(gca,'FontSize',18); 
h1=xlabel({'Grayscale'},'FontSize',20,'Rotation',0);%h1=xlabel({'灰度级'},'FontSize',18,'Rotation',0);
h2=ylabel({'Pixel count'},'FontSize',20,'Rotation',90);%h2=ylabel({'像','素','数'},'FontSize',18,'Rotation',0);
set(h1, 'Units', 'Normalized', 'Position', [0.5, -0.12, 0]); % 负值：ylabel 左移，正值： 右移；0：中间
set(h2, 'Units', 'Normalized', 'Position', [-0.06, 0.5, 0]); % 负值：ylabel 左移，正值： 右移；0：中间
set(gca,'LooseInset',get(gca,'TightInset'))
set(gca,'looseInset',[0 0 0 0])

% set(gca,'yTickLabel',num2str(get(gca,'yTick')'/100))
% hAxes = gca;
% hAxes.YRuler.SecondaryLabel.String = '×10^5';

histogram(gray,'BinWidth',5)
% print(gcf, '-djpeg', '-r300', ['D:\学习\小论文\','10.09原图直方图.bmp']);

%%
% quality1=testquality(img_final,0);

% imwrite(imgout1,[path,'yuva合成分离去噪图',num2str(num),'.jpg']);

% img=imread([path,'原图.jpg']); 
ssimval = ssim(img_final,I);

str1=sprintf('m2:%.4f  %.4f',Iomean,(Iomean)*255);  %输出类型为char
str2=sprintf('ssim:%.4f',ssimval);  %输出类型为char
display([str2,'  ',str1])
% % psnr_denoise = psnr(imgout1,img)
% quality_orig=testquality(I,0);%原图
% ssim(I,uint8(round(img*255)))

% imwrite(Img,[path,'原低照增强图','.jpg']);
% psnr_denoise1 = psnr(y_est,Img)

% qualityscore = SSEQ(y_est)
% quality=testquality(uint8(round(Img*255)));
%%

