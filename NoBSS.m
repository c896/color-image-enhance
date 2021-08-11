% clc;
%----------------ר�����հ�-------------%
clear ;
close all;


%%
%% �Ҷ��ۼ�����
% [filename, pathname] = uigetfile('*.*','������ͼ��1');
% img=im2double(imread([pathname, filename]));
I=imread('D:\Program Files\Polyspace\TU\���ն�\iris.png');
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
%���ӱ��Ͷ�
% mean2(round(simg*255)) imshow(simg)
% smean=mean2(simg);
% if smean>0.5
%     simg=imdivide(simg,smean/0.5);  
%     I_yuv=immultiply(I_yuv,3); %r�ķ�Χһ��Ϊ(0,5), ���Ϊ1��ʾ���ı�ͼ������    
% end

simg12=I_yuv(:,:,2); % max(max(simg12))
simg13=I_yuv(:,:,3);

% Iout=cat(3,simg,simg12,simg13);
% Iout1 = rgb_yuv(Iout,colorspace,true); %ת��RGB
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
simg1=adapthisteq(simg,'Distribution','uniform','ClipLimit',0.01); %'uniform'ƽ̹ 'rayleigh'���� 'exponential'����
simg1=0.5*simg+0.5*simg1;  %mean2(round(simg1*255))
% simg1=simg;
% figure,
% subplot(121),imshow(simg)
% subplot(122),imshow(simg1)

smean=mean2(simg1);
% p2=log(round(smean*1000)/1000+beta); %0.4 0.5֮��
p2=log(smean+beta); 
% p2=log(0.5); 
% if smean>0.4
% %     p2=log(round(smean*100)/100+0.05); %0.4 0.5֮��
%     p2=log(round(smean*1000)/1000+beta);
% else
%     p2=log(0.45); %0.4 0.5֮��
% end
%%
%---------------------------------------------------------------%
% p3=p2/log(mean2(simg));  
% simg=power(simg,p3);
% % % imshow(simg)

% ��˹�˲�
HSIZE= min(width,height);%��˹����˳ߴ�
q=sqrt(2);
SIGMA1=15;%���������c
SIGMA2=80;
SIGMA3=250;
F1 = fspecial('gaussian',HSIZE,SIGMA1/q);
F2 = fspecial('gaussian',HSIZE,SIGMA2/q) ;
F3 = fspecial('gaussian',HSIZE,SIGMA3/q) ;
gaus1= imfilter(simg, F1, 'replicate');
gaus2= imfilter(simg, F2, 'replicate');
gaus3= imfilter(simg, F3, 'replicate');
Iq=(gaus1+gaus2+gaus3)/3;    %��߶ȸ�˹�������Ȩ��Ȩ��Ϊ1/3
% gaus=(gaus*255);
% figure,imshow(gaus,[]);title('gaus���շ���');

% % �����˲�
% r = 16;
% eps = 0.01^2;
% % Iq = zeros(size(I));
% Iq = guidedfilter(simg,simg, r, eps); %��һ����������
% % figure,imshow(Iq)
%% Ƥ��ģ��
%---------------------------3-----------------------------------%
% Imx=max(max(simg)); % mean2(round(simg*255))

Beta1=[0.1605 0.0675 -25.7754];%ר���汾
% Beta1=[0.082 0.5949 -21.8727];
for i=1:width
    for j=1:height
        deta=1-simg1(i,j);
%         if deta<=0.05
%             simg1(i,j)=simg1(i,j)-0.05;
%             p1(i,j)=(Imean-gaus(i,j))/Imean; %������ֵ����Ԥ��ֵ���򽵵�ԭͼ����
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
p3=(Iomean-Iq)./Iomean; %ԭ  %������ֵ����Ԥ��ֵ���򽵵�ԭͼ����  imshow(Imean)
% gama=(ones(size(p1))*0.5).^p1;  imshow(p1)
% p=p2/log(median(median(simg)))-beta;
p=p2/log(mean2(simg1));
gama=power(p,p3);%���ݹ�ʽgammaУ���������Ĺ�ʽ���� mean2(gama)
simgout=power(simg1,gama);   %mean2(round(simgout*255))
% imshow(gama) imshow(simgout) % imshow(p1)
%%
%-------�ϳɷ���ȥ��ͼ-------%
% imgout=cat(3,Simg1,simg12,simg13);
imgout=cat(3,simgout,simg12,simg13);
imgout1 = rgb_yuv(imgout,colorspace,true); %ת��RGB
imgout1=reshape(imgout1,[width, height,3]);

% imgout1(find(imgout1>1))=1;
% imgout1(find(imgout1<0))=0;
imgout1=abs(imgout1);

ymin = min(min(imgout1));
ymax = max(max(imgout1));
% % imgout1 = (imgout1 - ymin)./(ymax-ymin);%*(maxorg-minorg)+minorg; %��һ��
imgout1 = 0.95.*(imgout1 - ymin)./(ymax-ymin);%*(maxorg-minorg)+minorg; %��һ��
%---------------------------3-----------------------------------%
% p4=p2./log(mean2(imgout1))-0.01; 
% imgout2=power(imgout1,p4);  % mean2(round(imgout2*255))

time=toc

% figure,imshow(imgout1,'border','tight','initialmagnification','fit');
% subplot(221),imshow(I_orig);title('ԭ����ͼ');% subplot(222),imshow(Img);title('ԭ������ǿͼ');
% subplot(223),imshow(imgout);title(['yuv�ϳɷ���ȥ��ͼ',method]);% subplot(224),imshow(imgout1);title(['תrgb�ϳɷ���ȥ��ͼ',method]);
img_final=uint8(imgout1*255);

figure,
subplot(221),imshow(I);title('ԭͼ');
subplot(222),imshow(Img);title('ԭ������ǿͼ');
subplot(223),imshow(img_final);title('���㷨');
% figure(1),imshowpair(I_orig,imgout1,'montage')
%% ֱ��ͼ
% %------------------��ǿͼ----------------------%
gray = rgb2gray(img_final);
% [count,x]=imhist(gray);
figure(),imhist(gray)
axis tight;
set(gca,'FontSize',18); 
h1=xlabel({'Grayscale'},'FontSize',20,'Rotation',0);%h1=xlabel({'�Ҷȼ�'},'FontSize',18,'Rotation',0);
h2=ylabel({'Pixel count'},'FontSize',20,'Rotation',90);%h2=ylabel({'��','��','��'},'FontSize',18,'Rotation',0);
set(h1, 'Units', 'Normalized', 'Position', [0.5, -0.12, 0]); % ��ֵ��ylabel ���ƣ���ֵ�� ���ƣ�0���м�
set(h2, 'Units', 'Normalized', 'Position', [-0.05, 0.5, 0]); % ��ֵ��ylabel ���ƣ���ֵ�� ���ƣ�0���м�
set(gca,'LooseInset',get(gca,'TightInset'))
set(gca,'looseInset',[0 0 0 0])

ax = gca;
ax.YAxis.Exponent = 3;

% set(gca,'yTickLabel',num2str(get(gca,'yTick')','%.2f'))
% set(gca,'ytick',get(gca,'yTick')'*2)

% print(gcf, '-djpeg', '-r300', ['D:\ѧϰ\С����\','10.09��ǿֱ��ͼ.bmp']);
% %------------------ԭͼ----------------------%
gray = rgb2gray(I);
figure(),imhist(gray)
axis tight;
set(gca,'FontSize',18); 
h1=xlabel({'Grayscale'},'FontSize',20,'Rotation',0);%h1=xlabel({'�Ҷȼ�'},'FontSize',18,'Rotation',0);
h2=ylabel({'Pixel count'},'FontSize',20,'Rotation',90);%h2=ylabel({'��','��','��'},'FontSize',18,'Rotation',0);
set(h1, 'Units', 'Normalized', 'Position', [0.5, -0.12, 0]); % ��ֵ��ylabel ���ƣ���ֵ�� ���ƣ�0���м�
set(h2, 'Units', 'Normalized', 'Position', [-0.06, 0.5, 0]); % ��ֵ��ylabel ���ƣ���ֵ�� ���ƣ�0���м�
set(gca,'LooseInset',get(gca,'TightInset'))
set(gca,'looseInset',[0 0 0 0])

% set(gca,'yTickLabel',num2str(get(gca,'yTick')'/100))
% hAxes = gca;
% hAxes.YRuler.SecondaryLabel.String = '��10^5';

histogram(gray,'BinWidth',5)
% print(gcf, '-djpeg', '-r300', ['D:\ѧϰ\С����\','10.09ԭͼֱ��ͼ.bmp']);

%%
% quality1=testquality(img_final,0);

% imwrite(imgout1,[path,'yuva�ϳɷ���ȥ��ͼ',num2str(num),'.jpg']);

% img=imread([path,'ԭͼ.jpg']); 
ssimval = ssim(img_final,I);

str1=sprintf('m2:%.4f  %.4f',Iomean,(Iomean)*255);  %�������Ϊchar
str2=sprintf('ssim:%.4f',ssimval);  %�������Ϊchar
display([str2,'  ',str1])
% % psnr_denoise = psnr(imgout1,img)
% quality_orig=testquality(I,0);%ԭͼ
% ssim(I,uint8(round(img*255)))

% imwrite(Img,[path,'ԭ������ǿͼ','.jpg']);
% psnr_denoise1 = psnr(y_est,Img)

% qualityscore = SSEQ(y_est)
% quality=testquality(uint8(round(Img*255)));
%%

