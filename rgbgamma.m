%------------RGB�Ľ��汾---------------%
% clc;
clear;
close all;

% DIR='D:\Cammera\7.02\1\480-640\';
% path='D:\Cammera\7.02\rgbresult1\';

% DIR='D:\Cammera\7.17\7.17-480-640\';
% path='D:\Cammera\7.17\7.17result\';

DIR='D:\Cammera\9.28\9.28-2-480-640\';
path='D:\Cammera\9.28\9.28-2result\';

imgFiles = dir([DIR,'*.jpg']);%����ͼ��ĸ�ʽ  dir('')�г�ָ��Ŀ¼���������ļ��к��ļ�
tic;
% I1 = im2double(imread('Lena512.png'));
% I2 = im2double(imread('Baboon512.png'));
I= imread([DIR , imgFiles(1).name]);
I_orig = im2double(I);
Imean=mean2(I_orig );
% subplot(122),imshow(I2);title('��������ͼ2');
[~,~,m]=size(I_orig);
[N, ~]= size(imgFiles);
[width,height]=size(I_orig(:,:,1));

num=25;%������

imgnum=200/num;%֡������

r=0.2;%�ߴ�����任ϵ��
% r=[50:100,50:100];%�ֲ��ߴ�
N2=N/imgnum;
%number1��Ҫ�������ͼ������
%number1��Ҫ�����ͼ�������
method='power';

% p1=log(90/255);
% p2=p1;
if Imean<=4/255
    p1=log(40/255); 
    p2=log(60/255);
    beta=0.005;
elseif Imean<=9/255
%     p1=log(Imean*10); %10��
    p1=log(60/255); 
%     p2=log(ceil(255*Imean)/25.5);
    p2=p1;
    beta=0.008;
elseif Imean<=15/255
    p1=log(90/255);
    p2=p1;
    beta=0.08;
else 
    p1=log(90/255);
    p2=log(120/255);
    beta=0.1;
end

switch method
    case 'power'
%         P=[0:5:20 35 60 75 90 100];
%         p = discretize(Imean,[-Inf P Inf]);
%         p=p+abs(P(p-1)-Imean)/(P(p)-P(p-1));
%         p=round(p/10,2); %����2λС��
        p=p1/log(Imean);
%         p=0.4;%٤����ǿָ��
        I1=power(I_orig,p);
%         path='D:\Cammera\7.17result\';
%         path='D:\Cammera\7.2\rgbresult1\';
    case 'retinex'
        I1=retinex1(I_orig);
        path='G:\��Ƭ\6.17\rgbresult2\';
    case 'other'
        I1=I_orig;
end
% figure,imshow(I1)

Ic=zeros(width,height,3);
if strcmp(method,'power')
%     for i=1:m
        sample1 = zeros(N2,width*height);
        sample2 = sample1;
        sample3 = sample1;
        %psnr_noise = 10*log10(peak^2/(mean((img11(:)-nimg1(:)).^2)));
        for j = 1:N2
            I=zeros(width,height,3);
            for k=1:imgnum
                num1=imgnum*(j-1)+k;
                Ii=im2double(imread([DIR ,imgFiles(num1).name]));
                I=I+Ii;
            end
            I=I./imgnum;
            p=p1/log(mean2(I));
            I=power(I,p); %imshow(I)
            Ic=Ic+I;
            sample1(j,:) = reshape(I(:,:,1),1,[]);
            sample2(j,:) = reshape(I(:,:,2),1,[]);
            sample3(j,:) = reshape(I(:,:,3),1,[]);
        end
        Ic=Ic./N2;   %imshow(Ic)
        %     [Wwasobi, Winit, ISR,eval(['signals',num2str(i)])]= iwasobi(sample(),1,0.99);
        %-----------------wasobiä����----------------------%
        [~, ~, ~,signals1]= iwasobi(sample1(),1,0.99);
        [~, ~, ~,signals2]= iwasobi(sample2(),1,0.99);
        [~, ~, ~,signals3]= iwasobi(sample3(),1,0.99);
        %     signals = fastica(sample(),'numOfIC',1);
%         eval(['signals',num2str(i),'=', 'signals',';']);
%     end
end
% elseif strcmp(method,'retinex')
%     for i=1:m
%         sample = zeros(N2,width*height);
%         %psnr_noise = 10*log10(peak^2/(mean((img11(:)-nimg1(:)).^2)));
%         for j = 1:N2
%             I=zeros(width,height,3);
%             for k=1:imgnum
%                 num1=imgnum*(j-1)+k;
%                 Ii=im2double(imread([DIR ,imgFiles(num1).name]));
%                 I=I+Ii;
%             end
%             I=I./imgnum;
%             I=retinex1(I);
% %             Ic=Ic+I;
%             sample(j,:) = reshape(I(:,:,i),1,[]);
%         end
% %         Ic=Ic./N2;
%         %     [Wwasobi, Winit, ISR,eval(['signals',num2str(i)])]= iwasobi(sample(),1,0.99);
%         %-----------------wasobiä����----------------------%
%         [~, ~, ~,signals]= iwasobi(sample(),1,0.99);
%         %     signals = fastica(sample(),'numOfIC',1);
%         eval(['signals',num2str(i),'=', 'signals',';']);
%     end
% else
%     for i=1:m
%         sample = zeros(N2,width*height);
%         %psnr_noise = 10*log10(peak^2/(mean((img11(:)-nimg1(:)).^2)));
%         for j = 1:N2
%             I=zeros(width,height,3);
%             for k=1:imgnum
%                 num1=imgnum*(j-1)+k;
%                 Ii=im2double(imread([DIR ,imgFiles(num1).name]));
%                 I=I+Ii;
%             end
%             I=I./imgnum;
% %             Ic=Ic+I;
%             sample(j,:) = reshape(I(:,:,i),1,[]);
%         end
% %         Ic=Ic./N2;
%         [~, ~, ~,signals]= iwasobi(sample(),1,0.99);
%         %     signals = fastica(sample(),'numOfIC',1);
%         eval(['signals',num2str(i),'=', 'signals',';']);
%     end
% end
clear sample1
clear sample2
clear sample3

for i=1:m
    %     for n=1:number1
    %         img=eval(['I',num2str(n)]);
    imgrgb=Ic(:,:,i);
%     imgrgb=I_orig(:,:,i);
%     imgrgb=I(:,:,i);
    %--------------�źŷ��������------------------%
    [maxorg,minorg,index]=signalsort1(eval(['signals',num2str(i)]),imgrgb,N2,'max',r);
    %         [simg,nsimg]=signalsort(eval(['signals',num2str(i)]),imgrgb,N2);
    
    %         for k=1:number2   %k��Ҫ����ķ��������
    %         H = signals(index(N+1-k),:);
    
    Signals=eval(['signals',num2str(i)]);
    H = Signals(index,:);
    Hmin = min(H);
    Hmax = max(H);
    H = (H - Hmin)./(Hmax-Hmin)*(maxorg-minorg)+minorg;
%     H = (H - Hmin)*(maxorg-minorg)./(Hmax-Hmin)+minorg;
%     H1=(H - Hmin)./(Hmax-Hmin);
    maxHH = max(H(:));
    simg = reshape(H,width,height);
    G = maxHH - H;
    nsimg = reshape(G,width,height);
    
    if ssim(nsimg,imgrgb) > ssim(simg,imgrgb)
        simg=nsimg;
    end
    
    eval(['simg',num2str(1),num2str(i),'=', 'simg',';']);
    %         eval(['simg',num2str(k),num2str(i),'=', 'simg',';']);
    
%     psnr_denoise = psnr(simg,imgrgb)
    %         [PSNR, MSE] = psnr(simg,I2(:,:,3))
    %         peak=2;
    %         psnr_denoise = 10*log10(peak^2/mean((simg(:)-imgrgb(:)).^2));
    %         ssimval_denoise = ssim(simg,imgrgb)
%     figure,imshow(simg);
    % %         imwrite(simg,[path,'�����',num2str(k),num2str(i),'.jpg']);
%     imwrite(simg,[path,'�����',num2str(1),num2str(i),'.jpg']);
    
    %         end
    %     end
end
clear signals1
clear signals2
clear signals3
% -------�ϳɷ���ȥ��ͼ------%

imgout=cat(3,eval(['simg',num2str(11)]),eval(['simg',num2str(12)]),eval(['simg',num2str(13)]));
% eval(['imgout',num2str(n),'=', 'imgout',';']);


% time=toc

%%
[h,s,v]=rgb2hsv(imgout);  
% ��˹�˲�
HSIZE= min(width,height);%��˹����˳ߴ�
q=sqrt(2);
SIGMA1=15;%���������c
SIGMA2=80;
SIGMA3=250;
F1 = fspecial('gaussian',HSIZE,SIGMA1/q);
F2 = fspecial('gaussian',HSIZE,SIGMA2/q) ;
F3 = fspecial('gaussian',HSIZE,SIGMA3/q) ;
gaus1= imfilter(v, F1, 'replicate');
gaus2= imfilter(v, F2, 'replicate');
gaus3= imfilter(v, F3, 'replicate');
gaus=(gaus1+gaus2+gaus3)/3;    %��߶ȸ�˹�������Ȩ��Ȩ��Ϊ1/3
% gaus=(gaus*255);
% figure,imshow(gaus,[]);title('gaus���շ���');

% % �����˲�
% r = 16;
% eps = 0.01^2;
% % Iq = zeros(size(I));
% p=simg;
% Iq = guidedfilter(gaus,simg, r, eps); %��һ����������
% % figure,imshow(Iq)

%%
%---------------------------2-----------------------------------%
Imean=0.5;   % mean2(L) mean2(v)  level;  0.5
Imx=max(max(v));
% figure,imhist(simg)
img1=adapthisteq(v,'Distribution','exponential','ClipLimit',0.01); %'uniform'ƽ̹ 'rayleigh'���� 'exponential'����
img1=0.5*v+0.5*img1;  %mean2(round(simg1*255))

% imhist(simg1)
% figure,
% subplot(121),imshow(simg)
% subplot(122),imshow(simg1)
%---------------------------3-----------------------------------%
% for i=1:width
%     for j=1:height
%         if Imx-img1(i,j)<=0.14
%             img1(i,j)=img1(i,j)-0.2; 
%             p1(i,j)=1.5*(Imean-gaus(i,j))/Imean;
%         else
%             p1(i,j)=(Imean-gaus(i,j))/Imean; %������ֵ����Ԥ��ֵ���򽵵�ԭͼ����
%         end
%     end
% end

Beta=[0.2328 0.1290 0.5532 46.4941 1.6096];
for i=1:width
    for j=1:height
        deta=Imx-img1(i,j);
%         if deta<=0.05
%             simg1(i,j)=simg1(i,j)-0.05;
%             p1(i,j)=(Imean-gaus(i,j))/Imean; %������ֵ����Ԥ��ֵ���򽵵�ԭͼ����
        if deta<=0.3
%             simg1(i,j)=simg1(i,j)-0.16; 
%             simg1(i,j)=simg1(i,j)-0.32+0.32./(1+exp(-12*deta.^1));
            img1(i,j)=img1(i,j)-Beta(1)+Beta(2)./(Beta(3)+exp(-Beta(4)*deta.^Beta(5)));
%             Imean(i,j)=0.5-0.5*(Beta(1)-Beta(2)./(Beta(3)+exp(-Beta(4)*deta.^Beta(5))));
%             p1(i,j)=(Imean-gaus(i,j))/Imean;
        end
    end
end


p1=(Imean-gaus)/Imean; %ԭ  %������ֵ����Ԥ��ֵ���򽵵�ԭͼ����
% gama=(ones(size(p1))*0.5).^p1;  imshow(p1)
p=p2/log(mean2(v))+beta;
gama=power(p,p1);%���ݹ�ʽgammaУ���������Ĺ�ʽ���� mean2(gamma)
vout=power(img1,gama);   %mean2(round(simg*255))

rgb=hsv2rgb(h,s,vout);   %ת��rgb�ռ���ʾ

ymin = min(min(rgb));
ymax = max(max(rgb));
rgb2 = (rgb - ymin)./(ymax-ymin);%*(maxorg-minorg)+minorg; %��һ��

figure,
subplot(221),imshow(I_orig); title('ԭͼ');
subplot(222),imshow(I1); title('ԭ������ǿͼ');
subplot(223),imshow(rgb2);title(['�ϳɷ���ȥ��ͼ',method]);

quality=testquality(uint8(round(rgb2*255)),1);

imwrite(rgb2,[path,'rgb�ϳɷ���ȥ��ͼ',num2str(200/imgnum),method,'.jpg']);

% psnr_denoise = psnr(rgb2,Ic)
ssimval_denoise = ssim(rgb2,Ic)
    
% imwrite(I1,[path,'ԭ������ǿͼ','.jpg']);
% figure,
% imshowpair(I1,imgout,'montage') %������ʾ
% qualityscore = SSEQ(imgout)

imwrite(Ic,[path(1:15),'�ο�ͼ480-640.jpg']);

