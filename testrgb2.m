%------------RGB改进版本---------------%
% clc;
clear;
close all;

% path='C:\Users\CG\Desktop\5.28\night1\rgbresult1\';
% DIR='C:\Users\CG\Desktop\5.28\night1\night1\';

% DIR='D:\Cammera\7.17-480-640\';
% DIR='D:\Cammera\7.2\1\576-704-1\';
DIR='D:\Cammera\7.2\1\480-640\';
imgFiles = dir([DIR,'*.jpg']);%输入图像的格式  dir('')列出指定目录下所有子文件夹和文件
tic;
% I1 = im2double(imread('Lena512.png'));
% I2 = im2double(imread('Baboon512.png'));
I= imread([DIR , imgFiles(1).name]);
I_orig = im2double(I);
Imean=mean2(I_orig );
% subplot(122),imshow(I2);title('待分离噪图2');
[~,~,m]=size(I_orig);
[N, ~]= size(imgFiles);
[width,height]=size(I_orig(:,:,1));

num=8;%分组数

imgnum=200/num;%帧叠加数

r=0.2;%尺寸比例变换系数
% r=[50:100,50:100];%局部尺寸
N2=N/imgnum;
%number1是要分离出的图的数量
%number1是要储存的图层的数量
method='power';

p1=log(90/255);
p2=p1;
switch method
    case 'power'
%         P=[0:5:20 35 60 75 90 100];
%         p = discretize(Imean,[-Inf P Inf]);
%         p=p+abs(P(p-1)-Imean)/(P(p)-P(p-1));
%         p=round(p/10,2); %保留2位小数
        p=p1/log(Imean);
%         p=0.4;%伽马增强指数
        I1=power(I_orig,p);
%         path='D:\Cammera\7.17result\';
        path='D:\Cammera\7.2\rgbresult1\';
    case 'retinex'
        I1=retinex1(I_orig);
        path='G:\照片\6.17\rgbresult2\';
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
        %-----------------wasobi盲分离----------------------%
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
%         %-----------------wasobi盲分离----------------------%
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
% tic
for i=1:m
    %     for n=1:number1
    %         img=eval(['I',num2str(n)]);
    imgrgb=Ic(:,:,i);
%     imgrgb=I_orig(:,:,i);
%     imgrgb=I(:,:,i);
    %--------------信号分离后排序------------------%
    [maxorg,minorg,index]=signalsort1(eval(['signals',num2str(i)]),imgrgb,N2,'max',r);
    %         [simg,nsimg]=signalsort(eval(['signals',num2str(i)]),imgrgb,N2);
    
    %         for k=1:number2   %k是要储存的分离层数量
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
    % %         imwrite(simg,[path,'分离层',num2str(k),num2str(i),'.jpg']);
%     imwrite(simg,[path,'分离层',num2str(1),num2str(i),'.jpg']);
    
    %         end
    %     end
end
clear signals1
clear signals2
clear signals3
% -------合成分离去噪图------%

imgout=cat(3,eval(['simg',num2str(11)]),eval(['simg',num2str(12)]),eval(['simg',num2str(13)]));
% eval(['imgout',num2str(n),'=', 'imgout',';']);

p2=p1/log(mean2(imgout));
imgout1=power(imgout,p2);

% % min(min(imgout))
% % max(max(imgout))

time=toc

figure,
subplot(221),imshow(I_orig); title('原图');
subplot(222),imshow(I1); title('原低照增强图');
subplot(223),imshow(imgout1);title(['合成分离去噪图',method]);

imwrite(imgout1,[path,'合成分离去噪图',num2str(200/imgnum),method,'.jpg']);

% imwrite(I1,[path,'原低照增强图','.jpg']);
% figure,
% imshowpair(I1,imgout,'montage') %并排显示
% qualityscore = SSEQ(imgout)
% quality=testquality(uint8(round(imgout1*255)));

%%
% [h,s,v]=rgb2hsv(imgout);  
% % 高斯滤波
% HSIZE= min(width,height);%高斯卷积核尺寸
% q=sqrt(2);
% SIGMA1=15;%论文里面的c
% SIGMA2=80;
% SIGMA3=250;
% F1 = fspecial('gaussian',HSIZE,SIGMA1/q);
% F2 = fspecial('gaussian',HSIZE,SIGMA2/q) ;
% F3 = fspecial('gaussian',HSIZE,SIGMA3/q) ;
% gaus1= imfilter(v, F1, 'replicate');
% gaus2= imfilter(v, F2, 'replicate');
% gaus3= imfilter(v, F3, 'replicate');
% gaus=(gaus1+gaus2+gaus3)/3;    %多尺度高斯卷积，加权，权重为1/3
% % gaus=(gaus*255);
% % figure,imshow(gaus,[]);title('gaus光照分量');
% 
% % 引导滤波
% % r = 16;
% % eps = 0.01^2;
% % % Iq = zeros(size(I));
% % p=v;
% % Iq = guidedfilter(gaus,v, r, eps); %第一个是引导量
% % % figure,imshow(Iq)
% %%
% % Iomean=0.5;
% % imshow(gaus1)
% % --------------------原版算法高斯基于二维伽马-------------------%
% Imean=0.5;   %mean2(v)  level;  0.5
% 
% alph=0.6;
% Ga=gaus.^0.4;
% Gb=1-(1-gaus).^0.6;
% % Ga=Iq.^0.6;
% % Gb=1-(1-Iq).^0.7;
% G=alph*Ga+(1-alph)*Gb;
% 
% p1=(Imean-G)/Imean; %原
% % gama=(ones(size(p1))*0.5).^p1;
% gama=power(Imean,p1);%根据公式gamma校正处理，论文公式有误
% vout=power(v,gama);
% 
% p3=p2/log(mean2(vout));
% vout=power(vout,p3);
% % mean2(round(vout*255))
% 
% rgb=hsv2rgb(h,s,vout);   %转回rgb空间显示
% 
% p4=p2./log(mean2(rgb)); 
% rgb=power(rgb,p4);
% % mean2(round(rgb*255))
% 
% figure,imshow(rgb);title('校正结果')
% % imwrite(rgb2,[path,'自适应Gamma二次校正',num2str(200/imgnum),method,'.jpg']);
% 
% % quality=testquality(uint8(round(imgout*255)),1);
% % quality1=testquality(uint8(round(rgb*255)),1);
% 
