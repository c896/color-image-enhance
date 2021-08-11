% img=imread('D:\Cammera\10.09\1\Pic_2020_10_09_180507_722.bmp');
% mean2(img)
%----------------------------retinex低照度图像增强-帧平均-----------------------%
% clc
clear 
close all
% path='G:\照片\6.16\';
% DIR='G:\照片\6.16\576-704\';

% DIR='D:\Cammera\10.09\2-480-640\'; 
% path='D:\Cammera\10.09\2result\';

DIR='D:\Cammera\7.02\1\480-640\';
path='D:\Cammera\7.02\';

% DIR='D:\Cammera\9.28\9.28-2-480-640\'; %均值1 0.0214 5.457  均值2 0.0178 4.539
% path='D:\Cammera\9.28\';

% path='D:\Cammera\7.17\';
% DIR='D:\Cammera\7.17\7.17-480-640\';

% DIR='D:\Cammera\10.09\1-480-640\'; 
% path='D:\Cammera\10.09\1result\';

imgFiles = dir([DIR,'*.jpg']);%输入图像的格式  dir('')列出指定目录下所有子文件夹和文件
tic

[N, ~]= size(imgFiles);
I=imread([DIR ,imgFiles(1).name]);
img=im2double(I);
% figure,imshow(img),title('原低照图')

% A=zeros(size(img));
for k=2:N
    A=im2double(imread([DIR ,imgFiles(k).name]));
    img=img+A;
%     sprintf('正在读取%s',imgFiles(k).name)
end
img=img./N;

% figure;imshow(I),title('原低照度图帧平均');
imwrite(img,[path,'原低照度帧平均480-640','.jpg']);
% mean2(uint8(round(img*255))) % mean2(img)*255
% img0=imread([path,'原低照度帧平均480-640','.jpg']);
% mean2(img0)
%%
%------------------power伽马增强-------------%
% Iomean=0.4; %90/255
% % img_mean=mean2(img);
% img_mean=sum(median(median(img)))/3;
% p=log(Iomean)/log(img_mean)+0.005;
% imp=power(img,p); %pmean=0.3184;
% 
% time=toc
% figure,imshow(imp)
% quality=testquality(uint8(round(imp*255)),0);
% % imwrite(imp,[path,'帧平均power480-640.jpg']);
% % % qualityscore = SSEQ(imp)
% 
% % quality_orig=testquality(I);
% % qualityscore = SSEQ(y)  % 44.7349
%%
%------------------retinex SSR MSR MSRCR伽马增强-------------%
% method=3;
% switch method 
%     case 1
%         imp=SSR(img,80);
%         imwrite(imp,[path,'帧平均SSR480-640','.jpg']);
%     case 2
%         imp=MSR(img,10,80,200);
%         imwrite(imp,[path,'帧平均MSR480-640','.jpg']);
%     case 3
%         imp=MSRCR(img,10,80,200);
%         imwrite(imp,[path,'帧平均MSRCR480-640','.jpg']);
% end
% 
% time=toc
% figure,imshowpair(img,imp,'montage')
% quality1=testquality(uint8(round(imp*255)),0);
% 
% imgpwer=im2double(imread([path,'帧平均power480-640.jpg'])); 
% ssimval = ssim(imp,imgpwer);
% str=sprintf('ssim:%.4f',ssimval);  %输出类型为char
% display(str)
% % ssimval = ssim(img,imp)

%%
% %----------------retinex增强------------------%
% %     f = 512 / max(size(I));
% %     I = imresize(I,f);
% %     I(I < 0) = 0;I(I > 1) = 1;
% 
% % a=0.01;b=0.2;c=1; 
% % icount=8;
% a=0.008;b=0.3;c=1;
% icount=4;
% step=2.4;%decide the times  %step=2.2;
% 
% I=img+1;
% Z=log2(I); %Z=log(I);
% Z=Z/max(Z(:));
% R=zeros(size(Z));
% % R1=zeros(size(Z));
% nn = 1;
% for method=1:1:nn
%     for i=1:3
%         if method==1            %zm算法
%             [R(:,:,i),L]=zm_retinex_Light1(Z(:,:,i),a,b,c,icount,0);
%         elseif method==2        %mccann算法
%             R(:,:,i)=zm_retinex_mccann992(Z(:,:,i),icount);
%         elseif method==3        %Kimmel算法程序1
% %             R(:,:,i)=zm_Kimmel(Z(:,:,i),a,b,icount,4.5); %最后一个参数无效
%             R(:,:,i)=zm_Kimmel(Z(:,:,i),a,b,1,4.5); %最后一个参数无效
%         elseif method==4        %Kimmel算法程序2
%             R(:,:,i)=Z(:,:,i)-zm_Devir_retinex(Z(:,:,i),a,b);
%         end
%     end
%     
%     m=mean2(R);s=std(R(:));
%     mini=max(m-step*s,min(R(:)));maxi=min(m+step*s,max(R(:)));
%     range=maxi-mini;
%     result=(R-mini)/range*0.8;
%     %result=max(result(:))-result;  
%     
% %     sprintf('正在处理%s',imgFiles(1).name)
% %     imwrite(result,[path,num2str(method),imgFiles(1).name]);
% %     imwrite(result,[path,'帧平均512retinex_Light','.jpg']);
% %     figure,imshow(result); title(['去噪后RGB增强图',num2str(method)]);
% end
% time1=toc
% figure,imshowpair(img,result,'montage')
% imwrite(result,[path,'帧平均zm_retinex_Light480-640.jpg']);
% 
% quality=testquality(uint8(round(result*255)),0);
% imgpwer=im2double(imread([path,'帧平均power480-640.jpg'])); 
% ssimval = ssim(result,imgpwer)

%%
% %---------------直方图均衡--HE------------------%
% S=img;
% Ir=histeq(S(:, :, 1),64);%n - 离散灰度级的数量64 （默认）
% Ig=histeq(S(:, :, 2),64);
% Ib=histeq(S(:, :, 3),64);
% J = cat(3, Ir,Ig,Ib);
% % imwrite(J,'C:\Users\shou\Desktop\new\1zeng.jpg');
% % transforms the intensity image I,returning J an intensity
% time=toc
% imhist(J)
% figure,imshowpair(S,J,'montage')
% imwrite(J,[path,'帧平均HE480-640.jpg']);
% 
% quality=testquality(uint8(round(J*255)),0);
% imgpwer=im2double(imread([path,'帧平均power480-640.jpg'])); 
% ssimval = ssim(J,imgpwer)
%%
%-------------------自适应直方图均衡化--CLAHE----------------%
% rimg = img(:,:,1);
% gimg = img(:,:,2);
% bimg = img(:,:,3);
% % resultr = adapthisteq(rimg,'ClipLimit',0.04);
% resultr = adapthisteq(rimg,'Distribution','exponential');%,'ClipLimit',0.01
% resultg = adapthisteq(gimg,'Distribution','exponential');
% resultb = adapthisteq(bimg,'Distribution','exponential');
% result = cat(3, resultr, resultg, resultb);
% 
% time=toc
% imhist(result)
% figure,imshowpair(img,result,'montage')
% imwrite(result,[path,'帧平均HE480-640.jpg']);
% 
% quality1=testquality(uint8(round(result*255)),0);
% imgpwer=im2double(imread([path,'帧平均power480-640.jpg'])); 
% ssimval = ssim(result,imgpwer)
%%
%---------------------LSCN---------------%
% imp=power(img,0.5);
% I1=uint8(imp*255);
% 
% V = rgb2v(I);
% V_flm = LSCN(V);
% [Contrast(k,1), Spatial_frequency(k,1), Gradient(k,1)] ...
%     = QEvaluation(V_flm);
% JND(k,1) = JND_zhan2(V_flm);
% result=v2rgb(I,V_flm);
% 
% time=toc
% figure,imshow([I result]);
% imwrite(result,[path,'帧平均LSCN480-640.jpg']);
% 
% % display([Contrast, Spatial_frequency, Gradient JND])
% quality1=testquality(result,0);
% 
% imgpwer=imread([path,'帧平均power480-640.jpg']); 
% ssimval = ssim(result,imgpwer)

%%
% 高斯滤波
[h,s,v]=rgb2hsv(img); 
HSIZE= min(size(img,1),size(img,2));%高斯卷积核尺寸
q=sqrt(2);
SIGMA1=15;%论文里面的c
SIGMA2=80;
SIGMA3=250;
F1 = fspecial('gaussian',HSIZE,SIGMA1/q);
F2 = fspecial('gaussian',HSIZE,SIGMA2/q) ;
F3 = fspecial('gaussian',HSIZE,SIGMA3/q) ;
gaus1= imfilter(v, F1, 'replicate');
gaus2= imfilter(v, F2, 'replicate');
gaus3= imfilter(v, F3, 'replicate');
gaus=(gaus1+gaus2+gaus3)/3;    %多尺度高斯卷积，加权，权重为1/3
% gaus=(gaus*255);
% figure,imshow(gaus,[]);title('gaus光照分量');

% --------------------原版算法高斯基于二维伽马-------------------%
Imean=0.5;   %mean2(v)  level;  0.5
% p=log(Iomean)/log(Imean).*0.5;
% p1=(gaus-Imean)/Imean;
p1=(Imean-gaus)/Imean; %原
% gama=(ones(size(p1))*0.5).^p1;
gama=power(Imean,p1);%根据公式gamma校正处理，论文公式有误
vout=power(v,gama);
rgb=hsv2rgb(h,s,vout);   %转回rgb空间显示

time=toc

figure,imshow(rgb);title('原版校正结果')
imwrite(rgb,[path,'帧平均SVLM480-640.jpg']);
quality=testquality(uint8(round(rgb*255)),0);

imgpwer=im2double(imread([path,'帧平均power480-640.jpg'])); 
ssimval = ssim(rgb,imgpwer)
