clear
% clc
path='C:\Users\CG\Desktop\5.28\';
tic
%%
% ----------------------------4�ֵ��ն�ͼ����ǿ------------------------%

I=im2double(imread('C:\Users\CG\Desktop\5.28\TU\���ն�\horse rider.PNG')); %IMG_3651.jpg
% f = 512 / max(size(I));
% S = imresize(I,f);

%a=0.01;b=0.2;c=1; 
% icount=8; 
% step=2.4;
a=0.008;b=0.3;c=1;
icount=4;
step=2.2;%decide the times 
I=I+1;
Z=log(I);
Z=Z/max(Z(:));
R=zeros(size(Z));
% nn = 3;
method=1;
for i=1:3
    if method==1            %zm�㷨
        R(:,:,i)=zm_retinex_Light1(Z(:,:,i),a,b,c,icount,0);
    elseif method==2        %mccann�㷨
        R(:,:,i)=zm_retinex_mccann992(Z(:,:,i),icount);
    elseif method==3        %Kimmel�㷨����1
%         R(:,:,i)=zm_Kimmel(Z(:,:,i),a,b,icount,4.5);
        R(:,:,i)=zm_Kimmel(Z(:,:,i),0.008,0.3,1,1.5); %���һ��������Ч
    elseif method==4        %Kimmel�㷨����2
        R(:,:,i)=Z(:,:,i)-zm_Devir_retinex(Z(:,:,i),a,b);
    end
end

m=mean2(R);s=std(R(:)); %��Ϊ�൱��mean( mean( A ) ) �൱�ڶ���һ������������ƽ��ֵ��
mini=max(m-step*s,min(R(:)));maxi=min(m+step*s,max(R(:)));
range=maxi-mini;
result1=(R-mini)/range*0.8;% result1=max(result1(:))-result1;
figure,imshow(result1);
ssimval = ssim(result1,I)
% 
% %         sprintf('���ڴ���%s',imgFiles(j).name)
% %         imwrite(result1,[path,num2str(method),imgFiles(j).name]);
% %         title(['ȥ���RGB��ǿͼ',num2str(method)]);
% toc

% imwrite(result1,[path,'zm_retinex_Light.jpg']);

%%
%--------------------tsmooth���ն�ͼ����ǿ����-----------------------%
% tic
% img = im2double(imread('night2.jpg'));
% f = 512 / max(size(img));
% L = imresize(img,f);
% L(L < 0) = 0;L(L > 1) = 1;
% T = max(max(L(:,:,1),L(:,:,2)),L(:,:,3));
% [m,n,k] = size(L);
% % T1 = tsmooth(T,0.15,2,0.05,1);
% T1 = tsmooth(T,0.15,2,0.05,4);
% % <span style="font-family:Arial, Helvetica, sans-serif;">% ������Ϊ��͸��ͼ</span>
% figure,imshow([T T1]);
% % I = 1 - ( ( ( repmat(1,[m,n,3]))  - L ) - repmat(0.95 * (1 - T1),[1 1 3]) ) ./  repmat(T1,[1 1 3]) ;
% I = 1 - ( ( ones(m,n,3)  - L ) - repmat(0.95 * (1 - T1),[1 1 3]) ) ./  repmat(T1,[1 1 3]) ;
% I(I < 0) = 0;I(I > 1) = 1;
% toc
% figure,imshow([L I]);
% ssimval = ssim(I,L)
% imwrite(I,[path,'ALTM_Retinex1.jpg']);

%%
%---------------CSDNһ�ֿ��ټ򵥶�����Ч�ĵ��ն�ͼ��ָ��㷨-------------%

% function outval = ALTM_Retinex1(I)
% tic
% I=im2double(imread('night2.jpg'));
% f = 512 / max(size(I));
% II = imresize(I,f);
% II(II < 0) = 0;II(II > 1) = 1;
% Ir=double(II(:,:,1)); Ig=double(II(:,:,2)); Ib=double(II(:,:,3));
% % Global Adaptation
% Lw = 0.299 * Ir + 0.587 * Ig + 0.114 * Ib;% input world luminance values %RGBתYUV
% Lwmax = max(max(Lw));% the maximum luminance value
% [m, n] = size(Lw);
% Lwaver = exp(sum(sum(log(0.001 + Lw))) / (m * n));% log-average luminance
% Lg = log(Lw / Lwaver + 1) / log(Lwmax / Lwaver + 1);
% gain = Lg ./ Lw;
% gain(find(Lw == 0)) = 0;
% outval = cat(3, gain .* Ir, gain .* Ig, gain .* Ib);
% toc
% figure; imshow(outval)
% ssimval = ssim(outval,II)
% imwrite(outval,[path,'ALTM_Retinex1.jpg']);

%%
%---------------------ԭ����оֲ������㷨--------------------%
% tic
% % function outval = ALTM_Retinex(I)
% I=im2double(imread('night2.jpg'));
% f = 512 / max(size(I));
% L = imresize(I,f);
% L(L < 0) = 0;L(L > 1) = 1;
% Ir=double(L(:,:,1)); Ig=double(L(:,:,2)); Ib=double(L(:,:,3));
% % Global Adaptation
% Lw = 0.299 * Ir + 0.587 * Ig + 0.114 * Ib;% input world luminance values
% Lwmax = max(max(Lw));% the maximum luminance value
% [m, n] = size(Lw);
% Lwaver = exp(sum(sum(log(0.001 + Lw))) / (m * n));% log-average luminance
% Lg = log(Lw / Lwaver + 1) / log(Lwmax / Lwaver + 1);
% % Local Adaptation
% kenlRatio = 0.01;
% krnlsz = floor(max([3, m * kenlRatio, n * kenlRatio]));
% Lg1 = maxfilt2(Lg, [krnlsz, krnlsz]);
% Lg1 = imresize(Lg1, [m, n]);
% Hg = guidedfilter(Lg, Lg1, 10, 0.01);
% eta = 36;
% alpha = 1 + eta * Lg / max(max(Lg));
% alpha = alpha .* (alpha .^ (1 ./ alpha));
% b = max(max(alpha));
% a = 1.35;
% alpha = 2 * atan(a * alpha / b) / pi * b;
% Lgaver = exp(sum(sum(log(0.001 + Lg))) / (m * n));
% lambda = 10;
% beta = lambda * Lgaver;
% Lout = alpha .* log(Lg ./ Hg + beta);
% %Lout = normfun(Lout, 1);
% Lout = SimplestColorBalance(Lout, 0.005, 0.001, 1);
% gain = Lout ./ Lw;
% gain(find(Lw == 0)) = 0;
% outval = cat(3, gain .* Ir, gain .* Ig, gain .* Ib);
% toc
% ssimval = ssim(outval,L)
% imshow(outval)
% % imwrite(outval,[path,'ALTM_Retinexԭ��.jpg']);

%%
%-----------------------�����оֲ������㷨----------------------------%

% % function outval = ALTM_Retinex2(I)
% tic
% I=imread('night2.jpg');
% f = 512 / max(size(I));
% L = imresize(I,f);
% % L(L < 0) = 0;L(L > 1) = 1;
% II=im2double(L);
% %figure,imshow(L)
% Ir=double(II(:,:,1)); Ig=double(II(:,:,2)); Ib=double(II(:,:,3));
% % Global Adaptation
% Lw = 0.299 * Ir + 0.587 * Ig + 0.114 * Ib;% input world luminance values
% Lwmax = max(max(Lw));% the maximum luminance value
% [m, n] = size(Lw);
% Lwaver = exp(sum(sum(log(0.001 + Lw))) / (m * n));% log-average luminance
% Lg = log(Lw / Lwaver + 1) / log(Lwmax / Lwaver + 1);
% % Global Adaptation
% Lw = double(L(:,:,1));
% Lw = Lw ./ 255.0;
% %Lw = 0.299 * Ir + 0.587 * Ig + 0.114 * Ib;% input world luminance values
% Lwmax = max(max(Lw));% the maximum luminance value
% [m, n] = size(Lw);
% Lwaver = exp(sum(sum(log(0.001 + Lw))) / (m * n));% log-average luminance
% Lg = log(Lw / Lwaver + 1) / log(Lwmax / Lwaver + 1);
% 
% Lout = SimplestColorBalance(Lg, 0.005, 0.001, 1);
% gain = Lout ./ Lw;
% gain(find(Lw == 0)) = 0;
% outval = cat(3, gain .* Ir, gain .* Ig, gain .* Ib);
% toc
% ssimval = ssim(outval,II)
% imshow(outval)
% imwrite(outval,[path,'ALTM_Retinex2.jpg']);

%%
%----------------RGB��ɫ�����Ż��㷨------------------%

% % function outval = ALTM_Retinex(I)
% tic
% I=im2double(imread('night2.jpg'));
% f = 512 / max(size(I));
% II = imresize(I,f);
% II(II < 0) = 0;II(II > 1) = 1;
% Ir=double(II(:,:,1)); Ig=double(II(:,:,2)); Ib=double(II(:,:,3));
% % Global Adaptation
% Lw = 0.299 * Ir + 0.587 * Ig + 0.114 * Ib;% input world luminance values
% Lwmax = max(max(Lw));% the maximum luminance value
% [m, n] = size(Lw);
% Lwaver = exp(sum(sum(log(0.001 + Lw))) / (m * n));% log-average luminance
% Lg = log(Lw / Lwaver + 1) / log(Lwmax / Lwaver + 1);
% % Local Adaptation
% kenlRatio = 0.01;
% krnlsz = floor(max([3, m * kenlRatio, n * kenlRatio]));
% Lg1 = maxfilt2(Lg, [krnlsz, krnlsz]);
% Lg1 = imresize(Lg1, [m, n]);
% Hg = guidedfilter(Lg, Lg1, 10, 0.01);
% eta = 36;
% alpha = 1 + eta * Lg / max(max(Lg));
% alpha = alpha .* (alpha .^ (1 ./ alpha));
% b = max(max(alpha));
% a = 1.35;
% alpha = 2 * atan(a * alpha / b) / pi * b;
% Lgaver = exp(sum(sum(log(0.001 + Lg))) / (m * n));
% lambda = 10;
% beta = lambda * Lgaver;
% Lout = alpha .* log(Lg ./ Hg + beta);
% %Lout = normfun(Lout, 1);
% Lout = SimplestColorBalance(Lout, 0.005, 0.001, 1);
% gain = Lout ./ Lw;
% gain(find(Lw == 0)) = 0;
% Ir = (1/2.0) .* (gain .* (Ir+Lw) + Ir - Lw);
% Ig = (1/2.0) .* (gain .* (Ig+Lw) + Ig - Lw);
% Ib = (1/2.0) .* (gain .* (Ib+Lw) + Ib - Lw);
% 
% % Ir = (1/2) * (Lg / Lw * (Ir + Lw) + Ir - Lw);
% % Ig = (1/2) * (Lg / Lw * (Ig + Lw) + Ig - Lw);
% % Ib = (1/2) * (Lg / Lw * (Ib + Lw) + Ib - Lw);
% 
% outval = cat(3, Ir,Ig,Ib);
% toc
% ssimval = ssim(outval,II)
% imshow(outval)
% imwrite(outval,[path,'ALTM_Retinex.jpg']);

%%
%--------------------ֱ��ͼ�Ҷȱ任----HE-----------------%

% close all;
% clear 
% tic
% I=imread('IMG_3651.jpg');
% % I = I(:, :, 1);
% f = 512 / max(size(I));
% S = imresize(I,f);
% 
% Ir=histeq(S(:, :, 1),64);
% Ig=histeq(S(:, :, 2),64);
% Ib=histeq(S(:, :, 3),64);
% J = cat(3, Ir,Ig,Ib);
% % imwrite(J,'C:\Users\shou\Desktop\new\1zeng.jpg');
% % transforms the intensity image I,returning J an intensity
% toc
% figure;
% imshow(J);
% ssimval = ssim(J,S)

%%
%-------------------����Ӧֱ��ͼ���⻯--CLAHE----------------%
clc
clear
tic
I = imread('C:\Users\CG\Desktop\5.28\night3.bmp');  %'IMG_3651.jpg'
f = 512 / max(size(I));
img = imresize(I,f);

rimg = img(:,:,1);
gimg = img(:,:,2);
bimg = img(:,:,3);
resultr = adapthisteq(rimg,'ClipLimit',0.04);
resultg = adapthisteq(gimg,'ClipLimit',0.04);
resultb = adapthisteq(bimg,'ClipLimit',0.04);
result = cat(3, resultr, resultg, resultb);

toc
imshow(result);
ssimval = ssim(result,img)
%%
%���� clown �Ի�ȡͼ�� X �����������ɫͼ map����ʾ X �� map ���ɵ�ͼ��
load clown 
imagesc(X)
colormap(map)
%ʹ�� contrast ������ǿͼ�� X �ԱȶȵĻҶ���ɫͼ��Ȼ��ʹ������ɫͼ������ʾ��
newmap = contrast(X);
colormap(newmap)
%ʹ�ý��ٵĻҶȼ�����ʾͼ��
load clown
imagesc(X)
newmap1 = contrast(X);
colormap(newmap1)
%��������ʹ�� contrast �ٴ���һ�������� 10 ���Ҷȵ���ɫͼ��ʹ������ɫͼ������ʾ�����Կ�������Ӱ�����ڱ�����ͬʱ��ʧ�˲���ϸ�ڡ�
newmap2 = contrast(X,10);
colormap(newmap2)

%%
%% �ֲ��Աȶ���ǿ
clear all;clc;close all;
ImgFile='E:\ͼ����\������˹ͼƬ��\DIP3E_Original_Images_CH03\tungsten_original.tif';
step=1;
para.E=4.0;
para.k0=0.4; % ��ֵ����ֵ
para.k1=0.02; % ��׼������ֵ
para.k2=0.4; % ��׼������ֵ
ImgIn=imread(ImgFile);
ImgHistEq=histeq(ImgIn,256);
ImgIn=double(ImgIn);
% ImgIn=double(rgb2gray(ImgIn));
[ MeanLocal,VarLocal ] = LocalStatistics( ImgIn, step );
[ ImgOut ] = LocalEnhancement( ImgIn, MeanLocal, VarLocal,para );

figure;imshow(uint8(ImgIn));title('ԭͼ');
figure;imshow(uint8(ImgOut));title('�ֲ�ͳ����ǿ');
figure;imshow(ImgHistEq);title('ȫ�ֻҶ���ǿ - ֱ��ͼ����');
%%
% colorspace = ' ';
% A=im2double(imread('IMG_3651.jpg'));
% Img= imresize(A,0.2);
% [I1, img_max, img_min] = rgb_to(Img,colorspace);
% figure;imshow(I1)
% y_est = rgb_to(I1, colorspace, true, img_max, img_min);
% figure;imshow(y_est)
% toc
% 
% function [o, o_max, o_min] = rgb_to(img, colormode, inverse, o_max, o_min)
% 
%     if exist('colormode', 'var') && strcmp(colormode, 'opp') %strcmp()�Ƚ�����char��������ֵ����С��
%         % Forward
%         A =[1/3 1/3 1/3; 0.5  0  -0.5; 0.25  -0.5  0.25];
%         % Inverse
%         B =[1 1 2/3;1 0 -4/3;1 -1 2/3];
%     else
%         % YCbCr
%         A = [0.299, 0.587, 0.114; -0.168737, -0.331263, 0.5;  0.5,  -0.418688,  -0.081313];
%         B = [1.0000, 0.0000, 1.4020; 1.0000, -0.3441, -0.7141; 1.0000, 1.7720, 0.0000];
%     end
% 
%     if exist('inverse', 'var') && inverse
%         % The inverse transform
%         o = (reshape(img, [size(img, 1) * size(img, 2), 3]) .* (o_max - o_min) + o_min) * B';
%     else 
%         % The color transform to YCbCr  Y=0.299R+0.587G+0.114B��Cb=0.564(B-Y)��Cr=0.713(R-Y)
%         o = reshape(img, [size(img, 1) * size(img, 2), 3]) * A';
%         %o(:, 2:3) = o(:, 2:3) + 0.5;
%         o_max = max(o, [], 1);
%         o_min = min(o, [], 1);
%         o = (o - o_min) ./ (o_max - o_min);
% %         scale = sum(A'.^2) ./ (o_max - o_min).^2;
%     end
%     
%     o = reshape(o, [size(img, 1), size(img, 2), 3]);
% end
% 
