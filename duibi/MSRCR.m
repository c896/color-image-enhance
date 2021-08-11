%---------带色彩恢复的多尺度---------------%
function result=MSRCR(img,sigma1,sigma2,sigma3)
% [filename,pathname]=uigetfile('*.*','融合图像');
% I=im2double(imread([pathname,filename]));
% img = imread('IMG_3651.jpg');
I=im2double(img);
% f = 512 / max(size(I));
% I = imresize(I,f);
% I(I < 0) = 0;I(I > 1) = 1;
% figure,imshow(I)
% tic;
R = I(:, :, 1);
G = I(:, :, 2);
B = I(:, :, 3);
% R = double(R);
% G = double(G);
% B = double(B);

[N1, M1] = size(R);

Rlog = log(R+1);%增强原理第一步取对数
Rfft2 = fft2(R);%进行二维傅里叶变换，将R通道分量由空间域变为频率域


%sigma1 = 128;
% sigma1 = 512;
F1 = fspecial('gaussian', [N1,M1], sigma1);%高斯低通滤波，有两个参数，hsize表示模板尺寸，默认值为【3 3】，sigma为滤波器的标准值，单位为像素，默认值为0.5.
%F1 = fspecial('unsharp', 0.2);                            F1为一个二维滤波器 
Efft1 = fft2(double(F1));%进行傅里叶变换，将滤波器F1由空间域变为频率域

%第二步高斯模板对原图像作卷积，即相当于对原图像作低通滤波，得到低通滤波后的图像D(x,y)
DR0 = Rfft2.* Efft1; %R分量和滤波器进行点乘
DR = ifft2(DR0);%进行傅里叶反变换，将频率域变为空间域

%第三步在对数域中，用原图像减去低通滤波后的图像，得到高频增强的图像G(x,y)=S’(x,y)-log(D(x,y))
DRlog = log(DR +1);
Rr1 = Rlog - DRlog;

% sigma2 = 256;
%sigma2 = 2500;
F2 = fspecial('gaussian', [N1,M1], sigma2);
Efft2 = fft2(double(F2));

DR0 = Rfft2.* Efft2;
DR = ifft2(DR0);

DRlog = log(DR +1);
Rr2 = Rlog - DRlog;

% sigma3 = 512;
%sigma3 = 5100;
F3 = fspecial('gaussian', [N1,M1], sigma3);
Efft3 = fft2(double(F3));

DR0 = Rfft2.* Efft3;
DR = ifft2(DR0);

DRlog = log(DR +1);
Rr3 = Rlog - DRlog;

Rr = (Rr1 + Rr2 +Rr3)/3;

%a = 125;
a = 15;
II = imadd(R, G);
II = imadd(II, B);
Ir = immultiply(R, a);
%定义彩色恢复因子C
C = imdivide(Ir, II);
C = log(C+1);
% 将增强的分量乘以色彩恢复因子
Rr = immultiply(C, Rr);
%第四步对G(x,y)取反对数，得到增强后的图像R(x,y)=exp(G(x,y))
EXPRr = exp(Rr);
%第五步对R(x,y)作对比度增强，得到最终的结果图像。
MIN = min(min(EXPRr));
MAX = max(max(EXPRr));
EXPRr = (EXPRr - MIN)/(MAX - MIN);
EXPRr = adapthisteq(EXPRr);
% G通道处理
Glog = log(G+1);
Gfft2 = fft2(G);

DG0 = Gfft2.* Efft1;
DG = ifft2(DG0);

DGlog = log(DG +1);
Gg1 = Glog - DGlog;


DG0 = Gfft2.* Efft2;
DG = ifft2(DG0);

DGlog = log(DG +1);
Gg2 = Glog - DGlog;


DG0 = Gfft2.* Efft3;
DG = ifft2(DG0);

DGlog = log(DG +1);
Gg3 = Glog - DGlog;

Gg = (Gg1 + Gg2 +Gg3)/3;

Ig = immultiply(G, a);
C = imdivide(Ig, II);
C = log(C+1);

Gg = immultiply(C, Gg);
EXPGg = exp(Gg);
MIN = min(min(EXPGg));
MAX = max(max(EXPGg));
EXPGg = (EXPGg - MIN)/(MAX - MIN);
EXPGg = adapthisteq(EXPGg);
%B通道处理
Blog = log(B+1);
Bfft2 = fft2(B);

DB0 = Bfft2.* Efft1;
DB = ifft2(DB0);

DBlog = log(DB +1);
Bb1 = Blog - DBlog;


DB0 = Bfft2.* Efft2;
DB = ifft2(DB0);

DBlog = log(DB +1);
Bb2 = Blog - DBlog;


DB0 = Bfft2.* Efft3;
DB = ifft2(DB0);

DBlog = log(DB +1);
Bb3 = Blog - DBlog;

Bb = (Bb1 + Bb2 +Bb3)/3;

Ib = immultiply(B, a);
C = imdivide(Ib, II);
C = log(C+1);

Bb = immultiply(C, Bb);
EXPBb = exp(Bb);
MIN = min(min(EXPBb));
MAX = max(max(EXPBb));
EXPBb = (EXPBb - MIN)/(MAX - MIN);
EXPBb = adapthisteq(EXPBb);

result = cat(3, EXPRr, EXPGg, EXPBb);  

% toc;
% ssimval = ssim(result,I)
% figure();
% imshow(I),title('原图');
% figure();
% imshow(result),title('增强图');
% imwrite(result,'C:\Users\shou\Desktop\061to.jpg')