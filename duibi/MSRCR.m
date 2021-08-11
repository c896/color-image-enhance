%---------��ɫ�ʻָ��Ķ�߶�---------------%
function result=MSRCR(img,sigma1,sigma2,sigma3)
% [filename,pathname]=uigetfile('*.*','�ں�ͼ��');
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

Rlog = log(R+1);%��ǿԭ���һ��ȡ����
Rfft2 = fft2(R);%���ж�ά����Ҷ�任����Rͨ�������ɿռ����ΪƵ����


%sigma1 = 128;
% sigma1 = 512;
F1 = fspecial('gaussian', [N1,M1], sigma1);%��˹��ͨ�˲���������������hsize��ʾģ��ߴ磬Ĭ��ֵΪ��3 3����sigmaΪ�˲����ı�׼ֵ����λΪ���أ�Ĭ��ֵΪ0.5.
%F1 = fspecial('unsharp', 0.2);                            F1Ϊһ����ά�˲��� 
Efft1 = fft2(double(F1));%���и���Ҷ�任�����˲���F1�ɿռ����ΪƵ����

%�ڶ�����˹ģ���ԭͼ������������൱�ڶ�ԭͼ������ͨ�˲����õ���ͨ�˲����ͼ��D(x,y)
DR0 = Rfft2.* Efft1; %R�������˲������е��
DR = ifft2(DR0);%���и���Ҷ���任����Ƶ�����Ϊ�ռ���

%�������ڶ������У���ԭͼ���ȥ��ͨ�˲����ͼ�񣬵õ���Ƶ��ǿ��ͼ��G(x,y)=S��(x,y)-log(D(x,y))
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
%�����ɫ�ָ�����C
C = imdivide(Ir, II);
C = log(C+1);
% ����ǿ�ķ�������ɫ�ʻָ�����
Rr = immultiply(C, Rr);
%���Ĳ���G(x,y)ȡ���������õ���ǿ���ͼ��R(x,y)=exp(G(x,y))
EXPRr = exp(Rr);
%���岽��R(x,y)���Աȶ���ǿ���õ����յĽ��ͼ��
MIN = min(min(EXPRr));
MAX = max(max(EXPRr));
EXPRr = (EXPRr - MIN)/(MAX - MIN);
EXPRr = adapthisteq(EXPRr);
% Gͨ������
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
%Bͨ������
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
% imshow(I),title('ԭͼ');
% figure();
% imshow(result),title('��ǿͼ');
% imwrite(result,'C:\Users\shou\Desktop\061to.jpg')