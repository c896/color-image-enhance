function result=MSR(I,sigma1,sigma2,sigma3)

% [filename,pathname]=uigetfile('*.*','�ں�ͼ��');
% I=imread([pathname,filename]);
% img = imread('IMG_3651.jpg');
% f = 512 / max(size(img));
% I = imresize(img,f);
% tic

R = I(:, :, 1);
G = I(:, :, 2);
B = I(:, :, 3);
R0 = double(R);
G0 = double(G);
B0 = double(B);

[N1, M1] = size(R0);

Rlog = log(R0+1);%��ǿԭ���һ��ȡ����
Rfft2 = fft2(R0);%���ж�ά����Ҷ�任����Rͨ�������ɿռ����ΪƵ����

% sigma1 = 1800;
F1 = fspecial('gaussian', [N1,M1], sigma1);%��˹��ͨ�˲���������������hsize��ʾģ��ߴ磬Ĭ��ֵΪ��3 3����sigmaΪ�˲����ı�׼ֵ����λΪ���أ�Ĭ��ֵΪ0.5.
%F1 = fspecial('unsharp', 0.2);                            F1Ϊһ����ά�˲��� 
Efft1 = fft2(double(F1));%���и���Ҷ�任�����˲���F1�ɿռ����ΪƵ����

%�ڶ�����˹ģ���ԭͼ������������൱�ڶ�ԭͼ������ͨ�˲����õ���ͨ�˲����ͼ��D(x,y)
DR0 = Rfft2.* Efft1; %R�������˲������е��
DR = ifft2(DR0);%���и���Ҷ���任����Ƶ�����Ϊ�ռ���

%�������ڶ������У���ԭͼ���ȥ��ͨ�˲����ͼ�񣬵õ���Ƶ��ǿ��ͼ��G(x,y)=S��(x,y)-log(D(x,y))
DRlog = log(DR +1);
Rr1 = Rlog - DRlog;

%sigma2 = 2500;
F2 = fspecial('gaussian', [N1,M1], sigma2);
Efft2 = fft2(double(F2));

DR0 = Rfft2.* Efft2;
DR = ifft2(DR0);

DRlog = log(DR +1);
Rr2 = Rlog - DRlog;

% sigma3 = 5200;
%sigma3 = 5100;
F3 = fspecial('gaussian', [N1,M1], sigma3);
Efft3 = fft2(double(F3));

DR0 = Rfft2.* Efft3;
DR = ifft2(DR0);

DRlog = log(DR +1);
Rr3 = Rlog - DRlog;

Rr = (Rr1 + Rr2 +Rr3)/3;

EXPRr = exp(Rr);
MIN = min(min(EXPRr));
MAX = max(max(EXPRr));
EXPRr = (EXPRr - MIN)/(MAX - MIN);
EXPRr=real(EXPRr);%ȡ����Ҷ�任��ʵ��
EXPRr = adapthisteq(EXPRr);
% result = EXPRr;
% Gͨ������
Glog = log(G0+1);
Gfft2 = fft2(G0);

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

EXPGg = exp(Gg);
MIN = min(min(EXPGg));
MAX = max(max(EXPGg));
EXPGg = (EXPGg - MIN)/(MAX - MIN);
EXPGg = adapthisteq(EXPGg);
% %Bͨ������
Blog = log(B0+1);
Bfft2 = fft2(B0);

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

EXPBb = exp(Bb);
MIN = min(min(EXPBb));
MAX = max(max(EXPBb));
EXPBb = (EXPBb - MIN)/(MAX - MIN);
EXPBb = adapthisteq(EXPBb);
result = cat(3, EXPRr, EXPGg, EXPBb);  

% toc
% ssimval = ssim(result,im2double(I))
% figure();
% imshow(I),title('ԭͼ');
% figure();
% imshow(result),title('��ǿͼ');
% imwrite(result,'C:\Users\shou\Desktop\original\MSR.jpg')
