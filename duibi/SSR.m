function result=SSR(I,sigma)
% clc
% clear
% tic
% % [filename,pathname]=uigetfile('*.*','�ں�ͼ��');
% % I=im2double(imread([pathname,filename]));
% I=im2double(imread('C:\Users\CG\Desktop\5.28\TU\���ն�\horse rider.PNG'));
% sigma = 200;

% img = imread('IMG_3651.jpg');
% f = 512 / max(size(img));
% I = imresize(img,f);
%%
R = I(:, :, 1);
[N1, M1] = size(R);
R0 = double(R);
Rlog = log(R0+1);
Rfft2 = fft2(R0);
 

F = fspecial('gaussian', [N1,M1], sigma);
Efft = fft2(double(F));

DR0 = Rfft2.* Efft;
DR = ifft2(DR0);

DRlog = log(DR +1);
Rr = Rlog - DRlog;
EXPRr = exp(Rr);
MIN = min(min(EXPRr));
MAX = max(max(EXPRr));
EXPRr = (EXPRr - MIN)/(MAX - MIN);
EXPRr = adapthisteq(EXPRr);

G = I(:, :, 2);

G0 = double(G);
Glog = log(G0+1);
Gfft2 = fft2(G0);
  
DG0 = Gfft2.* Efft;
DG = ifft2(DG0);

DGlog = log(DG +1);
Gg = Glog - DGlog;
EXPGg = exp(Gg);
MIN = min(min(EXPGg));
MAX = max(max(EXPGg));
EXPGg = (EXPGg - MIN)/(MAX - MIN);
EXPGg = adapthisteq(EXPGg);

B = I(:, :, 3);

B0 = double(B);
Blog = log(B0+1);
Bfft2 = fft2(B0);
  
DB0 = Bfft2.* Efft;
DB = ifft2(DB0);

DBlog = log(DB+1);
Bb = Blog - DBlog;
EXPBb = exp(Bb);
MIN = min(min(EXPBb));
MAX = max(max(EXPBb));
EXPBb = (EXPBb - MIN)/(MAX - MIN);
EXPBb = adapthisteq(EXPBb);

result = cat(3, EXPRr, EXPGg, EXPBb);

% toc
% 
% ssimval = ssim(result,im2double(I))
% figure();
% imshow(I),title('ԭͼ');
% figure();
% imshow([I result]),title('ȥ��ͼ');