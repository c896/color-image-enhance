% img=imread('D:\Cammera\10.09\1\Pic_2020_10_09_180507_722.bmp');
% mean2(img)
%----------------------------retinex低照度图像增强-帧平均-----------------------%
% clc
clear 
close all

DIR='D:\Cammera\mode\modenoise2\'; %均值0.0771  19.6605
path='D:\Cammera\mode\result2\';
imgpwer=imread([path,'House256.jpg']); 

% path='G:\照片\6.16\';
% DIR='G:\照片\6.16\576-704\';

% DIR='D:\Cammera\10.09\2-480-640\'; 
% path='D:\Cammera\10.09\2result\';

% DIR='D:\Cammera\7.02\1\480-640\';
% path='D:\Cammera\7.02\';

% DIR='D:\Cammera\9.28\9.28-2-480-640\'; %均值1 0.0214 5.457  均值2 0.0178 4.539
% path='D:\Cammera\9.28\';

% path='D:\Cammera\7.17\';
% DIR='D:\Cammera\7.17\7.17-480-640\';

% DIR='D:\Cammera\10.09\1-480-640\'; 
% path='D:\Cammera\10.09\1result\';
%%
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
%-------------SVLM--------------------%
% Image = uint8(round(power(img,0.5)*255));
Image = uint8(round(img*255));
[row,col,dim]=size(Image);
Outimage=zeros(row,col,dim);
E=ones(row,col);
Image_gray = double(rgb2gray(Image));
%Image_gray = double(Image);
Image_norm = Image_gray/255;
diet=mean(sqrt(diag(cov(Image_gray))));
mmm=mean2(Image_gray);
for i = 1:row
    for j = 1:col
        sum =(Image_gray(i,j)-mmm)^2;
    end
end
dddd= sum/(row*col);
    % diet=diet^2;
if(diet<=40)
    p=2;
elseif(40<diet<=80)
    p=3-0.025*diet;
elseif(diet>80)
    p=1;
end

h1 = fspecial('gaussian',[9 9]);
h2 = fspecial('gaussian',[7 7]);
h3 = fspecial('gaussian',[5 5]);
h4 = fspecial('gaussian',[3 3]);
Image_gray_trans1 = imfilter(Image_gray,h1);
Image_gray_trans2 = imfilter(Image_gray,h2);
Image_gray_trans3 = imfilter(Image_gray,h3);
Image_gray_trans4 = imfilter(Image_gray,h4);
tic
Image_gray_trans = (Image_gray_trans1 + Image_gray_trans2 + Image_gray_trans3 + Image_gray_trans4)/4;%SVLM

% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%SVLM方法%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i = 1:row
    for j = 1:col
            Image_ratio(i,j) = 0.5^((128-Image_gray_trans(i,j))/128);
            O(i,j) = 255 * (Image_gray(i,j)/255)^Image_ratio(i,j);
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:row
    for j = 1:col
        if(O(i,j))
            E(i,j)=(Image_gray_trans(i,j)/O(i,j))^p;
            S(i,j)=255*(O(i,j)/255)^E(i,j);
        else
            E(i,j) = 0;
            S(i,j) = 0;
        end

    end
end

for i = 1:row
    for j = 1:col
        if(O(i,j))
            Outimage(i,j,1)= S(i,j) * (double(Image(i,j,1))/O(i,j));
            Outimage(i,j,2)= S(i,j) * (double(Image(i,j,2))/O(i,j));
            Outimage(i,j,3)= S(i,j) * (double(Image(i,j,3))/O(i,j));
        else
            Outimage(i,j,:) = 0;
        end

    end
end

time=toc
figure,imshow(uint8(Outimage));
imwrite(uint8(Outimage),[path,'帧平均SVLM480-640.jpg']);
% ssimval = ssim(uint8(Outimage),Image)
quality1=testquality(uint8(Outimage),1);

% imgpwer=imread([path,'帧平均power480-640.jpg']); 
ssimval = ssim(uint8(Outimage),imgpwer)
