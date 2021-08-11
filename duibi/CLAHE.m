% clc
clear 
close all
%第一组
% DIR='D:\Cammera\10.09\2-480-640\'; % 均值2 0.0096 2.4502
% path='D:\Cammera\10.09\2result\';
%第二组
DIR='D:\Cammera\9.28\9.28-2-480-640\'; %均值1 0.0179 4.5683 
path='D:\Cammera\9.28\9.28-2result\';
%第三组
% DIR='D:\Cammera\7.02\1\480-640\'; %均值0.0388  9.886
% path='D:\Cammera\7.02\1result\';
%第四组
% DIR='D:\Cammera\10.09\1-480-640\'; %均值1 0.0524 13.3742  
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
% 
% % figure;imshow(I),title('原低照度图帧平均');
% imwrite(img,[path,'原低照度帧平均480-640','.jpg']);
% % mean2(uint8(round(img*255))) % mean2(img)*255
% % img0=imread([path,'原低照度帧平均480-640','.jpg']);
% % mean2(img0)
%%
%-------------------自适应直方图均衡化--CLAHE----------------%
rimg = img(:,:,1);
gimg = img(:,:,2);
bimg = img(:,:,3);
% resultr = adapthisteq(rimg,'ClipLimit',0.04);
resultr = adapthisteq(rimg,'Distribution','exponential');%,'ClipLimit',0.01
resultg = adapthisteq(gimg,'Distribution','exponential');
resultb = adapthisteq(bimg,'Distribution','exponential');
result = cat(3, resultr, resultg, resultb);

time=toc

% imhist(result)
figure,imshowpair(img,result,'montage')
result1=uint8(result*255);

imwrite(img,[path,'帧平均CLAHE480-640-1','.jpg']);
%%
gray = rgb2gray(result1);
figure(2),imhist(gray)
axis tight;
set(gca,'LooseInset',get(gca,'TightInset'))
set(gca,'looseInset',[0 0 0 0])

quality1=testquality(result1,0);
imgpwer=imread([path,'帧平均power480-640.jpg']); 
ssimval = ssim(result1,imgpwer)
