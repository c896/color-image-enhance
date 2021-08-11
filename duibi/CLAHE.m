% clc
clear 
close all
%��һ��
% DIR='D:\Cammera\10.09\2-480-640\'; % ��ֵ2 0.0096 2.4502
% path='D:\Cammera\10.09\2result\';
%�ڶ���
DIR='D:\Cammera\9.28\9.28-2-480-640\'; %��ֵ1 0.0179 4.5683 
path='D:\Cammera\9.28\9.28-2result\';
%������
% DIR='D:\Cammera\7.02\1\480-640\'; %��ֵ0.0388  9.886
% path='D:\Cammera\7.02\1result\';
%������
% DIR='D:\Cammera\10.09\1-480-640\'; %��ֵ1 0.0524 13.3742  
% path='D:\Cammera\10.09\1result\';

imgFiles = dir([DIR,'*.jpg']);%����ͼ��ĸ�ʽ  dir('')�г�ָ��Ŀ¼���������ļ��к��ļ�
tic

[N, ~]= size(imgFiles);
I=imread([DIR ,imgFiles(1).name]);
img=im2double(I);
% figure,imshow(img),title('ԭ����ͼ')

% A=zeros(size(img));
for k=2:N
    A=im2double(imread([DIR ,imgFiles(k).name]));
    img=img+A;
%     sprintf('���ڶ�ȡ%s',imgFiles(k).name)
end
img=img./N;
% 
% % figure;imshow(I),title('ԭ���ն�ͼ֡ƽ��');
% imwrite(img,[path,'ԭ���ն�֡ƽ��480-640','.jpg']);
% % mean2(uint8(round(img*255))) % mean2(img)*255
% % img0=imread([path,'ԭ���ն�֡ƽ��480-640','.jpg']);
% % mean2(img0)
%%
%-------------------����Ӧֱ��ͼ���⻯--CLAHE----------------%
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

imwrite(img,[path,'֡ƽ��CLAHE480-640-1','.jpg']);
%%
gray = rgb2gray(result1);
figure(2),imhist(gray)
axis tight;
set(gca,'LooseInset',get(gca,'TightInset'))
set(gca,'looseInset',[0 0 0 0])

quality1=testquality(result1,0);
imgpwer=imread([path,'֡ƽ��power480-640.jpg']); 
ssimval = ssim(result1,imgpwer)
