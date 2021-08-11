% clc
clear 
close all

zu=9;
[DIR,path,xlspath]=Zu(zu);

imgFiles = dir([DIR,'*.bmp']);%输入图像的格式  dir('')列出指定目录下所有子文件夹和文件
tic

% [N, ~]= size(imgFiles);
N=200;
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

figure;imshow(I),title('原低照度图帧平均');
imwrite(img,[path,'原低照度帧平均480-640','.jpg']);

% mean2(uint8(round(img*255))) % mean2(img)*255
% img0=imread([path,'原低照度帧平均480-640','.jpg']);
% mean2(img0)
%%
%------------------power伽马增强-------------%
Iomean=0.4; %90/255
% img_mean=mean2(img);
img_mean=sum(median(median(img)))/3;
p=log(Iomean)/log(img_mean)+0.005;
imp=power(img,p); %pmean=0.3184;

time=toc
figure,imshow(imp)
imp1=uint8(imp*255);
quality=testquality(imp1,0);
imwrite(imp1,[path,'帧平均power480-640.jpg']);
% % qualityscore = SSEQ(imp)
% imgpwer=imread([path,'帧平均power480-640.jpg']); 
% ssimval = ssim(imp1,imgpwer)

gray = rgb2gray(imp1);

figure(),imhist(gray)
axis tight;
set(gca,'FontSize',11); 
% set(gca,'YTickLabel',{'0','1','2','3','4','5','6','7','8','9'},'FontSize',15) %设置Y坐标轴刻度处显示的字符
set(gca,'LooseInset',get(gca,'TightInset'))
set(gca,'looseInset',[0 0 0 0])

