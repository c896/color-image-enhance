function rgb1=adaptgamma(DIR,path,xlspath,zu)
% % clc
% clear 
% close all

sprintf('当前第%d组，path:%s',zu,DIR)
% imgpwer=imread([path,'帧平均power480-640.jpg']);
if zu<5
    imgFiles = dir([DIR,'*.jpg']);%输入图像的格式  dir('')列出指定目录下所有子文件夹和文件
else
    imgFiles = dir([DIR,'*.bmp']);
end
% [N, ~]= size(imgFiles);
N=200;
%%
tic

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
% figure;imshow(I),title('原低照度图帧平均');
% imwrite(img,[path,'原低照度帧平均480-640','.jpg']);
% % mean2(uint8(round(img*255))) % mean2(img)*255
% % img0=imread([path,'原低照度帧平均480-640','.jpg']);
% % mean2(img0)

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

rgb1=uint8(rgb*255);
% figure,imshow(rgb1);title('原版校正结果')
figure,imshowpair(img,rgb1,'montage')
%%
imwrite(rgb1,[path,'帧平均adaptgamma校正480-640-1','.jpg']);

gray = rgb2gray(rgb1);
figure(2),imhist(gray)
axis tight;
set(gca,'LooseInset',get(gca,'TightInset'))
set(gca,'looseInset',[0 0 0 0])

quality=testquality(rgb1,1);

% ssimval = ssim(rgb1,imgpwer);
% Psnr=psnr(rgb1,imgpwer);  
% str=sprintf('ssim:%.4f psnr:%.4f',ssimval,Psnr);  %输出类型为char
% display(str)

% value=[quality ssimval Psnr time];
value=[quality time];
value=roundn(value,-4);  %保留几位小数 b=vpa(value,4)保留有效数字

mRowRange='R4:W4';
writematrix(value,xlspath,'Sheet',1,'Range',mRowRange) % xlswrite('D:\学习\小论文\data.xlsx',value,1,mRowRange);


