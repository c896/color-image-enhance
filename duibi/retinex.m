function imp1=retinex(DIR,path,xlspath,zu)
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

% figure;imshow(I),title('原低照度图帧平均');
% imwrite(img,[path,'原低照度帧平均480-640','.jpg']);
% mean2(uint8(round(img*255))) % mean2(img)*255
% img0=imread([path,'原低照度帧平均480-640','.jpg']);
% mean2(img0)
%%
% ------------------retinex SSR MSR MSRCR伽马增强-------------%
method=3;
switch method 
    case 1
        imp=SSR(img,80);
%         imwrite(imp,[path,'帧平均SSR480-640','.jpg']);
    case 2
        imp=MSR(img,10,80,200);
%         imwrite(imp,[path,'帧平均MSR480-640','.jpg']);
    case 3
        imp=MSRCR(img,10,80,200);
%         imwrite(imp,[path,'帧平均MSRCR480-640','.jpg']);
end

time=toc
figure,imshowpair(img,imp,'montage')
imp1=uint8(imp*255);

gray = rgb2gray(imp1);
figure(2),imhist(gray)
axis tight;
set(gca,'LooseInset',get(gca,'TightInset'))
set(gca,'looseInset',[0 0 0 0])


switch method
    case 1
        imwrite(imp1,[path,'帧平均SSR480-640-1','.jpg']);
%         print(gcf, '-djpeg', '-r300', [path 'SSR-1.bmp']);
    case 2
        imwrite(imp1,[path,'帧平均MSR480-640-1','.jpg']);
%         print(gcf, '-djpeg', '-r300', [path 'MSR-1.bmp']);
    case 3
        imwrite(imp1,[path,'帧平均MSRCR480-640-1','.jpg']);
%         print(gcf, '-djpeg', '-r300', [path 'MSRCR-1.bmp']);
end

quality1=testquality(imp1,1);

value=[quality1 time];

% ssimval = ssim(imp1,imgpwer);
% Psnr=psnr(imp1,imgpwer);  
% str=sprintf('ssim:%.4f psnr:%.4f',ssimval,Psnr);  %输出类型为char
% display(str)

% value=[quality1 ssimval Psnr time];
value=roundn(value,-4);  %保留几位小数 b=vpa(value,4)保留有效数字

mRowRange='R3:W3';
writematrix(value,xlspath,'Sheet',1,'Range',mRowRange) % xlswrite('D:\学习\小论文\data.xlsx',value,1,mRowRange);

