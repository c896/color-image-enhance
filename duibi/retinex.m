function imp1=retinex(DIR,path,xlspath,zu)
% % clc
% clear 
% close all
sprintf('��ǰ��%d�飬path:%s',zu,DIR)
% imgpwer=imread([path,'֡ƽ��power480-640.jpg']); 
if zu<5
    imgFiles = dir([DIR,'*.jpg']);%����ͼ��ĸ�ʽ  dir('')�г�ָ��Ŀ¼���������ļ��к��ļ�
else
    imgFiles = dir([DIR,'*.bmp']);
end
% [N, ~]= size(imgFiles);
N=200;
tic

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

% figure;imshow(I),title('ԭ���ն�ͼ֡ƽ��');
% imwrite(img,[path,'ԭ���ն�֡ƽ��480-640','.jpg']);
% mean2(uint8(round(img*255))) % mean2(img)*255
% img0=imread([path,'ԭ���ն�֡ƽ��480-640','.jpg']);
% mean2(img0)
%%
% ------------------retinex SSR MSR MSRCR٤����ǿ-------------%
method=3;
switch method 
    case 1
        imp=SSR(img,80);
%         imwrite(imp,[path,'֡ƽ��SSR480-640','.jpg']);
    case 2
        imp=MSR(img,10,80,200);
%         imwrite(imp,[path,'֡ƽ��MSR480-640','.jpg']);
    case 3
        imp=MSRCR(img,10,80,200);
%         imwrite(imp,[path,'֡ƽ��MSRCR480-640','.jpg']);
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
        imwrite(imp1,[path,'֡ƽ��SSR480-640-1','.jpg']);
%         print(gcf, '-djpeg', '-r300', [path 'SSR-1.bmp']);
    case 2
        imwrite(imp1,[path,'֡ƽ��MSR480-640-1','.jpg']);
%         print(gcf, '-djpeg', '-r300', [path 'MSR-1.bmp']);
    case 3
        imwrite(imp1,[path,'֡ƽ��MSRCR480-640-1','.jpg']);
%         print(gcf, '-djpeg', '-r300', [path 'MSRCR-1.bmp']);
end

quality1=testquality(imp1,1);

value=[quality1 time];

% ssimval = ssim(imp1,imgpwer);
% Psnr=psnr(imp1,imgpwer);  
% str=sprintf('ssim:%.4f psnr:%.4f',ssimval,Psnr);  %�������Ϊchar
% display(str)

% value=[quality1 ssimval Psnr time];
value=roundn(value,-4);  %������λС�� b=vpa(value,4)������Ч����

mRowRange='R3:W3';
writematrix(value,xlspath,'Sheet',1,'Range',mRowRange) % xlswrite('D:\ѧϰ\С����\data.xlsx',value,1,mRowRange);

