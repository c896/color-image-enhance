% clc
clear 
close all

zu=9;
[DIR,path,xlspath]=Zu(zu);

imgFiles = dir([DIR,'*.bmp']);%����ͼ��ĸ�ʽ  dir('')�г�ָ��Ŀ¼���������ļ��к��ļ�
tic

% [N, ~]= size(imgFiles);
N=200;
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

figure;imshow(I),title('ԭ���ն�ͼ֡ƽ��');
imwrite(img,[path,'ԭ���ն�֡ƽ��480-640','.jpg']);

% mean2(uint8(round(img*255))) % mean2(img)*255
% img0=imread([path,'ԭ���ն�֡ƽ��480-640','.jpg']);
% mean2(img0)
%%
%------------------power٤����ǿ-------------%
Iomean=0.4; %90/255
% img_mean=mean2(img);
img_mean=sum(median(median(img)))/3;
p=log(Iomean)/log(img_mean)+0.005;
imp=power(img,p); %pmean=0.3184;

time=toc
figure,imshow(imp)
imp1=uint8(imp*255);
quality=testquality(imp1,0);
imwrite(imp1,[path,'֡ƽ��power480-640.jpg']);
% % qualityscore = SSEQ(imp)
% imgpwer=imread([path,'֡ƽ��power480-640.jpg']); 
% ssimval = ssim(imp1,imgpwer)

gray = rgb2gray(imp1);

figure(),imhist(gray)
axis tight;
set(gca,'FontSize',11); 
% set(gca,'YTickLabel',{'0','1','2','3','4','5','6','7','8','9'},'FontSize',15) %����Y������̶ȴ���ʾ���ַ�
set(gca,'LooseInset',get(gca,'TightInset'))
set(gca,'looseInset',[0 0 0 0])

