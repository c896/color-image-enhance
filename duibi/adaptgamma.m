function rgb1=adaptgamma(DIR,path,xlspath,zu)
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
%%
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
% 
% figure;imshow(I),title('ԭ���ն�ͼ֡ƽ��');
% imwrite(img,[path,'ԭ���ն�֡ƽ��480-640','.jpg']);
% % mean2(uint8(round(img*255))) % mean2(img)*255
% % img0=imread([path,'ԭ���ն�֡ƽ��480-640','.jpg']);
% % mean2(img0)

%%
% ��˹�˲�
[h,s,v]=rgb2hsv(img); 
HSIZE= min(size(img,1),size(img,2));%��˹����˳ߴ�
q=sqrt(2);
SIGMA1=15;%���������c
SIGMA2=80;
SIGMA3=250;
F1 = fspecial('gaussian',HSIZE,SIGMA1/q);
F2 = fspecial('gaussian',HSIZE,SIGMA2/q) ;
F3 = fspecial('gaussian',HSIZE,SIGMA3/q) ;
gaus1= imfilter(v, F1, 'replicate');
gaus2= imfilter(v, F2, 'replicate');
gaus3= imfilter(v, F3, 'replicate');
gaus=(gaus1+gaus2+gaus3)/3;    %��߶ȸ�˹�������Ȩ��Ȩ��Ϊ1/3
% gaus=(gaus*255);
% figure,imshow(gaus,[]);title('gaus���շ���');

% --------------------ԭ���㷨��˹���ڶ�ά٤��-------------------%
Imean=0.5;   %mean2(v)  level;  0.5
% p=log(Iomean)/log(Imean).*0.5;
% p1=(gaus-Imean)/Imean;
p1=(Imean-gaus)/Imean; %ԭ
% gama=(ones(size(p1))*0.5).^p1;
gama=power(Imean,p1);%���ݹ�ʽgammaУ���������Ĺ�ʽ����
vout=power(v,gama);
rgb=hsv2rgb(h,s,vout);   %ת��rgb�ռ���ʾ

time=toc

rgb1=uint8(rgb*255);
% figure,imshow(rgb1);title('ԭ��У�����')
figure,imshowpair(img,rgb1,'montage')
%%
imwrite(rgb1,[path,'֡ƽ��adaptgammaУ��480-640-1','.jpg']);

gray = rgb2gray(rgb1);
figure(2),imhist(gray)
axis tight;
set(gca,'LooseInset',get(gca,'TightInset'))
set(gca,'looseInset',[0 0 0 0])

quality=testquality(rgb1,1);

% ssimval = ssim(rgb1,imgpwer);
% Psnr=psnr(rgb1,imgpwer);  
% str=sprintf('ssim:%.4f psnr:%.4f',ssimval,Psnr);  %�������Ϊchar
% display(str)

% value=[quality ssimval Psnr time];
value=[quality time];
value=roundn(value,-4);  %������λС�� b=vpa(value,4)������Ч����

mRowRange='R4:W4';
writematrix(value,xlspath,'Sheet',1,'Range',mRowRange) % xlswrite('D:\ѧϰ\С����\data.xlsx',value,1,mRowRange);


