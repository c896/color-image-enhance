% img=imread('D:\Cammera\10.09\1\Pic_2020_10_09_180507_722.bmp');
% mean2(img)
%----------------------------retinex���ն�ͼ����ǿ-֡ƽ��-----------------------%
% clc
clear 
close all
% path='G:\��Ƭ\6.16\';
% DIR='G:\��Ƭ\6.16\576-704\';

% DIR='D:\Cammera\10.09\2-480-640\'; 
% path='D:\Cammera\10.09\2result\';

DIR='D:\Cammera\7.02\1\480-640\';
path='D:\Cammera\7.02\';

% DIR='D:\Cammera\9.28\9.28-2-480-640\'; %��ֵ1 0.0214 5.457  ��ֵ2 0.0178 4.539
% path='D:\Cammera\9.28\';

% path='D:\Cammera\7.17\';
% DIR='D:\Cammera\7.17\7.17-480-640\';

% DIR='D:\Cammera\10.09\1-480-640\'; 
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

% figure;imshow(I),title('ԭ���ն�ͼ֡ƽ��');
imwrite(img,[path,'ԭ���ն�֡ƽ��480-640','.jpg']);
% mean2(uint8(round(img*255))) % mean2(img)*255
% img0=imread([path,'ԭ���ն�֡ƽ��480-640','.jpg']);
% mean2(img0)
%%
%------------------power٤����ǿ-------------%
% Iomean=0.4; %90/255
% % img_mean=mean2(img);
% img_mean=sum(median(median(img)))/3;
% p=log(Iomean)/log(img_mean)+0.005;
% imp=power(img,p); %pmean=0.3184;
% 
% time=toc
% figure,imshow(imp)
% quality=testquality(uint8(round(imp*255)),0);
% % imwrite(imp,[path,'֡ƽ��power480-640.jpg']);
% % % qualityscore = SSEQ(imp)
% 
% % quality_orig=testquality(I);
% % qualityscore = SSEQ(y)  % 44.7349
%%
%------------------retinex SSR MSR MSRCR٤����ǿ-------------%
% method=3;
% switch method 
%     case 1
%         imp=SSR(img,80);
%         imwrite(imp,[path,'֡ƽ��SSR480-640','.jpg']);
%     case 2
%         imp=MSR(img,10,80,200);
%         imwrite(imp,[path,'֡ƽ��MSR480-640','.jpg']);
%     case 3
%         imp=MSRCR(img,10,80,200);
%         imwrite(imp,[path,'֡ƽ��MSRCR480-640','.jpg']);
% end
% 
% time=toc
% figure,imshowpair(img,imp,'montage')
% quality1=testquality(uint8(round(imp*255)),0);
% 
% imgpwer=im2double(imread([path,'֡ƽ��power480-640.jpg'])); 
% ssimval = ssim(imp,imgpwer);
% str=sprintf('ssim:%.4f',ssimval);  %�������Ϊchar
% display(str)
% % ssimval = ssim(img,imp)

%%
% %----------------retinex��ǿ------------------%
% %     f = 512 / max(size(I));
% %     I = imresize(I,f);
% %     I(I < 0) = 0;I(I > 1) = 1;
% 
% % a=0.01;b=0.2;c=1; 
% % icount=8;
% a=0.008;b=0.3;c=1;
% icount=4;
% step=2.4;%decide the times  %step=2.2;
% 
% I=img+1;
% Z=log2(I); %Z=log(I);
% Z=Z/max(Z(:));
% R=zeros(size(Z));
% % R1=zeros(size(Z));
% nn = 1;
% for method=1:1:nn
%     for i=1:3
%         if method==1            %zm�㷨
%             [R(:,:,i),L]=zm_retinex_Light1(Z(:,:,i),a,b,c,icount,0);
%         elseif method==2        %mccann�㷨
%             R(:,:,i)=zm_retinex_mccann992(Z(:,:,i),icount);
%         elseif method==3        %Kimmel�㷨����1
% %             R(:,:,i)=zm_Kimmel(Z(:,:,i),a,b,icount,4.5); %���һ��������Ч
%             R(:,:,i)=zm_Kimmel(Z(:,:,i),a,b,1,4.5); %���һ��������Ч
%         elseif method==4        %Kimmel�㷨����2
%             R(:,:,i)=Z(:,:,i)-zm_Devir_retinex(Z(:,:,i),a,b);
%         end
%     end
%     
%     m=mean2(R);s=std(R(:));
%     mini=max(m-step*s,min(R(:)));maxi=min(m+step*s,max(R(:)));
%     range=maxi-mini;
%     result=(R-mini)/range*0.8;
%     %result=max(result(:))-result;  
%     
% %     sprintf('���ڴ���%s',imgFiles(1).name)
% %     imwrite(result,[path,num2str(method),imgFiles(1).name]);
% %     imwrite(result,[path,'֡ƽ��512retinex_Light','.jpg']);
% %     figure,imshow(result); title(['ȥ���RGB��ǿͼ',num2str(method)]);
% end
% time1=toc
% figure,imshowpair(img,result,'montage')
% imwrite(result,[path,'֡ƽ��zm_retinex_Light480-640.jpg']);
% 
% quality=testquality(uint8(round(result*255)),0);
% imgpwer=im2double(imread([path,'֡ƽ��power480-640.jpg'])); 
% ssimval = ssim(result,imgpwer)

%%
% %---------------ֱ��ͼ����--HE------------------%
% S=img;
% Ir=histeq(S(:, :, 1),64);%n - ��ɢ�Ҷȼ�������64 ��Ĭ�ϣ�
% Ig=histeq(S(:, :, 2),64);
% Ib=histeq(S(:, :, 3),64);
% J = cat(3, Ir,Ig,Ib);
% % imwrite(J,'C:\Users\shou\Desktop\new\1zeng.jpg');
% % transforms the intensity image I,returning J an intensity
% time=toc
% imhist(J)
% figure,imshowpair(S,J,'montage')
% imwrite(J,[path,'֡ƽ��HE480-640.jpg']);
% 
% quality=testquality(uint8(round(J*255)),0);
% imgpwer=im2double(imread([path,'֡ƽ��power480-640.jpg'])); 
% ssimval = ssim(J,imgpwer)
%%
%-------------------����Ӧֱ��ͼ���⻯--CLAHE----------------%
% rimg = img(:,:,1);
% gimg = img(:,:,2);
% bimg = img(:,:,3);
% % resultr = adapthisteq(rimg,'ClipLimit',0.04);
% resultr = adapthisteq(rimg,'Distribution','exponential');%,'ClipLimit',0.01
% resultg = adapthisteq(gimg,'Distribution','exponential');
% resultb = adapthisteq(bimg,'Distribution','exponential');
% result = cat(3, resultr, resultg, resultb);
% 
% time=toc
% imhist(result)
% figure,imshowpair(img,result,'montage')
% imwrite(result,[path,'֡ƽ��HE480-640.jpg']);
% 
% quality1=testquality(uint8(round(result*255)),0);
% imgpwer=im2double(imread([path,'֡ƽ��power480-640.jpg'])); 
% ssimval = ssim(result,imgpwer)
%%
%---------------------LSCN---------------%
% imp=power(img,0.5);
% I1=uint8(imp*255);
% 
% V = rgb2v(I);
% V_flm = LSCN(V);
% [Contrast(k,1), Spatial_frequency(k,1), Gradient(k,1)] ...
%     = QEvaluation(V_flm);
% JND(k,1) = JND_zhan2(V_flm);
% result=v2rgb(I,V_flm);
% 
% time=toc
% figure,imshow([I result]);
% imwrite(result,[path,'֡ƽ��LSCN480-640.jpg']);
% 
% % display([Contrast, Spatial_frequency, Gradient JND])
% quality1=testquality(result,0);
% 
% imgpwer=imread([path,'֡ƽ��power480-640.jpg']); 
% ssimval = ssim(result,imgpwer)

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

figure,imshow(rgb);title('ԭ��У�����')
imwrite(rgb,[path,'֡ƽ��SVLM480-640.jpg']);
quality=testquality(uint8(round(rgb*255)),0);

imgpwer=im2double(imread([path,'֡ƽ��power480-640.jpg'])); 
ssimval = ssim(rgb,imgpwer)
