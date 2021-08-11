% function  imgout=wasobirbg(path1,DIR,imgFiles,img)
clc;clear all;close all;

path='C:\Users\CG\Desktop\5.28\result2\';
DIR='C:\Users\CG\Desktop\5.28\night\';%输入图片所在文件夹的路径
% DIR='C:\Users\CG\Desktop\5.28\noise\';
% DIR='C:\Users\CG\Desktop\5.28\videonoise\';
imgFiles = dir([DIR,'*.jpg']);%输入图像的格式  dir('')列出指定目录下所有子文件夹和文件

% I1 = im2double(imread('Lena512.png'));
% I2 = im2double(imread('Baboon512.png'));
I1 = im2double(imread([DIR , imgFiles(1).name]));
I2 = im2double(imread([DIR , imgFiles(4).name]));
figure(1);
subplot(121),imshow(I1); title('待分离噪图1');
subplot(122),imshow(I2);title('待分离噪图2');

tic;
[N, ~]= size(imgFiles);
[width,height]=size(I1(:,:,1));
% imgdata=zeros(3,width*height);
% img=power(img,0.4);
% imshow(img)

for i=1:3
    sample = zeros(N,width*height);
    %psnr_noise = 10*log10(peak^2/(mean((img11(:)-nimg1(:)).^2)));
    
    for j = 1:N
        I=im2double(imread([DIR ,imgFiles(j).name]));
        I=power(I0,0.4);
        sample(j,:) = reshape(I(:,:,i),1,[]);
    end
 
%     [Wwasobi, Winit, ISR,eval(['signals',num2str(i)])]= iwasobi(sample(),1,0.99);
    [~, ~, ~,signals]= iwasobi(sample(),1,0.99);
    eval(['signals',num2str(i),'=', 'signals',';']);   
end
% tic
for i=1:3
    for m=1:2
        %     time=toc
        img=eval(['I',num2str(m)]);
        imgrgb=img(:,:,i);
        [simg,nsimg]=signalsort(eval(['signals',num2str(i)]),imgrgb,N);
        if ssim(nsimg,imgrgb) > ssim(simg,imgrgb)
    %         eval(['simg',num2str(i),'=', 'nsimg1',';']);
            simg=nsimg;
    %         imgdata(i+3*(k-1),:)=G;
        end

        eval(['simg',num2str(m),num2str(i),'=', 'simg',';']);     
        psnr_denoise = psnr(simg,imgrgb) 
%         [PSNR, MSE] = psnr(simg,I2(:,:,3)) 
%         peak=2;
%         psnr_denoise = 10*log10(peak^2/mean((simg(:)-imgrgb(:)).^2));
        ssimval_denoise = ssim(simg,imgrgb)
        %     imwrite(simg,[path,'分离层',num2str(k),num2str(i),'.bmp']);
        imwrite(simg,[path,'分离层',num2str(m),num2str(i),'.jpg']);
        %     imwrite(imgr1,[path1,'分离层2',num2str(i),'.bmp']);
        
    end
end
time=toc
% imgout=imgcompound(imgdata,width,height);
% imgout1=imgcompound(imgdata1,width,height);
% figure,
% subplot(131),imshow(uint8(simg1),[]);subplot(132),imshow(simg2,[]);subplot(133),imshow(simg3,[]);
for n=1:2
    imgout=cat(3,eval(['simg',num2str(n),num2str(1)]),eval(['simg',num2str(n),num2str(2)]),eval(['simg',num2str(n),num2str(3)]));
    eval(['imgout',num2str(n),'=', 'imgout',';']);
end

figure,
subplot(121),imshow(imgout);title('合成分离去噪图1');
subplot(122),imshow(imgout1);title('合成分离去噪图2');
imwrite(imgout,[path,'合成分离去噪图1','.jpg']);
imwrite(imgout1,[path,'合成分离去噪图2','.jpg']);

% imgo=power(img2,0.3);
% figure,imshow(imgo)
% image= CBM3D(imgo,0.03);
% figure,imshow(image)