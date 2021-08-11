%--------------相对于前3个版本 采用分块处理，解决了大尺寸图片分离困难耗时的问题- 采用块的img_max img_min还原RGB--------%
clc; clear all; close all;

path='C:\Users\CG\Desktop\5.28\night\yuvresult4\';
DIR='G:\照片\6.8\1\';
imgFiles = dir([DIR,'*.jpg']);%输入图像的格式  dir('')列出指定目录下所有子文件夹和文件

% I_orig = im2double(imread('Lena512.png'));
% [I_origyuv, img_max, img_min] = rgb_to(I_orig,colorspace);
% I_origyuv = reshape(I_origyuv, [width, height, 3]);

Img = im2double(imread([DIR , imgFiles(1).name]));
[width,height,m]=size(Img);
figure;
imshow(Img); title('待分离噪图');
colorspace=' ';
r_size = 504;  %块大小
% [blockorg,r_num,c_num,cnt] =imgblock(Img,r_size); %分块
[~,r_num,c_num,cnt] =imgblock(Img,r_size); %分块
[~, img_max, img_min] = rgb_to(Img,colorspace); %全局img_max, img_min

% r_num = floor((width-r_size)/r_size)+1; %512*512则为127
% c_num = floor((height-r_size)/r_size)+1;
% cnt=r_num*c_num;

tic;

% [N,~]= size(imgFiles);
N=6;
N1=width*height;
N2=r_size*r_size;
sample1 = zeros(N,N2);
sample2 = zeros(N,N2);
sample3 = zeros(N,N2);

Sample1 = zeros(N,N1);
Sample2 = zeros(N,N1);
Sample3 = zeros(N,N1);

% Ic2=zeros(r_size,r_size,m);
% Ic3=zeros(r_size,r_size,m);
y_est = zeros(r_size,r_size,cnt*3);
Xmean=zeros(N,m);

for i=1:cnt %cnt
    Ic=zeros(r_size,r_size,m);
    for j = 1:N
        %         if i==1
        I=im2double(imread([DIR ,imgFiles(j).name]));  %此处重复读取了cnt次
        
        I = rgb_to(I,colorspace);
        Sample1(j,:)= I(:,1)';
        Sample2(j,:)= I(:,2)';
        Sample3(j,:)= I(:,3)';
        I = reshape(I, [width, height, 3]);
        [block,~,~,~] =imgblock(I,r_size);
        %         end
        Ic=Ic+block(:,:,m*i-m+1:m*i);
        %         Ic2=Ic2+block2(:,:,m*i-m+2);
        %         Ic3=Ic3+block3(:,:,m*i-m+3);
        sample1(j,:)= reshape(block(:,:,m*i-m+1),1,[]);
        sample2(j,:)= reshape(block(:,:,m*i-m+2),1,[]);
        sample3(j,:)= reshape(block(:,:,m*i-m+3),1,[]);
        if j==N
            for k=1:m
                x=eval(['Sample',num2str(k)]);
                Xmean(:,k)=mean(x,2);
            end
        end
    end
    Ic=Ic./N;
    %     Ic2=Ic2./N;
    %     Ic3=Ic3./N;
    %     Ic=cat(3,Ic1,Ic2,Ic3);
    %     [Wwasobi, Winit, ISR,eval(['signals',num2str(i)])]= iwasobi(sample(),1,0.99);
    %-----------------wasobi盲分离----------------------%
    % [Wwasobi, ~, ~,signals1]= iwasobi(sample1(),1,0.99);
    %     signals = fastica(sample(),'numOfIC',1);
    
    %     [~, img_max, img_min] = rgb_to(blockorg(:,:,m*i-m+1:m*i),colorspace);% 局部块img_max, img_min
    y_est(:,:,3*i-2:3*i)=yuvwasobi(sample1,sample2,sample3,Ic,r_size,i, Xmean,img_max, img_min,path);
    
end
clear Sample1
clear Sample2
clear Sample3
clear block
%psnr_noise = 10*log10(peak^2/(mean((img11(:)-nimg1(:)).^2)));
img_out=zeros(width,height,3);
num=1;
for i = 0:r_num-1
    rs = 1+i*r_size;
    for j = 0:c_num-1  %先列后行分块（从左到右再从上到下）
        cs = 1+j*r_size;
        img_out(rs:rs+r_size-1,cs:cs+r_size-1,:)=y_est(:,:,3*num-2:3*num);
        num = num+1;
    end
end
figure;
imshow(img_out);title('yuv最终分离去噪图');
imwrite(img_out,[path,'yuv最终分离去噪图','.jpg']);

toc

