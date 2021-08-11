clear
close all
clc

% path='C:\Users\CG\Desktop\5.28\标准图\';
path1 ='C:\Users\CG\Desktop\5.28\noise\';
% % imageName='House256.jpg';
% imageName='Lena512.png';
% % imageName='Baboon512.png';
% % img=imread(strcat([path,imageName]));

[imageName, path] = uigetfile('*.*','待计算图像1');
img=im2double(imread([path, imageName]));

samplenum = 1;  %样本数量

%%
% randn('seed',0); %指定seed后，第一次调用rand()得到的结果是“确定的”，相当于给rand设定了一个startpoint，相同的seed，对应的startpoint相同
%RNG:https://ww2.mathworks.cn/help/matlab/ref/rng.html

% sigma = 25;  % Noise standard deviation. it should be in the same 
% S = 255;
% I_MAX = 255/S;
% sigma = sigma/S;
% z = img + sigma*randn(size(img));
%%
% y_noise_p = imnoise(img,'poisson');
% imshow(y_noise_p,[]); 
disp('imnoising started')
for i = 1:1:samplenum 
    y_noise_g = imnoise(img,'gaussian'); imshow(y_noise_g,[]); 
    y_noise_pg = imnoise(y_noise_g,'poisson');
    imwrite(y_noise_pg,[path1,'noise',num2str(i),imageName]);
    fprintf('生成%s\n',strcat('noise',num2str(i),imageName));% 显示正在处理的图像名 
end
figure,imshow(y_noise_pg,[]); 
title('泊松-高斯noised_image');

%%
% % img=imread(strcat([path,imageName]));
% % img=im2double(img);
% a=0.5^2;
% b=0.04^2;
% clipping_below=1;
% clipping_above=1;
% 
% if a == 0
%   z=img;
% else
%   chi = 1/a;
%   z=poissrnd(max(0,chi*img))/chi;
% %   imshow(z)
% end
% z=z+sqrt(b)*randn(size(img));
% 
% if clipping_above
%   z=min(z,1);
% end
% if clipping_below
%   z=max(0,z);
% end
% 
% figure,imshow(z,[]); 
% title('泊松-高斯noised_image');
% 
% J=imdivide(z,10);
% K=immultiply(K,10);
% 
% figure,
% subplot(121),imshow(J)
% subplot(122),imshow(K)
% % max(max(z))

%% 沈帆
% % img=im2double(img);
% % peaks = [1 2 5 10 20 30 60 120]; 
% peaks = [2]; % target peak values for the scaled image缩放图像的目标峰值
% sigmas = peaks/10;                % standard deviation of the Gaussian noise
% reps = 10;     
% pp=1;
% randn('seed',0);    % fixes seed of random noise
% rand('seed',0);
% 
% % mixed Poisson-Gaussian noise parameters:
% 
% peak = peaks(pp); % target peak value for the scaled image
% scaling = peak/max(img(:));
% y = img*scaling;
% 
% % Poisson scaling factor泊松比例因子
% alpha = 1;
% 
% % Gaussian component N(g,sigma^2)
% sigma = sigmas(pp);
% g = 0.0;
% % z1 = alpha*poissrnd(y); 
% % imshow(z1)
% % z2 = sigma*randn(size(y));  
% % imshow(z2)
% z = alpha*poissrnd(y) + sigma*randn(size(y)) + g;   
% 
% figure,imshow(z,[]); 

%%

disp('imnoising started')
for i = 1:1:samplenum 
    img_noise=imnoise(img,'gaussian',0,0.02); 
%     imwrite(img_noise,[path1,'noise',num2str(i),imageName]);
%     fprintf('生成%s\n',strcat('noise',num2str(i),imageName));% 显示正在处理的图像名 

end

figure(),
subplot(121),imshow(img)
subplot(122),imshow(img_noise)
% set(0,'defaultFigurePosition',[100,100,1000,500]);%修改图形图像位置的默认设置
% set(0,'defaultFigureColor',[1 1 1])%修改图形背景颜色的设置

% I_MAX=double(max(max(max(img))));
I_MAX=255;
PSNR = 10*log10(I_MAX^2/mean((img_noise(:)-img(:)).^2))


