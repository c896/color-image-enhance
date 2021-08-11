%-------------经过比较发现选用伽马增强power更好，人眼看着更柔和，其次是自适应直方图均衡化adapthisteq--------%
clc
clear
path='C:\Users\CG\Desktop\5.28\';
% I = imread('tire.tif');
I = im2double(imread('C:\Users\CG\Desktop\5.28\IMG_3651.jpg'));
f = 512 / max(size(I));
S = imresize(I,f);
%------------------------RGB通道-------------------------%
%%
% img=power(S,0.4);
% img=real(img);
% figure,imshow(img)
% imwrite(img,[path,'rgb增强图power','.jpg']);

%%
% tic

% q=32;
% Ir=histeq(S(:, :, 1),q);
% Ig=histeq(S(:, :, 2),q);
% Ib=histeq(S(:, :, 3),q);

%%
% Ir=adapthisteq(S(:,:,1),'NumTiles',[2 2],'clipLimit',0.05,'Distribution','uniform');
% Ig=adapthisteq(S(:,:,2),'NumTiles',[2 2],'clipLimit',0.05,'Distribution','uniform');
% Ib=adapthisteq(S(:,:,3),'NumTiles',[2 2],'clipLimit',0.05,'Distribution','uniform');

%%
% J = cat(3, Ir,Ig,Ib);
% toc
% figure;
% imshow(J);
% imwrite(J,[path,'rgb增强图histeq','.jpg']);
% ssimval = ssim(J,S)

%%
% J1 = adapthisteq(I(:,:,1),'clipLimit',0.02,'Distribution','rayleigh');
% J2 = adapthisteq(I(:,:,2),'clipLimit',0.02,'Distribution','rayleigh');
% J3 = adapthisteq(I(:,:,3),'clipLimit',0.02,'Distribution','rayleigh');
% J=cat(3,J1,J2,J3);
% imshowpair(I,J,'montage');
% title('Original Image (left) and Contrast Enhanced Image (right)')

%%
%----------------------YUV通道-----------------%
% [width,height,m]=size(S);
% colorspace=' ';
% [I_yuv, img_max, img_min] = rgb_to(S,colorspace);
% I_yuv = reshape(I_yuv, [width, height, 3]);
%%
% Img=power(I_yuv,0.4);
% % imshow(I_yuv);
% I_est = rgb_to(Img, colorspace, true, img_max, img_min);
% I_est=reshape(I_est, [width,height, 3]);
% imshow(I_est)
% imshowpair(S,I_est,'montage');
% imwrite(I_est,[path,'yuv增强图power','.jpg']);
%%
%--------------对比度受限的自适应直方图均衡化-----------------%
%'NumTiles',[8 8] 将图像分成 8 行 8 列的图块。 'uniform'平坦直方图,'rayleigh'钟形直方图,'exponential'曲线直方图

% I1 = adapthisteq(I_yuv(:,:,1),'NumTiles',[2 2],'clipLimit',0.1,'Distribution','uniform');
% I2 = adapthisteq(I_yuv(:,:,2),'NumTiles',[2 2],'clipLimit',0.1,'Distribution','uniform');
% I3 = adapthisteq(I_yuv(:,:,3),'NumTiles',[2 2],'clipLimit',0.1,'Distribution','uniform');

%%
%-----------直方图均衡化---------------%
% % histeq(I,n) 变换灰度图像 I，返回具有 n 个离散灰度级的灰度图像 J 。映射到 J 的 n 个灰度级中每个级别的像素个数大致相等，因此 J 的直方图大致平坦
% q=32;
% I1=histeq(I_yuv(:,:,1),q);
% I2=histeq(I_yuv(:,:,2),q);
% I3=histeq(I_yuv(:,:,3),q);

%%
%--------将输入强度图像的值映射到新值，以对输入数据中强度最低和最高的 1%（默认值）数据进行饱和处理，从而提高图像的对比度----------

% I1=imadjust(I_yuv(:,:,1),[0 1],[]);
% I2=imadjust(I_yuv(:,:,2),[0 1],[]);
% I3=imadjust(I_yuv(:,:,3),[0 1],[]);

%%
% Img=cat(3,I1,I2,I3);
% I_est = rgb_to(Img, colorspace, true, img_max, img_min);
% I_est=reshape(I_est, [width,height, 3]);
% imshow(I_est)
% imshowpair(S,I_est,'montage');
% imwrite(I_est,[path,'yuv增强图imadjust','.jpg']);

%%
%---------------索引图像tif-----------------%
% [X,map] = imread('shadow.tif'); %颜色图 map 是 double 类型的 256×3 矩阵
% shadow = ind2rgb(X,map); %将索引图像转换为 RGB 图像
% % disp(['Range of RGB image is [',num2str(min(RGB(:))),', ',num2str(max(RGB(:))),'].'])
% %检查 RGB 图像的值是否在 [0, 1] 范围内。
% shadow_lab = rgb2lab(shadow); %rgb转lab空间
% max_luminosity = 100;
% L = shadow_lab(:,:,1)/max_luminosity;
% 
% shadow_imadjust = shadow_lab;
% shadow_imadjust(:,:,1) = imadjust(L)*max_luminosity;
% shadow_imadjust = lab2rgb(shadow_imadjust);
% 
% shadow_histeq = shadow_lab;
% shadow_histeq(:,:,1) = histeq(L)*max_luminosity;
% shadow_histeq = lab2rgb(shadow_histeq);
% 
% shadow_adapthisteq = shadow_lab;
% shadow_adapthisteq(:,:,1) = adapthisteq(L)*max_luminosity;
% shadow_adapthisteq = lab2rgb(shadow_adapthisteq);

% figure
% montage({shadow,shadow_imadjust,shadow_histeq,shadow_adapthisteq},'Size',[1 4])
% title("Original Image and Enhanced Images using imadjust, histeq, and adapthisteq")