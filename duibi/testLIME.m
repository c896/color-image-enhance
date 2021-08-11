function rgb=testLIME(DIR,path,xlspath,zu)

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

Img=imread([DIR ,imgFiles(1).name]);
img=im2double(Img);
% figure,imshow(Img),title('原低照图')

% A=zeros(size(img));
for k=2:N
    A=im2double(imread([DIR ,imgFiles(k).name]));
    img=img+A;
%     sprintf('正在读取%s',imgFiles(k).name)
end
L=img./N;

%--------------------------------------------------------------
post = false; % Denoising?

para.lambda = .15; % Trade-off coefficient
% Although this parameter can perform well in a relatively large range,
% it should be tuned for different solvers and weighting strategies due to
% their difference in value scale.

% Typically, lambda for exact solver < for sped-up solver
% and using Strategy III < II < I
% ---> lambda = 0.15 is fine for SPED-UP SOLVER + STRATEGY III
% ......


para.sigma = 2; % Sigma for Strategy III
para.gamma = 0.7; %  Gamma Transformation on Illumination Map
para.solver = 1; % 1: Sped-up Solver; 2: Exact Solver
para.strategy = 3;% 1: Strategy I; 2: II; 3: III

%---------------------------------------------------------------
% tic
[I, T_ini,T_ref] = LIME(L,para);
time=toc

figure(1),imshowpair(L,I,'montage')
% figure(1);imshow(L);title('Input');
% figure(2);imshow(I);title('LIME');

%% Post Processing
if post
    YUV = rgb2ycbcr(I);
    Y = YUV(:,:,1);
    
    sigma_BM3D = 10;
    [~, Y_d] = BM3D(Y,Y,sigma_BM3D,'lc',0);
    
    I_d = ycbcr2rgb(cat(3,Y_d,YUV(:,:,2:3)));
    I_f = (I).*repmat(T_ref,[1,1,3])+I_d.*repmat(1-T_ref,[1,1,3]);
    
    figure(5);imshow(I_d);title('Denoised ');
    figure(6);imshow(I_f);title('Recomposed');
end

rgb=uint8(I*255);
imwrite(rgb,[path,'帧平均LIME480-640','.jpg']);

gray = rgb2gray(rgb);
figure(2),imhist(gray)
axis tight;
set(gca,'LooseInset',get(gca,'TightInset'))
set(gca,'looseInset',[0 0 0 0])

quality=testquality(rgb,1);

value=[quality time];
value=roundn(value,-4);  %保留几位小数 b=vpa(value,4)保留有效数字

mRowRange='R6:W6';
writematrix(value,xlspath,'Sheet',1,'Range',mRowRange) % xlswrite('D:\学习\小论文\data.xlsx',value,1,mRowRange);

