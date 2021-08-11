function [block,r_num,c_num,cnt] =imgblock(img,k,p,block_idx)
% img=im2double(img);
% 实现图像分块
% Inputs：
%        k: 块大小
%        p: 块移动步长
%        lambda_2D: 收缩阈值
%        delta: 收缩阈值
%  Outputs:
%        block: 返回的块
%        transform_block: 变换后的块
%        block2row_idx: 块索引与图像块的左上角行坐标对应关系
%        block2col_idx: 块索引与图像块的左上角列坐标对应关系
%

% clc; clear all; close all;
% path='C:\Users\CG\Desktop\5.28\night\yuvresult4\';
% DIR='G:\照片\6.8\1\';
% imgFiles = dir([DIR,'*.jpg']);%输入图像的格式  dir('')列出指定目录下所有子文件夹和文件
% img = im2double(imread([DIR , imgFiles(1).name]));
% [row,col,m]=size(img);
% p=504;
% k=504;

[row,col,m]=size(img);
 
if ~exist('p','var')
    p=k; 
end
if ~exist('block_idx','var')
    block_idx=0;
end
% [row,col] = size(img);

% r_num 和 c_num分别表示行和列上可以采集的块的数目
r_num = floor((row-k)/p)+1; %512*512则为127
c_num = floor((col-k)/p)+1;
block = zeros(k,k,r_num*c_num*m);

% block2row_idx = zeros(1,r_num*c_num);
% block2col_idx = zeros(1,r_num*c_num);

cnt = 1;
for i = 0:r_num-1
    rs = 1+i*p;
    for j = 0:c_num-1  %先列后行分块（从左到右再从上到下）
        cs = 1+j*p;
        block(:,:,m*cnt-m+1:m*cnt) = img(rs:rs+k-1,cs:cs+k-1,:);
        if block_idx==1
            block2row_idx(cnt) = rs;
            block2col_idx(cnt) = cs;
        end      
        cnt = cnt+1;
    end
end
cnt=cnt-1;

% k=0;
% for i = 1 : r_num
%     for j = 1 : c_num
%         k = k + 1;
%         subplot(r_num,c_num, k);
%         imshow(block(:,:,m*k-m+1:m*k));      
%     end
% end
