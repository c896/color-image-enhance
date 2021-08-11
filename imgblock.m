function [block,r_num,c_num,cnt] =imgblock(img,k,p,block_idx)
% img=im2double(img);
% ʵ��ͼ��ֿ�
% Inputs��
%        k: ���С
%        p: ���ƶ�����
%        lambda_2D: ������ֵ
%        delta: ������ֵ
%  Outputs:
%        block: ���صĿ�
%        transform_block: �任��Ŀ�
%        block2row_idx: ��������ͼ�������Ͻ��������Ӧ��ϵ
%        block2col_idx: ��������ͼ�������Ͻ��������Ӧ��ϵ
%

% clc; clear all; close all;
% path='C:\Users\CG\Desktop\5.28\night\yuvresult4\';
% DIR='G:\��Ƭ\6.8\1\';
% imgFiles = dir([DIR,'*.jpg']);%����ͼ��ĸ�ʽ  dir('')�г�ָ��Ŀ¼���������ļ��к��ļ�
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

% r_num �� c_num�ֱ��ʾ�к����Ͽ��Բɼ��Ŀ����Ŀ
r_num = floor((row-k)/p)+1; %512*512��Ϊ127
c_num = floor((col-k)/p)+1;
block = zeros(k,k,r_num*c_num*m);

% block2row_idx = zeros(1,r_num*c_num);
% block2col_idx = zeros(1,r_num*c_num);

cnt = 1;
for i = 0:r_num-1
    rs = 1+i*p;
    for j = 0:c_num-1  %���к��зֿ飨�������ٴ��ϵ��£�
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
