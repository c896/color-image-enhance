% By lyqmath
% DLUT School of Mathematical Sciences 2008
% BLOG：http://blog.sina.com.cn/lyqmath

function [PSNR, MSE] = psnr(X, Y)
% 计算峰值信噪比PSNR、均方根误差MSE
% 如果输入Y为空，则视为X与其本身来计算PSNR、MSE

if nargin<2
    D = X;
else
    if any(size(X)~=size(Y))
        error('The input size is not equal to each other!');
    end
    D = X-Y;
end
MSE = sum(D(:).*D(:))/numel(X);%prod(size(X));  %B = prod(A) 将A矩阵不同维的元素的乘积返回到矩阵B
PSNR = 10*log10(255^2/MSE);
