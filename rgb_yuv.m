function o = rgb_yuv(img, colormode, inverse)

[w,h,~]=size(img);
% colorspace = ' ';
if exist('colormode', 'var') && strcmp(colormode, 'opp') %strcmp()比较两个char型数组的字典序大小的
    % Forward
    A =[1/3 1/3 1/3; 0.5  0  -0.5; 0.25  -0.5  0.25];
    % Inverse
    B =[1 1 2/3;1 0 -4/3;1 -1 2/3];
else
    % YCbCr
      A = [0.299, 0.587, 0.114; -0.168737, -0.331263, 0.5;  0.5,  -0.418688,  -0.081313]; %原
      B = [1.0000, 0.0000, 1.4020; 1.0000, -0.3441, -0.7141; 1.0000, 1.7720, 0.0000];
%     A = [0.299, 0.587, 0.114; -0.147, -0.289, 0.437;  0.615,  -0.515,  -0.100];
%     B = [1.0000, -0.0000, 1.1398; 1.0004, -0.3938, -0.5805; 0.9980, 2.0279, -0.0005];
end

if exist('inverse', 'var') && inverse
    % The inverse transform
    o = reshape(img, [w * h, 3]) * B';
else
    % The color transform to YCbCr  Y=0.299R+0.587G+0.114B，Cb=0.564(B-Y)，Cr=0.713(R-Y)
    o = reshape(img, [w * h, 3]) * A';
end

%     o = reshape(o, [size(img, 1), size(img, 2), 3]);
end