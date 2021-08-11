function [o, o_max, o_min] = rgb_to(img, colormode, inverse, o_max, o_min)

[w,h,~]=size(img);

if exist('colormode', 'var') && strcmp(colormode, 'opp') %strcmp()比较两个char型数组的字典序大小的
    % Forward
    A =[1/3 1/3 1/3; 0.5  0  -0.5; 0.25  -0.5  0.25];
    % Inverse
    B =[1 1 2/3;1 0 -4/3;1 -1 2/3];
else
    % YCbCr
%     A = [0.299, 0.587, 0.114; -0.168737, -0.331263, 0.5;  0.5,  -0.418688,  -0.081313];
%     B = [1.0000, 0.0000, 1.4020; 1.0000, -0.3441, -0.7141; 1.0000, 1.7720, 0.0000];
    A = [0.299, 0.587, 0.114; -0.147, -0.289, 0.437;  0.615,  -0.515,  -0.100];
    B = [1.0000, -0.0000, 1.1398; 1.0004, -0.3938, -0.5805; 0.9980, 2.0279, -0.0005];
end

if exist('inverse', 'var') && inverse
    % The inverse transform
    o = (reshape(img, [w * h, 3]) .* (o_max - o_min) + o_min) * B';
else
    % The color transform to YCbCr  Y=0.299R+0.587G+0.114B，Cb=0.564(B-Y)，Cr=0.713(R-Y)
    o = reshape(img, [w * h, 3]) * A';
    %o(:, 2:3) = o(:, 2:3) + 0.5;
    o_max = max(o, [], 1);
    o_min = min(o, [], 1);
    o = (o - o_min) ./ (o_max - o_min);
    %         scale = sum(A'.^2) ./ (o_max - o_min).^2;
end

%     o = reshape(o, [size(img, 1), size(img, 2), 3]);
end