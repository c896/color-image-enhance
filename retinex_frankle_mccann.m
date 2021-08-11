function Retinex = retinex_frankle_mccann(L, nIterations)

% RETINEX_FRANKLE_McCANN: 
%         Computes the raw Retinex output from an intensity image, based on the
%         original model described in:
%         Frankle, J. and McCann, J., "Method and Apparatus for Lightness Imaging"
%         US Patent #4,384,336, May 17, 1983
% INPUT:  L           - logarithmic single-channel intensity image to be processed
%         nIterations - number of Retinex iterations
%
% OUTPUT: Retinex     - raw Retinex output
%
% NOTES:  - The input image is assumed to be logarithmic and in the range [0..1]
%         - To obtain the retinex "sensation" prediction, a look-up-table needs to
%         be applied to the raw retinex output
%         - For colour images, apply the algorithm individually for each channel
global  RR IP OP NP Maximum
RR = L;
Maximum = max(L(:));                                 % maximum color value in the image
[nrows, ncols] = size(L);

shift = 2^(fix(log2(min(nrows, ncols)))-1);          % initial shift

%这里程序提示数据类型不一致，所以我对Maximum强制取double类型，...
%...后面的OP等全局变量前的double也是这个原因；
OP = double(Maximum)*ones(nrows, ncols);                     % initialize Old Product

while (abs(shift) >= 1)
   for i = 1:nIterations
      CompareWith(0, shift);                         % horizontal step
      CompareWith(shift, 0);                         % vertical step
   end
   shift = -shift/2;                                 % update the shift
end
Retinex = NP;

function CompareWith(s_row, s_col)
global RR IP OP NP Maximum
IP = OP;
if (s_row + s_col > 0)
   IP((s_row+1):end, (s_col+1):end) = double(OP(1:(end-s_row), 1:(end-s_col))) + ...
   double(RR((s_row+1):end, (s_col+1):end) )- double(RR(1:(end-s_row), 1:(end-s_col)));
else
   IP(1:(end+s_row), 1:(end+s_col)) = double(OP((1-s_row):end, (1-s_col):end)) + ...
   double(RR(1:(end+s_row),1:(end+s_col))) - double(RR((1-s_row):end, (1-s_col):end));
end
IP(IP > Maximum) = Maximum;                          % The Reset operation
NP = (IP + OP)/2;                                    % average with the previous Old Product
OP = NP;                                             % get ready for the next comparison
