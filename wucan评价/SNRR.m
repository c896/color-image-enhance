function [snr,vaiance,MSE]=SNRR(original_signal,recoveed_signal)
%filename:snr.m
%该文件是实现计算SNR(信噪比)
%单位是dB
%Time:2004/4/4
%author:Y.R.Zheng
%lennum:%数据长度
%aver:%计算均值
%variance:%计算方差
% original_signal = rand(3,200);
% recoveed_signal = rand(3,200);

[m1,n1]= size(original_signal);
[m2,n2]= size(recoveed_signal);
if n1 ~= n2
     error( sprintf( ' length of signal no same \n'));
end 

ave_o     = original_signal(1:m2,1:n2)-mean(original_signal(1:m2,1:n2),2)*ones(1,n2);
vaiance   = std(ave_o, 0, 2);
ave_r     = recoveed_signal-mean(recoveed_signal,2)*ones(1,n2);
vaiance_r = std(recoveed_signal,0,2);
MSEpre    = diag(1./vaiance)*ave_o-diag(1./vaiance_r)*ave_r;
MSE       = sqrt(diag(MSEpre*MSEpre'))/n2;
snr       = 10*log10(vaiance.*vaiance./MSE);
% figure;
% subplot(2,1,1); plot(ave_o/vaiance, 'b');         axis tight;ylabel('source');
% subplot(2,1,2); plot(ave_r/vaiance_r, 'b');       axis tight;ylabel('Result ');
