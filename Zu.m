function [DIR,path,xlspath]=Zu(z)

switch z
    case 0        
        %��0��
        DIR='D:\С����\10.09\1¥��-480-640\'; % ��ֵ2 0.0096 2.4502
        path='D:\С����\10.09\1result\';
%         xlspath='D:\С����\10.09\1result\std\data1std.xlsx';
        xlspath='D:\С����\6����������\Scene0.xlsx';
    case 1
        %��һ��
        DIR='D:\С����\12.05\5����2-480-640\'; %
        path='D:\С����\12.05\5result2\';
%         xlspath='D:\С����\12.05\5result2\std\data5std.xlsx';
        xlspath='D:\С����\6����������\Scene1.xlsx';
    case 2
        %�ڶ���
        DIR='D:\С����\12.15\1¥�ݿ�1-480-640\'; %
        path='D:\С����\12.15\1result1\';
%         xlspath='D:\С����\12.15\1result1\data7std.xlsx';  
        xlspath='D:\С����\6����������\Scene2.xlsx';        
    case 3
        %������
        DIR='D:\С����\12.15\3����\3-480-640\'; %
        path='D:\С����\12.15\3����\3result\';
%         xlspath='D:\С����\12.15\3����\3result\data9std.xlsx';
        xlspath='D:\С����\6����������\Scene3.xlsx';        
%         %������
%         DIR='D:\С����\7.02\3����-480-640\'; %��ֵ0.0388  9.886
%         path='D:\С����\7.02\3result\';
% %         xlspath='D:\С����\7.02\3result\std\data3std.xlsx';
%         xlspath='C:\Users\CG\Desktop\�޸����\6����������\Scene3.xlsx';
    case 4 
        %������
        DIR='D:\С����\9.28\2С��-480-640\'; %��ֵ1 0.0179 4.5683
        path='D:\С����\9.28\2result\';
%         xlspath='D:\С����\9.28\2result\std\data2std.xlsx';
        xlspath='D:\С����\6����������\Scene4.xlsx';
    case 5
        %������
        DIR='D:\С����\10.09\4ѧ��¥-480-640\'; %��ֵ1 0.0524 13.3742
        path='D:\С����\10.09\4result\';
%         xlspath='D:\С����\10.09\4result\std\data4std.xlsx';
        xlspath='D:\С����\6����������\Scene5.xlsx';
    case 6
        %������
        DIR='D:\С����\12.05\6У�ſ�-480-640\'; %
        path='D:\С����\12.05\6result\';
%         xlspath='D:\С����\12.05\6result\std\data6std.xlsx';
        xlspath='D:\С����\6����������\Scene6.xlsx';
    case 7
        %������
        DIR='D:\С����\12.15\2��¥1-480-640\'; %
        path='D:\С����\12.15\2result1\';
%         xlspath='D:\С����\12.15\2result1\data8std.xlsx';
        xlspath='D:\С����\6����������\duoyu\data8std.xlsx';
end
