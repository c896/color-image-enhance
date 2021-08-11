function [DIR,path,xlspath]=Zu(z)

switch z
    case 0        
        %第0组
        DIR='D:\小论文\10.09\1楼梯-480-640\'; % 均值2 0.0096 2.4502
        path='D:\小论文\10.09\1result\';
%         xlspath='D:\小论文\10.09\1result\std\data1std.xlsx';
        xlspath='D:\小论文\6个场景数据\Scene0.xlsx';
    case 1
        %第一组
        DIR='D:\小论文\12.05\5湖面2-480-640\'; %
        path='D:\小论文\12.05\5result2\';
%         xlspath='D:\小论文\12.05\5result2\std\data5std.xlsx';
        xlspath='D:\小论文\6个场景数据\Scene1.xlsx';
    case 2
        %第二组
        DIR='D:\小论文\12.15\1楼梯口1-480-640\'; %
        path='D:\小论文\12.15\1result1\';
%         xlspath='D:\小论文\12.15\1result1\data7std.xlsx';  
        xlspath='D:\小论文\6个场景数据\Scene2.xlsx';        
    case 3
        %第三组
        DIR='D:\小论文\12.15\3过道\3-480-640\'; %
        path='D:\小论文\12.15\3过道\3result\';
%         xlspath='D:\小论文\12.15\3过道\3result\data9std.xlsx';
        xlspath='D:\小论文\6个场景数据\Scene3.xlsx';        
%         %第三组
%         DIR='D:\小论文\7.02\3书桌-480-640\'; %均值0.0388  9.886
%         path='D:\小论文\7.02\3result\';
% %         xlspath='D:\小论文\7.02\3result\std\data3std.xlsx';
%         xlspath='C:\Users\CG\Desktop\修改意见\6个场景数据\Scene3.xlsx';
    case 4 
        %第四组
        DIR='D:\小论文\9.28\2小道-480-640\'; %均值1 0.0179 4.5683
        path='D:\小论文\9.28\2result\';
%         xlspath='D:\小论文\9.28\2result\std\data2std.xlsx';
        xlspath='D:\小论文\6个场景数据\Scene4.xlsx';
    case 5
        %第五组
        DIR='D:\小论文\10.09\4学科楼-480-640\'; %均值1 0.0524 13.3742
        path='D:\小论文\10.09\4result\';
%         xlspath='D:\小论文\10.09\4result\std\data4std.xlsx';
        xlspath='D:\小论文\6个场景数据\Scene5.xlsx';
    case 6
        %第六组
        DIR='D:\小论文\12.05\6校门口-480-640\'; %
        path='D:\小论文\12.05\6result\';
%         xlspath='D:\小论文\12.05\6result\std\data6std.xlsx';
        xlspath='D:\小论文\6个场景数据\Scene6.xlsx';
    case 7
        %第七组
        DIR='D:\小论文\12.15\2大楼1-480-640\'; %
        path='D:\小论文\12.15\2result1\';
%         xlspath='D:\小论文\12.15\2result1\data8std.xlsx';
        xlspath='D:\小论文\6个场景数据\duoyu\data8std.xlsx';
end
