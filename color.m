img1=imread('C:\Users\CG\Desktop\5.28\result2\多通道分离层R.jpg');
img2=imread('C:\Users\CG\Desktop\5.28\result2\多通道分离层G.jpg');
img3=imread('C:\Users\CG\Desktop\5.28\result2\多通道分离层B.jpg');
% 
% img4=imread('C:\Users\CG\Desktop\5.28\result2\分离层R.jpg');
% img5=imread('C:\Users\CG\Desktop\5.28\result2\分离层G.jpg');
% img6=imread('C:\Users\CG\Desktop\5.28\result2\分离层B.jpg');

% 
% diff1=img4-img1;
% diffmean1=mean(max(diff1));
% 
% diff2=img5-img2;
% diffmean2=mean(max(diff2));
% 
% diff3=img6-img3;
% diffmean3=mean(max(diff3));
% 
% % img7=img1+round(diff1);
% % img8=img2+round(diff2);
% % img9=img3+round(diff3);

% imout=cat(3,img7,img8,img9);
% imshow(imout)
% 
% imout1=cat(3,img4,img5,img6);
% imshow(imout1)


%src为rgb图像，saturation为调节的饱和度值，调节范围为【-100，100】
% function Image_new = SaturationAdjustment(src,saturation)
% Image=src;
% Image=double(Image);
% R=Image(:,:,1);
% G=Image(:,:,2);
% B=Image(:,:,3);
saturation=0;
R=double(img1);
G=double(img2);
B=double(img3);

[row, col] = size(R);
R_new=R;
G_new=G;
B_new=B;
%%%% Increment, 饱和度调整增量（-100,100）photoshop的范围
Increment=saturation;

%换算成调整比率
Increment=Increment/100;

%利用HSL模式求得颜色的S和L
for i=1:row
    for j=1:col
        rgbMax=max(R(i,j),max(G(i,j),B(i,j)));
        rgbMin=min(R(i,j),min(G(i,j),B(i,j)));
        Delta=(rgbMax-rgbMin)/255;
   
        if(Delta==0)                              %如果delta=0，则饱和度S=0，所以不能调整饱和度
            continue;
        end
        value = (rgbMax + rgbMin)/255;
        L=value/2;                                %Lightness
        
        if(L<0.5)                                 %根据明度L计算饱和度S
            S=Delta/value;
        else
            S =Delta/(2 - value);
        end
        %具体的饱和度调整，Increment为饱和度增减量
        if (Increment>=0)
            if((Increment+S)>=1)
                alpha=S;
            else
                alpha=1-Increment;
            end
          alpha=1/alpha-1;
          R_new(i,j) = L*255 + (R(i,j) - L * 255) * (1+alpha);
          G_new(i,j) = L*255 + (G(i,j) - L * 255) * (1+alpha);
          B_new(i,j) = L*255 + (B(i,j) - L * 255) * (1+alpha); 
        end
    end
end     
Image_new(:,:,1)=R_new;
Image_new(:,:,2)=G_new;
Image_new(:,:,3)=B_new;
imshow(Image_new)

% end

