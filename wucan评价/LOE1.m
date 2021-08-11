function score=LOE1(X,Y)
X1=Illumination(X,'max_c');
Y1=Illumination(Y,'max_c');
[m,n]=size(X1);
RD=zeros(m,n);
U1=[];
U2=[];

for k=1:m
    for l=1:n
        
        for i=1:m
            for j=1:n
                if X1(k,l)>X1(i,j)
                    U1=1;
                else
                    U1=0;
                end
                if Y1(k,l)>Y1(i,j)
                    U2=1;
                else
                    U2=0;
                end
                RD(k,l)=xor(U1,U2);
            end
        end
        
    end
end

score=(1/m*n)*sum(sum(RD));
            
            