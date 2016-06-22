N=5;
X=(0:1/(N-1):1);
Y=(0:1/(N-1):1);
Xc=X-X((N+1)/2);
Yc=(Y-Y((N+1)/2))*(-1);
A=zeros(size(X));
for i=1:N
for j=1:N
   % if(j<((N+1)/2))
    %    A(i,j)=rad2deg(0.5*atan((Yc(i))/Xc(j)))+90;
    %elseif(i>((N+1)/2)) & (j>=((N+1)/2))
    %    A(i,j)=rad2deg(0.5*atan((Yc(i))/Xc(j)))+180;
    %else
    %    A(i,j)=rad2deg(0.5*atan(Yc(i)/Xc(j)));
    %end
    A(i,j)=rad2deg(myatan(Yc(i),Xc(j)));
end
end

s=size(A);
Ap=zeros(size(A));
m=1;
for k=s(1):-1:1
Ap(m,:)=A(k,:);
m=m+1;
end
figure;
contourf(X,Y,Ap,'showtext','on');
colormap('jet');
ym=meshgrid(1:-1/(N-1):0)';
xm=meshgrid(0:1/(N-1):1);
syms x y f;
for i=1:N
for j=1:N
f(i,j)=(y-ym(i,j))-(tan(deg2rad(A(i,j))))*(x-xm(i,j));
end
end


