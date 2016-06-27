N=129;
%{
pinwheel location
--------------------------
          y axis
            ^
     -      |      +
            |
------------|-------------> x axix
            |
     +      |      -      

---------------------------
'-' sign stands for clockwise rotation
'+' sign stands for counter clockwise rotation

%}
%==========================================================================
% STEP 1: Compute the right-top pinwheel matrix
% By taking the tangent of each coordinates
%==========================================================================
X_rt=(0:1/(N-1):1);
Y_rt=(0:1/(N-1):1);
Xc=X_rt-X_rt((N+1)/2);
Yc=(Y_rt-Y_rt((N+1)/2))*(-1);
A_rt=zeros(size(X_rt));
for i=1:N
for j=1:N
   % if(j<((N+1)/2))
    %    A(i,j)=rad2deg(0.5*atan((Yc(i))/Xc(j)))+90;
    %elseif(i>((N+1)/2)) & (j>=((N+1)/2))
    %    A(i,j)=rad2deg(0.5*atan((Yc(i))/Xc(j)))+180;
    %else
    %    A(i,j)=rad2deg(0.5*atan(Yc(i)/Xc(j)));
    %end
    A_rt(i,j)=rad2deg(myatan(Yc(i),Xc(j)));
end
end
%==========================================================================
% SETP 2: Compute the left-top pinwheel matrix
% flip the right-top pinwheel matrix along the y axis 
%==========================================================================
size_lt= size(A_rt);
A_lt=zeros(size_lt);
lt=1;
for i=size_lt:-1:1
  A_lt(:,lt)=A_rt(:,i);
  lt=lt+1;
end
A_lt=A_lt(1:size_lt,1:(size_lt-1));
%==========================================================================
%STEP 3: compute the left-bottom pinwheel matrix
% flip the left-top pinwheel matrix along the x-axis
%==========================================================================
size_lb= size(A_lt);
A_lb=zeros(size_lb);
lb=1;
for i=size_lb(1):-1:1
  A_lb(lb,:)=A_lt(i,:);
  lb=lb+1;
end
A_lb=A_lb(2:size_lb(1),1:size_lb(2));
%==========================================================================
%Step 4: Compute the right-bottom pinwheel matrix
%flip the right-top pinwheel matrix alone x-axis
%==========================================================================
size_rb= size(A_rt);
A_rb=zeros(size_rb);
rb=1;
for i=size_rb(1):-1:1
  A_rb(rb,:)=A_rt(i,:);
  rb=rb+1;
end
A_rb=A_rb(2:size_rb(1),1:size_rb(2));
%==========================================================================
%Step 5: construct the matrix of the whole unit cell
%By concatenating the left-top,right-top,left-bottom, right-bottom pinwheel
%matrix
%==========================================================================
A_left=vertcat(A_lt,A_lb);
A_right=vertcat(A_rt,A_rb);
A=horzcat(A_left,A_right);

%==========================================================================
%Plot the unit cell
%==========================================================================
s=size(A_plot);
Ap=zeros(s);
m=1;
for k=s(1):-1:1
Ap(m,:)=0.5*A(k,:);
m=m+1;
end

X_lt=(-1:(1/(N-1)):0);
s_lt=size(X_lt);
s_rt=size(X_rt);

Y_lb=(-1:(1/(N-1)):0);
s_lb=size(X_lt);
figure;
X_plot=[X_lt(1:s_lt(2)-1) X_rt];
Y_plot=[Y_lb(1:s_lb(2)-1) Y_rt];
contourf(X_plot,Y_plot,Ap,'showtext','on');
colormap('jet');



%
Ax=sqrt(0.5*(1-cos(deg2rad(A))));
Bx=sqrt(0.5*(1+cos(deg2rad(A))));
fft_Ax=fft2(Ax);
fft_Bx=fft2(Bx);
mag_Ax =abs(fftshift(fft_Ax));
mag_Bx=abs(fftshift(fft_Bx));
figure, colormap gray, 
imshow(mag_Ax), 
figure, colormap gray,
imshow(mag_Bx)

