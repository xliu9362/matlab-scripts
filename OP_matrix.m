N=129; %number of points
%{
pinwheel location
--------------------------
          y axis
            ^
 --  -   |  |  |   +   --
            |
------------|-------------> x axix
            |
 --  +   |  |  |   -   --   

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
theta_left=vertcat(A_lt,A_lb);
theta_right=vertcat(A_rt,A_rb);
theta=horzcat(theta_left,theta_right);

%==========================================================================
%Plot the unit cell
%==========================================================================
total_size=size(theta);
phi=zeros(total_size);
m=1;
for k=total_size(1):-1:1
phi(m,:)=0.5*theta(k,:);
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
contourf(X_plot,Y_plot,phi,'showtext','on');
colormap('jet');

%NFFT = 2^nextpow2(s(1));
%==========================================================================
% partial direvative d/dx detects the vertical bar, 
% d/dy detects the horizontal bar.
% Op angle can also be dexcribed as : Phi(x,y)= a(x,y)d/dx +b(x,y)d/dy
% 1) find the matrix of A(x.y) and B(x,y)  
% 2) perform 2D Discrete fourier transform on Ax and Bx
% 3) truncate Ax, Bx with first few terms that has large magnitude
% 4) perform idft on truncated Ax,Bx
%==========================================================================
%find Ax,Bx from orientation angle theta.
Ax=sqrt(0.5*(1-cos(deg2rad(theta))));
Bx=sqrt(0.5*(1+cos(deg2rad(theta))));
%2d dft on Ax, Bx
dft_Ax=mydft(Ax);
dft_Bx=mydft(Bx);
%magnitude of dft_Ax,dft_Bx
mag_Ax =abs(dft_Ax);
mag_Bx=abs(dft_Bx);
%plot the magnitude of dft_Ax, and dft_Bx 
size_dft=size(dft_Ax);
Fx=((1-(size_dft(1)+1)/2):1:(size_dft(1)-((size_dft(1)+1)/2)));
Fy=(size_dft(1)-((size_dft(1)+1)/2)):-1:((1-(size_dft(1)+1)/2));
figure, 
imagesc(Fx,Fy,fftshift(mag_Ax)); 
title('Magnitude of DFT of a(x,y)'); 

figure,
imagesc(Fx,Fy,fftshift(mag_Bx));
title('Magnitude of DFT of b(x,y)'); 
%magnitude of Kx
Kx=mag_Ax(1,1:10);
figure,
stem((0:9),Kx);
hold on;
stem((0:9),mag_Ax(2,1:10));
stem((0:9),mag_Ax(3,1:10));
stem((0:9),mag_Ax(4,1:10));
stem((0:9),mag_Ax(5,1:10));
hold off;
title('Magnitude of (Kx,Ky=0,1,2,3,4)')
legend('Kx,Ky=0','Kx,Ky=1','Kx,Ky=2','Kx,Ky=3','Kx,Ky=4');
%magnitude of Ky
Ky=mag_Ax(1:10,1);
figure,
stem((0:9),Ky);
hold on;
stem((0:9),mag_Ax(1:10,2));
stem((0:9),mag_Ax(1:10,3));
stem((0:9),mag_Ax(1:10,4));
stem((0:9),mag_Ax(1:10,5));
hold off;
title('Magnitude of (Kx=0,1,2,3,4,Ky)')
legend('Kx=0,Ky','Kx=1,Ky','Kx=2,Ky','Kx=3,Ky','Kx=4,Ky');
%truncate Ax, Bx with the first few terms
filter=zeros(size_dft);
f_row_sp=((size_dft(1)+1)/2)-4;
f_row_ep=((size_dft(1)+1)/2)+4;
f_col_sp=((size_dft(2)+1)/2)-5;
f_col_ep=((size_dft(2)+1)/2)+5;
filter(f_row_sp:f_row_ep,f_col_sp:f_col_ep)=ones(9,11);
filter_Ax=(fftshift(dft_Ax)).*filter;
filter_Bx=(fftshift(dft_Bx)).*filter;
%figure
%imagesc(Fx,Fy,abs(filter_Ax));
%idft transform back to spacial domain
filter_Ax=ifft2(ifftshift(filter_Ax));
filter_Bx=ifft2(ifftshift(filter_Bx));
filter_angle=atan2d(real(filter_Ax),real(filter_Bx)); 
% need to add pi to the bottom half of each pinwheel
start_p=(total_size(1)+3)/4;
end_p=(total_size(1)+1)-start_p;
filter_angle_pi=filter_angle;
filter_angle_pi(start_p:end_p,:)=180-filter_angle(start_p:end_p,:);
% plot the Op angle after filtering
s=size(filter_angle_pi);
fa_plot=zeros(s);
m=1;
for k=s(1):-1:1
fa_plot(m,:)=filter_angle_pi(k,:);
m=m+1;
end
figure;
contourf(X_plot,Y_plot,real(fa_plot),'showtext','on');
colormap('jet');
title('Reconstructed OP angle Plot');