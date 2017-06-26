N=127; %number of points of one pinwheel
% delta_x=1/N;
%sample points range from [0 to 1-delta_x]
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
X_rt=(0:1/(N):1-(1/N));
Y_rt=(0:1/(N):1-(1/N));
% shift the coordinats for taking the tangent,
%let the pinwheel center be @ (0,0).
Xc=X_rt-0.5;  
Yc=fliplr((Y_rt-0.5));
% initialize the right-top pinwheel matrix
%assign each coordinate a theta value
[X Y]= meshgrid(Xc,Yc);
A_rt=(atan2d(Y,X)+180);
figure;
imagesc(Xc,Yc,A_rt);
colorbar;
axis square;

%==========================================================================
% SETP 2: Compute the left-top pinwheel matrix
% flip the right-top pinwheel matrix along the y axis 
%==========================================================================
size_rt=size(A_rt);
col=ones(size_rt(1),1).*0.5;
y_value=Yc';
right_most_col=atan2d(y_value,col)+180;
temp_A_lt=[A_rt right_most_col];
A_lt=fliplr(temp_A_lt);
size_lt=size(A_lt);
% dicard the left-most column to avoid repetition of the y-axis when
% fliping
A_lt=A_lt(1:size_lt(1),1:(size_lt(2)-1)); 

%==========================================================================
%STEP 3: compute the left-bottom pinwheel matrix
% flip the left-top pinwheel matrix along the x-axis
%==========================================================================
row=ones(1,size_rt(2)+1).*0.5;
x_value=[Xc 0.5];
top_extra_row=atan2d(row,x_value)+180;
temp_A_lb=fliplr([top_extra_row; temp_A_lt]);
size_lb=size(temp_A_lb);
A_lb=flipud(temp_A_lb);
% dicard the top-most row to avoid repetition of the x-axis when
% fliping
A_lb=A_lb(2:size_lb(1),1:size_lb(2)-1);

%==========================================================================
%Step 4: Compute the right-bottom pinwheel matrix
%flip the right-top pinwheel matrix alone x-axis
%==========================================================================
row_rb=ones(1,size_rt(2)).*0.5;
x_value_rb=[Xc];
top_extra_row_rb=atan2d(row_rb,x_value_rb)+180;
temp_A_rb=flipud([top_extra_row_rb; A_rt]);
size_rb= size(temp_A_rb);
% dicard the top-most row to avoid repetition of the x-axis when
% fliping
A_rb=temp_A_rb(2:size_rb(1),1:size_rb(2));
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
%{
total_size=size(theta);
phi=zeros(total_size);
m=1;
for k=total_size(1):-1:1
phi(m,:)=0.5*theta(k,:);
m=m+1;
end
%}
phi=theta.*0.5;
X_plot=[((X_rt)-1) X_rt];
Y_plot=[(Y_rt-1) Y_rt];
% for ploting the pinwheel, we have N points and we need N intervals.
% in order to have N intervals, we need to copy the bottom row to the top 
% to represent the row corresponding to y=1 
% and copy the left-most column to the right to represent the column
% corresponding to x=1.
% this is for plotting ONLY!
phi_plot=[phi(N*2,:);phi(:,:)];
phi_plot=[phi_plot(:,:),phi_plot(:,1)];
X_axis=[X_plot,1];
Y_axis=[Y_plot,1];
figure;imagesc(X_axis,Y_axis,phi_plot);
colormap('hsv')
axis square;
xlabel('x [mm]')
ylabel('y[mm]')
set(gca, 'YDir', 'normal');
title('OP Angle Contour')
colorbar();
figure;
contourf(X_axis,Y_axis,phi_plot,'showtext','on');
colormap('jet');
xlabel('x(mm)');
ylabel('y(mm)');
title('OP Angle Contour')
axis square;
colorbar;
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
%Ax=sqrt(0.5*(1-cos(deg2rad(theta))));
%Bx=sqrt(0.5*(1+cos(deg2rad(theta))));
Ax=sin(deg2rad(phi));
Bx=cos(deg2rad(phi));
figure;imagesc(X_plot,Y_plot,Ax);
axis square;
colormap('jet');
xlabel('x [mm]')
ylabel('y[mm]')
set(gca, 'YDir', 'normal');
title('Contour plot of Ax');
set(gca,'Box','off');   
axesPosition = get(gca,'Position');        
hNewAxes = axes('Position',axesPosition,...
                'Color','none',...          
                'YLim',[0 253],...           
                'XLim',[0 253],...  
                'XAxisLocation','top',...
                'YAxisLocation','right',... 
                'Box','off');               
ylabel(hNewAxes,'Sample index in Y Direction');
xlabel(hNewAxes,'Sample index in X Direction');
colorbar;
figure;imagesc(X_plot,Y_plot,Bx);
colormap('jet')
xlabel('x [mm]')
ylabel('y[mm]')
set(gca, 'YDir', 'normal');
title('Contour plot of Bx');
xlim([-1.0, 0.992125984251969]);
ylim([-1.0, 0.992125984251969]);
axis square;
colorbar;
set(gca,'Box','off');  
axesPosition = get(gca,'Position');        
hNewAxes = axes('Position',axesPosition,...
                'Color','none',...          
                'YLim',[0 253],...            
                'XLim',[0 253],...  
                'XAxisLocation','top',...
                'YAxisLocation','right',...  
                'Box','off');               
ylabel(hNewAxes,'Sample index in Y Direction');
xlabel(hNewAxes,'Sample index in X Direction');
axis square;

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
title('Magnitude plot of the DFT of a(x,y)'); 
axis square;
axis([-20 20 -20 20]);
xlabel('Kx(mm^-^1)');
ylabel('Ky(mm^-^1)');
colormap('pink');
colorbar;
caxis([0 4.3e4]);
figure,
imagesc(Fx,Fy,fftshift(mag_Bx));
title('Magnitude plot of the DFT of b(x,y)'); 
axis square;
axis([-20 20 -20 20]);
xlabel('Kx(mm^-^1)');
ylabel('Ky(mm^-^1');
colormap('pink');
colormap(flipud(colormap));
colorbar;
caxis([0 4.3e4]);
%truncate Ax, Bx with the first few terms
filter=zeros(size_dft);
f_row_sp=((size_dft(1))/2+1)-6;
f_row_ep=((size_dft(1))/2+1)+6;
f_col_sp=((size_dft(2))/2+1)-3;
f_col_ep=((size_dft(2))/2+1)+3;
filter(f_row_sp:f_row_ep,f_col_sp:f_col_ep)=ones(13,7);
filter_Ax=(fftshift(dft_Ax)).*filter;
filter_Bx=(fftshift(dft_Bx)).*filter;
figure, 
imagesc(Fx,Fy,abs(filter_Ax)); 
axis square;
axis([-20 20 -20 20]);
xlabel('Kx(mm^-^1)');
ylabel('Ky(mm^-^1)');
colormap('pink');
colorbar;
caxis([0 4.3e4]);
title('Filtered DFT of a(x,y)'); 
figure, 
imagesc(Fx,Fy,abs(filter_Bx)); 
axis square;
axis([-20 20 -20 20]);
xlabel('Kx(mm^-^1)');
ylabel('Ky(mm^-^1)');
colormap('pink');
title('Filtered DFT of b(x,y)');
caxis([-10000 4.3e4]);
colorbar;
%idft transform back to spacial domain
filter_Ax=ifft2(ifftshift(filter_Ax));
filter_Bx=ifft2(ifftshift(filter_Bx));
filter_angle=atan2d(real(filter_Ax),real(filter_Bx));
filter_angle_pi=abs(filter_angle);

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
axis square;
xlabel('x(mm)');
ylabel('y(mm)');
colorbar;
figure;imagesc(X_plot,Y_plot,real(fa_plot));
colormap('hsv')
axis square;
xlabel('x [mm]')
ylabel('y[mm]')
set(gca, 'YDir', 'normal');
title('Reconstructed OP angle Plot');
colorbar();

