N=127;
delta_t=0.2e-3; % time resolution in seconds;
fs=1/delta_t; %sampling frequency
%w_max=pi*fs;
w_max=fs*pi;
N_s=14001*5; % number of samples;
t_range=(0:delta_t:(N_s-1)*delta_t);
t=((-250*delta_t*1e3):delta_t*1e3:(250*delta_t*1e3)); % plot range  for 50ms
%phi=generate_OP_matrix;
%twenty_unitcell_plot=[phi(:,:),phi(:,:),phi(:,:),phi(:,:),phi(:,:)];
%twenty_unitcell_plot=[twenty_unitcell_plot(:,:);twenty_unitcell_plot(:,:);twenty_unitcell_plot(:,:);twenty_unitcell_plot(:,:);twenty_unitcell_plot(:,:)];
X_twenty_unitcell=(0:1/(N_s):10-(1/N_s))-3;
Y_twenty_unitcell=(0:1/(N_s):10-(1/N_s))-1;
[x,y]=meshgrid(-3:0.01:7,-1:0.01:9);
A=deg2rad(157.5);
x_0=1.75;
y_0=3.25;
%y_0=2.75;
x_1=2.25;
%m5=[1.6 4.55];

m5=[2.15 4.6];
m4=[0.7 4.5];
m3=[-0.3 4.89]; %good one
%m3=[-0.3 4.87];
m2=[-0.85 4.5];
m1=[-2.1 5]; %good one
%m1=[-2.1 4.9];
% get the orientation preference of m1-m5 , from the OP map contour.
%
m5_row_idx= floor((m5(2)+1)*N);
m5_col_idx=floor((m5(1)+1)*N);
m5_op=twenty_unitcell_plot(m5_row_idx,m5_col_idx)
m4_row_idx= floor((m4(2)+1)*N);
m4_col_idx=floor((m4(1)+1)*N);
m4_op=twenty_unitcell_plot(m4_row_idx,m4_col_idx)
m3_row_idx= floor((m3(2)+1)*N);
m3_col_idx=abs(floor((m3(1)+1)*N));
m3_op=twenty_unitcell_plot(m3_row_idx,m3_col_idx)
m2_row_idx= floor((m2(2)+1)*N);
m2_col_idx=floor((m2(1)+1)*N);
m2_op=twenty_unitcell_plot(m2_row_idx,m2_col_idx)
m1_row_idx= floor((m1(2)+1)*N);
m1_col_idx=abs(floor((m1(1)+1)*N));
m1_op=twenty_unitcell_plot(m1_row_idx,m1_col_idx)
f=exp(-0.5*(((((x-x_0).*cos(A)+(y-y_0).*sin(A)).^2)./7)+(((-(x-x_0).*sin(A)+(y-y_0).*cos(A)).^2)./0.2)));
%basic_fn=((0.5*(cos(pi*(x-x_0))+1)).^2).*((0.5*(cos(pi*(y-y_0))+1)).^2);
basic_fn=((0.5*(cos(pi*(x-x_0))+1))).*((0.5*(cos(pi*(y-y_0))+1)));
figure;
ax1=axes;
imagesc(X_twenty_unitcell,Y_twenty_unitcell,twenty_unitcell_plot);hold on;
plot([-3.0, 7.0], [1, 1],'k','LineStyle','--','LineWidth',0.5);
plot([1,1],[-1.0, 9.0],'k','LineStyle','--','LineWidth',0.5);
plot([3,3],[-1.0, 9.0],'k','LineStyle','--','LineWidth',0.5);
plot([-3.0, 7.0], [3, 3],'k','LineStyle','--','LineWidth',0.5);
plot([5,5],[-1.0, 9.0],'k','LineStyle','--','LineWidth',0.5);
plot([-3.0, 7.0], [5, 5],'k','LineStyle','--','LineWidth',0.5);
plot([7,7],[-1.0, 9.0],'k','LineStyle','--','LineWidth',0.5);
plot([-3.0, 9.0], [7, 7],'k','LineStyle','--','LineWidth',0.5);
plot([-1,-1],[-1.0, 9.0],'k','LineStyle','--','LineWidth',0.5);
plot(m1(1),m1(2),'marker','x',  'LineWidth',3, 'MarkerSize',10, 'MarkerEdgeColor','k');
text(m1(1),m1(2),'\bf M1','FontSize',24);
plot(m3(1),m3(2),'marker','x',  'LineWidth',3, 'MarkerSize',10, 'MarkerEdgeColor','k');
text(m3(1),m3(2),'\bf M3','FontSize',24);
plot(m5(1),m5(2),'marker','x',  'LineWidth',3, 'MarkerSize',10, 'MarkerEdgeColor','k');
text(m5(1),m5(2),'\bf M5','FontSize',24);
plot(m2(1),m2(2),'marker','x',  'LineWidth',3, 'MarkerSize',10, 'MarkerEdgeColor','k');
text(m2(1),m2(2),'\bf M2','FontSize',24);
plot(m4(1),m4(2),'marker','x',  'LineWidth',3, 'MarkerSize',10, 'MarkerEdgeColor','k');
text(m4(1),m4(2),'\bf M4','FontSize',24);
hold off;
axis square;
colormap(ax1,'hsv');
set(gca, 'YDir', 'normal');
view(2);
ax2=axes;
contour(x,y,f.*basic_fn,'LineWidth',1.2,'LevelStep',0.2);hold on;
%contour(x,y,f,'LineWidth',1.5,'LineStyle','-.');
plot(x_0,y_0,'marker','.',  'LineWidth',3, 'MarkerSize',10, 'MarkerEdgeColor','w');
text(x_0,y_0,'\bf s1','color','w','FontSize',24);

f=exp(-0.5*(((((x-x_1).*cos(A)+(y-y_0).*sin(A)).^2)./7)+(((-(x-x_1).*sin(A)+(y-y_0).*cos(A)).^2)./0.2)));
basic_fn=((0.5*(cos(pi*(x-x_1))+1))).*((0.5*(cos(pi*(y-y_0))+1)));
contour(x,y,f.*basic_fn,'LineWidth',1.2,'LineStyle','--','LevelStep',0.2);
plot(x_1,y_0,'marker','.',  'LineWidth',3, 'MarkerSize',10, 'MarkerEdgeColor','w');
text(x_1,y_0,'\bf s2','color','w','FontSize',24);
%contour(x,y,f,'LineWidth',1.5,'LineStyle','-.');
hold off;
colormap(ax2,flipud('gray'))
linkaxes([ax1,ax2])
% Hide the top axes
ax2.Visible = 'off';
ax2.XTick = [];
ax2.YTick = [];
axis([ax1 ax2], 'square');
xlabel(ax1,'x(mm)');
ylabel(ax1,'y(mm)');
cb1 = colorbar(ax1,'Position',[.01 .11 .0675 .815]);
cb2 = colorbar(ax2,'Position',[.90 .11 .0675 .815]);
title(ax1,'Strong responsed measurement points @ M1,M3,M5 with Stimuls:157.5 degree light bar; ');

%===============================================================================
%}
%plot temporal Correlation plot
s2=[x_1 y_0]*1e-3;%m source piont #2
s1=[x_0 y_0]*1e-3;%m  source point #1
CF_m1_m3=temporal_correlation_fn(s1,s2,m1*1e-3,m3*1e-3);
CF_m1_m3_max=temporal_correlation_fn(s1,s2,m3*1e-3,m3*1e-3);


%
CF_m1_m3_plot=[CF_m1_m3((length(CF_m1_m3)-249):length(CF_m1_m3)),CF_m1_m3(1:251)];
[M,I] = max(CF_m1_m3_plot(:));
[I_row, I_col] = ind2sub(size(CF_m1_m3_plot),I);
max_value=CF_m1_m3_plot(I_row, I_col);
[M,I] = min(CF_m1_m3_plot(:));
[I_row, I_col] = ind2sub(size(CF_m1_m3_plot),I);
min_value=CF_m1_m3_plot(I_row, I_col);
[O P]=max(CF_m1_m3_max(:));
[I_row, I_col] = ind2sub(size(CF_m1_m3_max),P);
max_value_CF_m1_m3=CF_m1_m3_max(I_row, I_col);
figure;
if max_value > max_value_CF_m1_m3
    plot(t,(CF_m1_m3_plot./max_value));
else
plot(t,(CF_m1_m3_plot./max_value_CF_m1_m3));
end
title('normalize to C(m1,m1,t),Time domain Correlation Function C(\bf m1,\bf m3,\tau), OP= 157, S1&S2 are located in parallel');
axis square;
grid on;
xlabel('t(ms)');
ylabel('C(\bf m1,\bf m3,\tau)'); 
legend('OP=157,m1&m3');

CF_m1_m5=temporal_correlation_fn(s1,s2,m1*1e-3,m5*1e-3);
CF_m1_m5_max=temporal_correlation_fn(s1,s2,m3*1e-3,m3*1e-3);
%
CF_m1_m5_plot=[CF_m1_m5((length(CF_m1_m5)-249):length(CF_m1_m5)),CF_m1_m5(1:251)];
[M,I] = max(CF_m1_m5_plot(:));
[I_row, I_col] = ind2sub(size(CF_m1_m5_plot),I);
max_value=CF_m1_m5_plot(I_row, I_col);
[M,I] = min(CF_m1_m5_plot(:));
[I_row, I_col] = ind2sub(size(CF_m1_m5_plot),I);
min_value=CF_m1_m5_plot(I_row, I_col);
[O P]=max(CF_m1_m5_max(:));
[I_row, I_col] = ind2sub(size(CF_m1_m5_max),P);
max_value_CF_m1_m5=CF_m1_m5_max(I_row, I_col);
figure;
if max_value > max_value_CF_m1_m5
    plot(t,(CF_m1_m5_plot./max_value));
else
plot(t,(CF_m1_m5_plot./max_value_CF_m1_m5));
end
title('normalize to C(m1,m1,t),Time domain Correlation Function C(\bf m1,\bf m5,\tau), OP= 157, S1&S2 are located in parallel');
axis square;
grid on;
xlabel('t(ms)');
ylabel('C(\bf m1,\bf m5,\tau)'); 
legend('OP=157,m1&m5');

CF_m3_m5=temporal_correlation_fn(s1,s2,m3*1e-3,m5*1e-3);
CF_m3_m5_max=temporal_correlation_fn(s1,s2,m3*1e-3,m3*1e-3);
%
CF_m3_m5_plot=[CF_m3_m5((length(CF_m3_m5)-249):length(CF_m3_m5)),CF_m3_m5(1:251)];
[M,I] = max(CF_m3_m5_plot(:));
[I_row, I_col] = ind2sub(size(CF_m3_m5_plot),I);
max_value=CF_m3_m5_plot(I_row, I_col);
[M,I] = min(CF_m3_m5_plot(:));
[I_row, I_col] = ind2sub(size(CF_m3_m5_plot),I);
min_value=CF_m3_m5_plot(I_row, I_col);
[O P]=max(CF_m3_m5_max(:));
[I_row, I_col] = ind2sub(size(CF_m3_m5_max),P);
max_value_CF_m3_m5=CF_m3_m5_max(I_row, I_col);
figure;
if max_value > max_value_CF_m3_m5
    plot(t,(CF_m3_m5_plot./max_value));
else
plot(t,(CF_m3_m5_plot./max_value_CF_m3_m5));
end
title('normalize to C(m1,m1,t),Time domain Correlation Function C(\bf m3,\bf m5,\tau), OP= 157, S1&S2 are located in parallel');
axis square;
grid on;
xlabel('t(ms)');
ylabel('C(\bf m3,\bf m5,\tau)'); 
legend('OP=157,m3&m5');

CF_m2_m4=temporal_correlation_fn(s1,s2,m2*1e-3,m4*1e-3);
CF_m2_m4_max=temporal_correlation_fn(s1,s2,m2*1e-3,m2*1e-3);
%
CF_m2_m4_plot=[CF_m2_m4((length(CF_m2_m4)-249):length(CF_m2_m4)),CF_m2_m4(1:251)];
[M,I] = max(CF_m2_m4_plot(:));
[I_row, I_col] = ind2sub(size(CF_m2_m4_plot),I);
max_value=CF_m2_m4_plot(I_row, I_col);
[M,I] = min(CF_m2_m4_plot(:));
[I_row, I_col] = ind2sub(size(CF_m2_m4_plot),I);
min_value=CF_m2_m4_plot(I_row, I_col);
[O P]=max(CF_m2_m4_max(:));
[I_row, I_col] = ind2sub(size(CF_m2_m4_max),P);
max_value_CF_m2_m4=CF_m2_m4_max(I_row, I_col);
figure;
if max_value > max_value_CF_m2_m4
    plot(t,(CF_m2_m4_plot./max_value));
else
plot(t,(CF_m2_m4_plot./max_value_CF_m2_m4));
end
title('normalize to C(m1,m1,t),Time domain Correlation Function C(\bf m2,\bf m4,\tau), OP= 157, S1&S2 are located in parallel');
axis square;
grid on;
xlabel('t(ms)');
ylabel('C(\bf m2,\bf m4,\tau)'); 
legend('OP=157,m2&m4');

s1_90=[-0.9e-3 2.5e-3];
s2_90=[0.9e-3 2.5e-3];
A=deg2rad(90);
x_0=s1_90(1)*1e3;
y_0=s1_90(2)*1e3;
%y_0=2.75;
x_1=s2_90(1)*1e3;
f=exp(-0.5*(((((x-x_0).*cos(A)+(y-y_0).*sin(A)).^2)./7)+(((-(x-x_0).*sin(A)+(y-y_0).*cos(A)).^2)./0.2)));
%basic_fn=((0.5*(cos(pi*(x-x_0))+1)).^2).*((0.5*(cos(pi*(y-y_0))+1)).^2);
basic_fn=((0.5*(cos(pi*(x-x_0))+1))).*((0.5*(cos(pi*(y-y_0))+1)));
figure;
ax1=axes;
imagesc(X_twenty_unitcell,Y_twenty_unitcell,twenty_unitcell_plot);hold on;
plot([-3.0, 7.0], [1, 1],'k','LineStyle','--','LineWidth',0.5);
plot([1,1],[-1.0, 9.0],'k','LineStyle','--','LineWidth',0.5);
plot([3,3],[-1.0, 9.0],'k','LineStyle','--','LineWidth',0.5);
plot([-3.0, 7.0], [3, 3],'k','LineStyle','--','LineWidth',0.5);
plot([5,5],[-1.0, 9.0],'k','LineStyle','--','LineWidth',0.5);
plot([-3.0, 7.0], [5, 5],'k','LineStyle','--','LineWidth',0.5);
plot([7,7],[-1.0, 9.0],'k','LineStyle','--','LineWidth',0.5);
plot([-3.0, 7.0], [7, 7],'k','LineStyle','--','LineWidth',0.5);
plot([-1,-1],[-1.0, 9.0],'k','LineStyle','--','LineWidth',0.5);
plot(m1(1),m1(2),'marker','x',  'LineWidth',3, 'MarkerSize',10, 'MarkerEdgeColor','k');
text(m1(1),m1(2),'\bf M1','FontSize',24);
plot(m3(1),m3(2),'marker','x',  'LineWidth',3, 'MarkerSize',10, 'MarkerEdgeColor','k');
text(m3(1),m3(2),'\bf M3','FontSize',24);
plot(m5(1),m5(2),'marker','x',  'LineWidth',3, 'MarkerSize',10, 'MarkerEdgeColor','k');
text(m5(1),m5(2),'\bf M5','FontSize',24);
plot(m2(1),m2(2),'marker','x',  'LineWidth',3, 'MarkerSize',10, 'MarkerEdgeColor','k');
text(m2(1),m2(2),'\bf M2','FontSize',24);
plot(m4(1),m4(2),'marker','x',  'LineWidth',3, 'MarkerSize',10, 'MarkerEdgeColor','k');
text(m4(1),m4(2),'\bf M4','FontSize',24);
hold off;
axis square;
colormap(ax1,'hsv');
set(gca, 'YDir', 'normal');
view(2);
ax2=axes;
contour(x,y,f.*basic_fn,'LineWidth',1.2,'LevelStep',0.2);hold on;
%contour(x,y,f,'LineWidth',1.5,'LineStyle','-.');
plot(x_0,y_0,'marker','.',  'LineWidth',3, 'MarkerSize',10, 'MarkerEdgeColor','w');
text(x_0,y_0,'\bf s1','color','w','FontSize',24);

f=exp(-0.5*(((((x-x_1).*cos(A)+(y-y_0).*sin(A)).^2)./7)+(((-(x-x_1).*sin(A)+(y-y_0).*cos(A)).^2)./0.2)));
basic_fn=((0.5*(cos(pi*(x-x_1))+1))).*((0.5*(cos(pi*(y-y_0))+1)));
contour(x,y,f.*basic_fn,'LineWidth',1.2,'LineStyle','--','LevelStep',0.2);
plot(x_1,y_0,'marker','.',  'LineWidth',3, 'MarkerSize',10, 'MarkerEdgeColor','w');
text(x_1,y_0,'\bf s2','color','w','FontSize',24);
%contour(x,y,f,'LineWidth',1.5,'LineStyle','-.');
hold off;
colormap(ax2,flipud('gray'))
linkaxes([ax1,ax2])
% Hide the top axes
ax2.Visible = 'off';
ax2.XTick = [];
ax2.YTick = [];
axis([ax1 ax2], 'square');
xlabel(ax1,'x(mm)');
ylabel(ax1,'y(mm)');
cb1 = colorbar(ax1,'Position',[.01 .11 .0675 .815]);
cb2 = colorbar(ax2,'Position',[.90 .11 .0675 .815]);
title(ax1,'Strong responsed measurement points @ M2,M4 with Stimuls:157.5 degree light bar; ');


CF_m2_m4=temporal_correlation_fn(s1_90,s2_90,m2*1e-3,m4*1e-3);
CF_m2_m4_max=temporal_correlation_fn(s1_90,s2_90,m2*1e-3,m2*1e-3);
%
CF_m2_m4_plot=[CF_m2_m4((length(CF_m2_m4)-249):length(CF_m2_m4)),CF_m2_m4(1:251)];
[M,I] = max(CF_m2_m4_plot(:));
[I_row, I_col] = ind2sub(size(CF_m2_m4_plot),I);
max_value=CF_m2_m4_plot(I_row, I_col);
[M,I] = min(CF_m2_m4_plot(:));
[I_row, I_col] = ind2sub(size(CF_m2_m4_plot),I);
min_value=CF_m2_m4_plot(I_row, I_col);
[O P]=max(CF_m2_m4_max(:));
[I_row, I_col] = ind2sub(size(CF_m2_m4_max),P);
max_value_CF_m2_m4=CF_m2_m4_max(I_row, I_col);
figure;
if max_value > max_value_CF_m2_m4
    plot(t,(CF_m2_m4_plot./max_value));
else
plot(t,(CF_m2_m4_plot./max_value_CF_m2_m4));
end
title('normalize to C(m1,m1,t),Time domain Correlation Function C(\bf m2,\bf m4,\tau), OP= 90, S1&S2 are located in parallel');
axis square;
grid on;
xlabel('t(ms)');
ylabel('C(\bf m2,\bf m4,\tau)'); 
legend('OP=90,m2&m4');